import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from text_utils import TextEncoder
from utils import encode_dataset_lm, flatten, iter_data_lm, find_trainable_variables, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path
from utils import load_pickle_dataset

import tokenization
from tokenization import load_idx_to_token
from tensorflow.contrib.training import HParams
import jieba

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

# GPT-2 version
def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

# GPT-2 version
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

# GPT-2 version
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

# GPT-2 version
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

# GPT-2 version
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

# GPT-2 version
def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

# GPT-2 version
def attn(x, scope, n_state, *, past, hparams, train=False):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        w = dropout(w, attn_pdrop, train)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        a = dropout(a, resid_pdrop, train)
        return a, present

# GPT-2 version
def mlp(x, scope, n_state, *, hparams, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

# GPT-2 version
def block(x, scope, *, past, hparams, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams, train=train)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams, train=train)
        x = x + m
        return x, present

# GPT-2 version
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

# GPT-2 version
def positions_for(tokens, past_length=0):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def embed_gpt1(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

# bert-embedding
def embed(X, we, use_one_hot=True, depth=None):
    if X.shape.ndims == 2:
        X = tf.expand_dims(X, axis=[-1])

    flat_X = tf.reshape(X, [-1])
    
    if use_one_hot:
        one_hot_X = tf.one_hot(flat_X, depth=depth)
        output = tf.matmul(one_hot_X, we)

    X_shape = shape_list(X)
    output = tf.reshape(output, X_shape[0:-1]+[X_shape[-1]*n_embd])
    return output

def model(X, M, hparams, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))

        # h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X))
        # h = tf.nn.embedding_lookup(wte, X) + tf.nn.embedding_lookup(wpe, positions_for(X))
        # wpe = dropout(wpe, embd_pdrop, train)
        # wte = dropout(wte, embd_pdrop, train)

        h = embed(X, wte, depth=n_vocab) + embed(positions_for(X), wpe, depth=n_ctx)
        h = dropout(h, embd_pdrop, train)
        # pX = positions_for(X)
        # print('shape of pX is\n', shape_list(pX))
        # print('shape of pX is--\n', pX.shape.ndims)
        # hp = embed(pX, wpe, depth=n_ctx)

        # h = hx + hp

        for layer in range(n_layer):
            h, _ = block(h, 'h%d'%layer, past=None, hparams=hparams)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd]) # B*T' x C 
        lm_logits = tf.matmul(lm_h, wte, transpose_b=True) # B*T' x V
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1]) # B x T'
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        if train:
            return lm_losses
        else:
            return lm_losses, tf.reshape(lm_logits, [shape_list(X)[0], shape_list(X)[1]-1, -1])

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, _xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            lm_losses = model(*_xs, hparams=hparams,train=True, reuse=do_reuse)
            train_loss = tf.reduce_mean(lm_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append(lm_losses)
    ops = [tf.concat(gpu_ops, 0)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, _xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            lm_losses, _ = model(*_xs, hparams=hparams, train=False, reuse=True)
            gpu_ops.append(lm_losses)
    ops = [tf.concat(gpu_ops, 0)]
    return ops

def transform_lm(sents):
    n_batch = len(sents)
    # max_len = get_max_length(sents)
    xmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    for i, sent in enumerate(sents):
        sent_len = min(len(sent), n_ctx)
        xmb[i, :sent_len] = sent[:n_ctx]
        mmb[i, :sent_len] = 1
    return xmb, mmb

# def get_max_length(sents):
#     max_len = max([len(sent) for sent in sents])
#     return max_len

def iter_apply_lm(Xs):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb in iter_data_lm(Xs, n_batch=n_batch_train, truncate=False, verbose=True):
        xmb, mmb = transform_lm(xmb)
        res = sess.run(eval_lm_loss, {X_valid:xmb, M_valid:mmb})
        results.append(res)
    return np.mean(results)

def log_lm():
    global best_score
    tr_cost = float(iter_apply_lm(train_data[:n_valid]))
    va_cost = float(iter_apply_lm(valid_data))
    tr_ppl = math.pow(math.e, tr_cost)
    va_ppl = math.pow(math.e, va_cost)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_ppl=tr_ppl, va_ppl=va_ppl)
    print('%d %d %.3f %.3f %.3f %.3f'%(n_epochs, n_updates, tr_cost, va_cost, tr_ppl, va_ppl))
    predict_lm()
    if submit:
        score = va_ppl
        # print('score is {}, best score is {}'.format(score, best_score))
        if score < best_score:
            best_score = score
            # save(os.path.join(save_dir, desc, 'best_params.jl'))
            save(os.path.join(save_dir, desc, 'model.ckpt'))


# def save(path):
#     ps = sess.run(params)
#     joblib.dump(ps, make_path(path))

def save(path):
    save_path = saver.save(sess, make_path(path))
    print('save the best model!')

argmax = lambda x:np.argmax(x, 1)

def predict_lm():
    x = [['[CLS]'],]
    for i in range(len(x)):
        x[i] = tokenizer.convert_tokens_to_ids(x[i])
    text = ''
    for i in range(n_ctx-1):
        xmb, mmb = transform_lm(x)
        logits = sess.run(eval_lm_logits, {X_valid:xmb, M_valid:mmb})
        logits = np.squeeze(logits, axis=0)
        logits = logits[i]
        idx = np.argmax(logits, 0)
        if idx_to_vocab[idx] == '[SEP]':
            print(text)
            return
        text += idx_to_vocab[idx]
        x[0] += [idx]
    print(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--train_data_path', type=str, default='data/')
    parser.add_argument('--valid_data_path', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_warmup', type=float, default=0.1)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--vocab_file', type=str, default=None, required=True)
    parser.add_argument('--phrase_token', action='store_true')
    
    args = parser.parse_args()
    print(args)
    hparams = default_hparams()
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    if phrase_token:
        tokenizer = tokenization.JiebaTokenizer(vocab_file=vocab_file)
    else:
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    n_vocab = len(tokenizer.vocab)
    idx_to_vocab = load_idx_to_token(vocab_file)
    
    # train_dataset = encode_dataset_lm(train_data_path, tokenizer)
    train_dataset = load_pickle_dataset(train_data_path)
    print('------------\ntrain_data is\n', train_dataset[:3])
    max_len = 256
    n_ctx = min(max([len(sent[:max_len]) for sent in train_dataset])+2, n_ctx)
    train_data = [[tokenizer.cls()] + sent[:max_len] + [tokenizer.eos()] for sent in train_dataset]
    # train_data = [tokenizer.convert_tokens_to_ids(sent) for sent in train_dataset]
    random.shuffle(train_data)

    print('----------\nn_ctx is {}\nn_vocab is {}\n----------'.format(n_ctx, n_vocab))
    
    # valid_dataset = encode_dataset_lm(valid_data_path, tokenizer, is_train=False)

    valid_dataset = load_pickle_dataset(valid_data_path)
    valid_data = [[tokenizer.cls()] + sent[:max_len] + [tokenizer.eos()] for sent in valid_dataset]
    # valid_data = [tokenizer.convert_tokens_to_ids(sent) for sent in valid_dataset]

    n_train = len(train_data)
    n_valid = len(valid_data)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter

    hparams.n_vocab = n_vocab
    hparams.n_ctx = n_ctx
    hparams.n_embd = n_embd
    hparams.n_head = n_head
    hparams.n_layer = n_layer

    print('--------\nn_train={}, n_valid={}, n_batch_train={}, n_updates_total={}\n--------\n'.format(n_train, n_valid, n_batch_train, n_updates_total))

    X_train = tf.placeholder(tf.int32, [n_batch_train, n_ctx])
    M_train = tf.placeholder(tf.float32, [n_batch_train, n_ctx])
    X_valid = tf.placeholder(tf.int32, [None, n_ctx])
    M_valid = tf.placeholder(tf.float32, [None, n_ctx])

    # X_train = tf.placeholder(tf.int32, [n_batch_train, None])
    # M_train = tf.placeholder(tf.float32, [n_batch_train, None])
    # X_valid = tf.placeholder(tf.int32, [None, None])
    # M_valid = tf.placeholder(tf.float32, [None, None])

    train, lm_losses = mgpu_train(X_train, M_train)
    lm_loss = tf.reduce_mean(lm_losses)

    saver = tf.train.Saver()
    
    params = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    eval_lm_losses, eval_lm_logits = model(X_valid, M_valid, hparams=hparams, train=False, reuse=True)
    eval_lm_loss = tf.reduce_mean(eval_lm_losses)

    n_updates = 0
    n_epochs = 0
    best_score = 35000
    log_lm()
    for i in range(n_iter):
        for xmb in iter_data_lm(shuffle(train_data, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            xmb, mmb = transform_lm(xmb)
            cost, _ = sess.run([lm_loss, train], {X_train:xmb, M_train:mmb})
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log_lm()
        n_epochs += 1
        log_lm()
    # predict_lm()
    