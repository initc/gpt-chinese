import argparse
import tokenization
import pickle
from tqdm import tqdm
import os

def encoder_to_file(text_file, tokenizer, save_file):
    with oepn(text_file, "rb") as f:
        test_file = pickle.load(f)
        context_ids = []
        for data in tqdm(test_file):
            ids = []
            for d in data:
                ids.extend(tokenizer.convert_text_to_ids(d))
            context_ids.append(ids)
    with open(save_file, "wb") as f:
        pickle.dump(context_ids, f)
    return len(context_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--valid-data', type=str)
    parser.add_argument('--save-dir', type=str, default='data/')
    parser.add_argument('--vocab_file', type=str, default=None, required=True)
    args = parser.parse_args()

    tokenizer = tokenization.JiebaTokenizer(vocab_file=vocab_file)
    save_train_file = os.path.join(args.save_dir, "train-ids.pk")
    lens_ids = encoder_to_file(args.train_data, tokenizer, save_train_file)
    print('|| data {} :: write data into :: {}'.format(lens_ids, save_train_file))
    for i, valid_file in enumerate(args.valid_data.split(",")):
        save_valid_file = os.path.join(args.save_dir, "valid-ids-{}.pk".format(i))
        lens_ids = encoder_to_file(valid_file, tokenizer, save_valid_file)
        print('|| data {} :: write data into :: {}'.format(lens_ids, save_valid_file))
    
    






