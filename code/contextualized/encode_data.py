import json, re
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def preprocess(text, lower, reddit):

    if lower:
        text = text.lower()
    if reddit:
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')

    #remove urls
    text = re.sub(r"https?:\/\/[^\s]+", '', text) 

    return text


def encode_posts(data_path, tknzr, lower, reddit):

    data = pd.read_csv(data_path)
    post_ids = data['id']
    texts = data['text'].astype(str)

    encodings = [tknzr.encode(preprocess(t, lower, reddit), truncation=True) for t in tqdm(texts)] #for batched processing: set padding='max_length'
    post2encoding = {post_id: encoding for post_id, encoding in zip(post_ids, encodings)}

    return post2encoding


def encode_targets(targets_path, tknzr):

    with open(targets_path, 'r') as infile:
        targets = [x.replace('\n', '') for x in infile.readlines()]
        
    target2encoding = {t: tknzr.encode(t, add_special_tokens=False) for t in targets}

    return target2encoding


def main(input_path, output_path, model_name, datatype, lower=False, reddit=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if datatype == 'posts':
        encoding_dict = encode_posts(input_path, tokenizer, lower, reddit)
    elif datatype == 'targets':
        encoding_dict = encode_targets(input_path, tokenizer)

    with open(output_path, 'w') as outfile:
        json.dump(encoding_dict, outfile)


if __name__ == '__main__':
    
    input_path = '[path to csv-file with "text" and "id" columns or txt-file with one target word per line]'
    output_path = '[filepath to json-file]'
    model_name = '[name of model from HuggingFace transformers library]'
    datatype = 'posts' # or 'targets'
    reddit = False # or True
    lower = False # or True
    
    main(input_path, output_path, model_name, datatype, lower=lower, reddit=reddit)
    