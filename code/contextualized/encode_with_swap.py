import json
from transformers import AutoTokenizer

def main(post2encoding_path, output_path, swap, tokenizer_name):
    
    # NOTE: code only works for swap words that consist of one subword (after tokenization)
     
    with open(post2encoding_path, 'r') as infile:
        post2encoding = json.load(infile)
    
    tknzr = AutoTokenizer.from_pretrained(tokenizer_name)
    recipient = tknzr.encode(swap[0], add_special_tokens=False)[0]
    donor = tknzr.encode(swap[1], add_special_tokens=False)[0]
    placeholder = tknzr.encode(tknzr.unk_token, add_special_tokens=False)[0]
    
    man_post2encoding = {}
    for post, encoding in post2encoding.items():
        if donor in encoding:
            man_post2encoding[post] = [x if x != donor else placeholder for x in encoding]
        if recipient in encoding:
            man_post2encoding[post] = [x if x != recipient else donor for x in encoding]
    
    with open(output_path, 'w') as outfile:
        json.dump(man_post2encoding, outfile)


if __name__ == '__main__':

    post2encoding_path = '[filepath to json-file that maps post ids to encodings]'
    output_path = '[filepath to json-file]'
    swap = ('guns', 'taxes') # the first will be replaced by the second
    tokenizer_name = '[name of model from HuggingFace transformers library]'
    
    main(post2encoding_path, output_path, swap, tokenizer_name)
        
