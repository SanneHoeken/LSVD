import torch, json, pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict
from tqdm import tqdm

def get_topk_substitutes(sent, target_idx, model, tokenizer, k):

    mask_id = tokenizer.mask_token_id
    sent[target_idx] = mask_id
    inputs = torch.tensor([sent])

    with torch.no_grad():
        logits = model(inputs).logits

    softmax = torch.softmax(logits, dim=-1).squeeze_()
    topk_probs, topk_indices = torch.topk(softmax, k, sorted=True, dim=-1)
    topk_indices = torch.transpose(topk_indices, 0, 1)
    substitutes = [tokenizer.decode(ind[target_idx]) for ind in topk_indices]
    #topk_probs = torch.transpose(topk_probs, 0, 1)
    #probs = [prob[target_idx].item() for prob in topk_probs]
    
    return substitutes


def main(post2encoding_path, target2usage_path, tokenizer_name, model_name, output_path, k=10):
     
    # get encoded posts
    with open(post2encoding_path, 'r') as infile:
        post2encoding = json.load(infile)

    with open(target2usage_path, 'rb') as infile:
        target2mentions = pickle.load(infile)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    substitutes = dict()
    for t in tqdm(target2mentions):
        substitutes[t] = defaultdict(int)
        for post_id, target_idx in zip(target2mentions[t]['post_ids'], target2mentions[t]['target_idx']):
            post = post2encoding[post_id]
            subs = get_topk_substitutes(post, target_idx[0], model, tokenizer, k)
            for sub in subs:
                substitutes[t][sub] += 1

    with open(output_path, 'w') as outfile:
       json.dump(substitutes, outfile)


if __name__ == '__main__':

    
    tokenizer_name = '[name of model from HuggingFace transformers library] or path to model directory'
    model_name = '[name of model from HuggingFace transformers library] or path to model directory'
    k = 10

    post2encoding_path = '[filepath to json-file that maps post ids to encodings]'
    target2usage_path = '[path to pickle stored dictionary with usages]'
    output_path = '[filepath to json-file]'
    
    main(post2encoding_path, target2usage_path, tokenizer_name, model_name, output_path, k=k)
    
    