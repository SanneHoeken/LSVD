import torch 
from transformers import AutoTokenizer, AutoModelForMaskedLM
from statistics import mean

sent = "I reconsidered you"
targets = ["reconsidered"]
topk = 10

model_name = "bert-base-uncased" #"xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
mask_id = tokenizer.mask_token_id
mask_token = tokenizer.decode(mask_id)
model = AutoModelForMaskedLM.from_pretrained(model_name)

for target in targets:

    target_ids = tokenizer.encode(target, add_special_tokens=False)
    masked_sent = sent.replace(target, mask_token*len(target_ids))

    inputs = tokenizer(masked_sent, return_tensors='pt')
    mask_idxs = torch.where(inputs['input_ids'] == mask_id)[1].tolist()

    with torch.no_grad():
        logits = model(**inputs).logits
    softmax = torch.softmax(logits, dim=-1).squeeze_()
    topk_probs, topk_indices = torch.topk(softmax, topk, sorted=True, dim=-1)
    topk_indices = torch.transpose(topk_indices, 0, 1)
    topk_probs = torch.transpose(topk_probs, 0, 1)

    print("Sentence:", masked_sent)
    print(f"Top {topk} candidate fillings (average probability over subwords):")
    for i, (ind, prob) in enumerate(zip(topk_indices, topk_probs)):
        mask_prob = round(mean([prob[mask_idx].item() for mask_idx in mask_idxs]), 3)
        word = ''.join([tokenizer.decode(ind[mask_id]) for mask_id in mask_idxs])
        print(f'{i+1}\t{word} ({mask_prob})')