import torch, json
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training with probability 0.15 
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main(model_dir, output_dir, post2encoding_path, batch_size, epochs):

    # Setup CUDA / GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed(42, n_gpu)
    
    # Load sentence encodings
    with open(post2encoding_path, "r") as infile:
        post2encoding = json.load(infile)
    sentence_encodings = torch.tensor(list(post2encoding.values()))
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    model.to(device)

    # Create dataloader and set scheduler for training
    sampler = RandomSampler(sentence_encodings)
    dataloader = DataLoader(sentence_encodings, sampler=sampler, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.zero_grad()
    for i in range(epochs):
        #print(f"Epoch {i+1} out of {epochs}...")
        print("Epoch", i+1, "out of", epochs)
        for step, batch in enumerate(tqdm(dataloader)):
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(inputs, labels=labels) 
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step() 
            model.zero_grad()
        
        model.save_pretrained(output_dir)
        

if __name__ == '__main__':
    
    model_dir = "bert-base-uncased"
    output_dir = "bert-base-uncased-FT_ccoha"
    post2encoding_path = 'ccoha-all_post2encoding.json'
    batch_size = 8
    epochs = 3

    main(model_dir, output_dir, post2encoding_path, batch_size, epochs)