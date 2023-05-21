import torch, pickle, json, os
from transformers import AutoModel
from tqdm import tqdm

def find_target_mentions(post2encoding, target2encoding):
    
    target2mentions = {t : {'post_ids': [], 'target_idx': [], 'embeddings': []} for t in target2encoding}

    # iterate over posts and find target word mentions
    print('Finding target word mentions...')
    for post_id, post_encodings in tqdm(post2encoding.items()):
        for i in range(len(post_encodings)):
            for t, target_ids in target2encoding.items():
                if post_encodings[i:i+len(target_ids)] == target_ids:
                    target2mentions[t]['post_ids'].append(post_id)
                    target2mentions[t]['target_idx'].append((i, i+len(target_ids)))           

    return target2mentions


def extract_representations(post2encoding, target2mentions, model_name, layer_selection):                    
    
        # load model
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()

        # iterate over all post mentions of all target words
        for n, t in enumerate(target2mentions):
            print(f"Extracting {len(target2mentions[t]['post_ids'])} representations of '{t}' (target {n+1} out of {len(target2mentions)})...")
            for i, post_id in enumerate(tqdm(target2mentions[t]['post_ids'])):
                
                # feed post encodings to the model    
                input_ids = torch.tensor([post2encoding[post_id]])
                encoded_layers = model(input_ids)[-1]
                
                # extract selection of hidden layer(s)
                if type(layer_selection) == int:
                    vecs = encoded_layers[layer_selection].squeeze(0)
                elif type(layer_selection) == list:
                    selected_encoded_layers = [encoded_layers[x] for x in layer_selection]
                    vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
                elif layer_selection == 'all':
                    vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
                
                # target word selection 
                vecs = vecs.detach()
                start_idx, end_idx = target2mentions[t]['target_idx'][i]
                vecs = vecs[start_idx:end_idx]
                
                # aggregate sub-word embeddings (by averaging)
                vector = torch.mean(vecs, 0)
                if torch.isnan(vector).any():
                    print(t, post_id, start_idx, end_idx)
                
                target2mentions[t]['embeddings'].append(vector)

        return target2mentions
        

def main(post2encoding_path, target2encoding_path, target2usage_path, model_name, 
         layer_selection='all', find=True, extract=True):
     
    # get encoded posts
    with open(post2encoding_path, 'r') as infile:
        post2encoding = json.load(infile)

    # get encoded targets
    with open(target2encoding_path, 'r') as infile:
        target2encoding = json.load(infile)

    # map target words to mentions in posts
    if find:
        target2mentions = find_target_mentions(post2encoding, target2encoding)
        with open(target2usage_path, 'wb') as outfile:
            pickle.dump(target2mentions, outfile)
    else:
        assert os.path.isfile(target2usage_path)
        with open(target2usage_path, 'rb') as infile:
            target2mentions = pickle.load(infile)

    # extract representations of target words mentions in posts
    if extract:
        target2vectors = extract_representations(post2encoding, target2mentions, model_name, layer_selection)
        with open(target2usage_path, 'wb') as outfile:
            pickle.dump(target2vectors, outfile)
    

if __name__ == '__main__':

    #cs = ['HillaryC', 'TheDonald1', 'TheDonald2', 'RandomR']
    cs = ['ccoha1', 'ccoha2']
    model_name = '../../output/models/bert-base-uncased-FT_ccoha'
    layer_selection = 'all'
    find = True
    extract = True
    
    for c in cs:
        print(c)
        post2encoding_path = f'../../output/data/bert-base-uncased-PT/{c}_post2encoding.json'
        target2encoding_path = f'../../output/data/bert-base-uncased-PT/ccoha_target2encoding.json'
        target2usage_path = f'../../output/data/bert-base-uncased-FT_ccoha/{c}_targets2usages'
        

        main(post2encoding_path, target2encoding_path, target2usage_path, model_name, 
            layer_selection=layer_selection, find=find, extract=extract)
    