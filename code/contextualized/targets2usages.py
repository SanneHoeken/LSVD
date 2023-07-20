import torch, pickle, json, os
from transformers import AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name):
        super(ContextEncoder, self).__init__()
        self.context_encoder = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)

    def forward(self, input_ids):
        context_output = self.context_encoder(input_ids)
        return context_output

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name):
        super(BiEncoderModel, self).__init__()
        self.context_encoder = ContextEncoder(encoder_name)

    def context_forward(self, context_input):
        return self.context_encoder.forward(context_input)


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


def extract_representations(post2encoding, target2mentions, model_name, 
                            layer_selection, wsd_biencoder_path, sent_class):                    
    
        # load model
        if wsd_biencoder_path:
            model = BiEncoderModel(model_name)
            model.load_state_dict(torch.load(wsd_biencoder_path, map_location=torch.device('cpu')), strict=False)
        elif sent_class:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        else:
            model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()

        # iterate over all post mentions of all target words
        for n, t in enumerate(target2mentions):
            print(f"Extracting {len(target2mentions[t]['post_ids'])} representations of '{t}' (target {n+1} out of {len(target2mentions)})...")
            for i, post_id in enumerate(tqdm(target2mentions[t]['post_ids'])):
                
                # feed post encodings to the model    
                input_ids = torch.tensor([post2encoding[post_id]])
                if wsd_biencoder_path:
                    encoded_layers = model.context_forward(input_ids)[-1]
                else:
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
         layer_selection='all', wsd_biencoder_path=None, sent_class=False,
         find=True, extract=True):
     
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
        target2vectors = extract_representations(post2encoding, target2mentions, model_name, 
                                                 layer_selection, wsd_biencoder_path, sent_class)
        with open(target2usage_path, 'wb') as outfile:
            pickle.dump(target2vectors, outfile)
    

if __name__ == '__main__':

   
    model_name = '[name of model from HuggingFace transformers library] or path to model directory'
    layer_selection = 'all'
    find = True
    extract = True
    
    wsd_biencoder_path = None # or '[path to] biencoder_xlmr.ckpt' for WSD XLM-R
    sent_class = False # or true for SENT XLM-R

    post2encoding_path = '[filepath to json-file that maps post ids to encodings]'
    target2encoding_path = '[filepath to json-file that maps targets to encodings]'
    target2usage_path = '[path to pickle dump dictionary with usages]'
    
    main(post2encoding_path, target2encoding_path, target2usage_path, model_name, 
        layer_selection=layer_selection, wsd_biencoder_path=wsd_biencoder_path, 
        sent_class=sent_class, find=find, extract=extract)
    