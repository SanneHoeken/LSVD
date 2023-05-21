import pickle, torch
from torchmetrics.functional import pairwise_cosine_similarity
import pandas as pd
from tqdm import tqdm

def apd(t2u_path1, t2u_path2, output_file):

    with open(t2u_path1, 'rb') as infile1:
        target2usages1 = pickle.load(infile1)

    with open(t2u_path2, 'rb') as infile2:
        target2usages2 = pickle.load(infile2)

    assert set(target2usages1.keys()) == set(target2usages2.keys())
    
    results = []
    for target in tqdm(target2usages1):
        vectors1 = torch.stack(target2usages1[target]['embeddings']) 
        vectors2 = torch.stack(target2usages2[target]['embeddings']) 
        pdist = pairwise_cosine_similarity(vectors1, vectors2) 
        apd = 1 - torch.mean(torch.flatten(pdist)).item() 
        result = {'target': target, 'distance': apd}
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    
    c1 = 'ccoha1'
    c2 = 'ccoha2'
    model = 'bert-base-uncased-FT_ccoha'
    t2u_path1 = f'../../output/data/{model}/{c1}_targets2usages'
    t2u_path2 = f'../../output/data/{model}/{c2}_targets2usages'
    output_file = f'../../output/results/{model}/{c1}_{c2}_APD.csv'
    
    apd(t2u_path1, t2u_path2, output_file)