import pickle, torch, random
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_manhattan_distance
from torch.nn.functional import normalize
import pandas as pd
from tqdm import tqdm


def calculate_apd(vectors1, vectors2, l1_normalize, metric):
    
    vectors1 = torch.stack(vectors1) 
    vectors2 = torch.stack(vectors2) 

    if l1_normalize:
        vectors1 = normalize(vectors1, p=1)
        vectors2 = normalize(vectors2, p=1)

    if metric == 'cosine':
        pdist = pairwise_cosine_similarity(vectors1, vectors2) 
        apd = 1 - torch.mean(torch.flatten(pdist)).item() 
    elif metric == 'manhattan':
        pdist = pairwise_manhattan_distance(vectors1, vectors2)
        apd = torch.mean(torch.flatten(pdist)).item() 
    
    return apd


def apd(t2u_path1, t2u_path2, output_file, l1_normalize=False, metric='cosine', sample=None):

    with open(t2u_path1, 'rb') as infile1:
        target2usages1 = pickle.load(infile1)

    with open(t2u_path2, 'rb') as infile2:
        target2usages2 = pickle.load(infile2)

    #assert set(target2usages1.keys()) == set(target2usages2.keys())
    
    results = []
    for target in target2usages1:
        
        vectors1 = target2usages1[target]['embeddings']
        vectors2 = target2usages2[target]['embeddings']
        
        if sample:
            sample1 = len(vectors1) if len(vectors1) < sample else sample
            sample2 = len(vectors2) if len(vectors2) < sample else sample
            vectors1_sample = random.sample(vectors1, sample1)
            vectors2_sample = random.sample(vectors2, sample2)
            apd = calculate_apd(vectors1_sample, vectors2_sample, l1_normalize, metric)
        else:
            apd = calculate_apd(vectors1, vectors2, l1_normalize, metric)
        
        result = {'target': target, 'distance': apd}
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    
    c1 = 'ccoha1'
    c2 = 'ccoha2'
    model = 'bert-base-uncased-PT'
    t2u_path1 = f'../../output/data/{model}/{c1}_targets2usages'
    t2u_path2 = f'../../output/data/{model}/{c2}_targets2usages'

    for sample in tqdm(range(1, 151)):
        for run in range(1, 11):
            output_file = f'../../output/results/{model}/sampled_ccoha/{c1}_{c2}_APD_sample={sample}_run={run}.csv'
            apd(t2u_path1, t2u_path2, output_file, sample=sample)