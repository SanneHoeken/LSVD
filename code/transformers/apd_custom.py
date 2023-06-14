import pickle, torch
from torchmetrics.functional import pairwise_cosine_similarity


def apd(t2u_path1, t2u_path2, target1, target2):

    with open(t2u_path1, 'rb') as infile1:
        target2usages1 = pickle.load(infile1)

    with open(t2u_path2, 'rb') as infile2:
        target2usages2 = pickle.load(infile2)

    vectors1 = target2usages1[target1]['embeddings']
    vectors2 = target2usages2[target2]['embeddings']
    vectors1 = torch.stack(vectors1) 
    vectors2 = torch.stack(vectors2) 
    pdist = pairwise_cosine_similarity(vectors1, vectors2) 
    apd = 1 - torch.mean(torch.flatten(pdist)).item() 
    print(apd)
        

if __name__ == '__main__':
    
    c1 = 'MNP-FD1'
    c2 = 'MNP-FD1'
    model = 'bert-base-dutch-cased-FT_UNION'
    t2u_path1 = f'../../output/data/{model}/{c1}_targets2usages'
    t2u_path2 = f'../../output/data/{model}/{c2}_targets2usages'
    target1 = 'liberaal'
    target2 = 'liberaal'

    apd(t2u_path1, t2u_path2, target1, target2)