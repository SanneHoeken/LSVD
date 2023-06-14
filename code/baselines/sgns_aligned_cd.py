from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import pandas as pd
from tqdm import tqdm

def main(targets_path, sgns_path1, sgns_path2, output_path):
    """
    Compute cosine distance between aligned sgns vectors
    """

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
        
    sgns1 = KeyedVectors.load_word2vec_format(sgns_path1)
    sgns2 = KeyedVectors.load_word2vec_format(sgns_path2)

    # Print cosine distance of targets to output file
    results = []
    for target in tqdm(targets):
        target = target.lower()
        vector1 = sgns1[target]
        vector2 = sgns2[target]
        distance = cosine(vector1, vector2) 
        result = {'target': target, 'distance': distance}
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    
    targets_path = '../../../../Data/Reddit/target_words.txt'
    c1 = 'TheDonald1'
    c2 = 'TheDonald2'
    sgns_path1 = f'../../output/data/{c1}4baselines/sgns-{c2}aligned'
    sgns_path2 = f'../../output/data/{c2}4baselines/sgns-{c1}aligned'
    
    output_path = f'../../output/results/sgns/{c1}_{c2}_CD.csv'

    main(targets_path, sgns_path1, sgns_path2, output_path)
