import numpy as np
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pandas as pd

def main(targets_path, ppmi_path1, ppmi_path2, vocab_path1, vocab_path2, output_path):
    """
    Compute cosine distance between ppmi vectors
    """

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    # Load vocabularies
    with open(vocab_path1, "r") as infile:
        vocab1 = [line.strip('\n') for line in infile.readlines()]
    w2i1 = {w: i for i, w in enumerate(vocab1)}
    
    with open(vocab_path2, "r") as infile:
        vocab2 = [line.strip('\n') for line in infile.readlines()]
    w2i2 = {w: i for i, w in enumerate(vocab2)}

    # Get ppmi matrices
    ppmi1 = load_npz(ppmi_path1)
    ppmi2 = load_npz(ppmi_path2)

    # Get vocab intersection and intersected column ids
    vocab_intersect = sorted(list(set(vocab1).intersection(vocab2)))
    intersected_columns1 = [int(w2i1[item]) for item in vocab_intersect]
    intersected_columns2 = [int(w2i2[item]) for item in vocab_intersect]

    # Print cosine distance of targets to output file
    results = []
    for target in tqdm(targets):
        target = target.lower()
        vector1 = ppmi1[:, w2i1[target]].toarray().flatten()[intersected_columns1]
        vector2 = ppmi2[:, w2i2[target]].toarray().flatten()[intersected_columns2]
        distance = cosine(vector1, vector2) 
        result = {'target': target, 'distance': distance}
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    
    targets_path = '[filepath to .txt file with one word per line]'
    ppmi_path1 = '[path to directory]' + '/ppmi.npz'
    ppmi_path2 = '[path to directory]' + '/ppmi.npz'
    vocab_path1 = '[path to directory]' + '/vocab.txt'
    vocab_path2 = '[path to directory]' + '/vocab.txt'
    
    output_path = '[filepath to .csv file]'

    main(targets_path, ppmi_path1, ppmi_path2, vocab_path1, vocab_path2, output_path)
