from py_stringmatching import similarity_measure
import json
import pandas as pd

def inverse_overlap_coefficient(targets2subs_path1, targets2subs_path2, output_file):
    
    oc = similarity_measure.overlap_coefficient.OverlapCoefficient()

    with open(targets2subs_path1, 'r') as infile:
        target2subs1 = json.load(infile)

    with open(targets2subs_path2, 'r') as infile:
        target2subs2 = json.load(infile)

    results = [] 
    for target in target2subs1:
        set1 = set(target2subs1[target].keys())
        set2 = set(target2subs2[target].keys())
        coeff = oc.get_raw_score(set1, set2)

        result = {'target': target, 'distance': 1-coeff}
        results.append(result)
        
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    
    c1 = 'FD1'
    c2 = 'FD2'
    model = 'bert-base-dutch-cased-PT'
    t2s_path1 = f'../../output/results/{model}/{c1}_topk-substitutes.json'
    t2s_path2 = f'../../output/results/{model}/{c2}_topk-substitutes.json'
    output_file = f'../../output/results/{model}/{c1}_{c2}_IOC.csv'
    inverse_overlap_coefficient(t2s_path1, t2s_path2, output_file)
