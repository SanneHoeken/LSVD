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
    
    t2s_path1 = '[path to json-file mapping targets to topk substitutes]'
    t2s_path2 = '[path to json-file mapping targets to topk substitutes]'
    output_file = '[path to csv-file]'
    inverse_overlap_coefficient(t2s_path1, t2s_path2, output_file)
