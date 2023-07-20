from scipy.spatial.distance import jensenshannon
import json
import pandas as pd

def jensen_shannon_divergence(targets2subs_path1, targets2subs_path2, output_file):

    with open(targets2subs_path1, 'r') as infile:
        target2subs1 = json.load(infile)

    with open(targets2subs_path2, 'r') as infile:
        target2subs2 = json.load(infile)

    results = [] 
    for target in target2subs1:
        set1 = set(target2subs1[target].keys())
        set2 = set(target2subs2[target].keys())
        intersect = set1.intersection(set2)
        counts1 = [target2subs1[target][sub] for sub in intersect]
        counts2 = [target2subs2[target][sub] for sub in intersect]
        prob_dist1 = [x / sum(counts1) for x in counts1]
        prob_dist2 = [x / sum(counts2) for x in counts2]
        jsd = jensenshannon(prob_dist1, prob_dist2)
        result = {'target': target, 'distance': jsd}
        results.append(result)
        
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    
    t2s_path1 = '[path to json-file mapping targets to topk substitutes]'
    t2s_path2 = '[path to json-file mapping targets to topk substitutes]'
    output_file = '[path to csv-file]'
    
    jensen_shannon_divergence(t2s_path1, t2s_path2, output_file)