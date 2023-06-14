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
    
    c1 = 'TheDonald1'
    c2s = ['HillaryC', 'TheDonald2']
    model = 'bert-base-uncased-FT_RandomR'
    for c2 in c2s:
        t2s_path1 = f'../../output/results/{model}/{c1}_topk-substitutes.json'
        t2s_path2 = f'../../output/results/{model}/{c2}_topk-substitutes.json'
        output_file = f'../../output/results/{model}/{c1}_{c2}_JSD.csv'
        jensen_shannon_divergence(t2s_path1, t2s_path2, output_file)