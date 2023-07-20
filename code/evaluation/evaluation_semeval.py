import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

output_file = '../output/results/ccoha_results.csv'
c1 = 'ccoha1'
c2 = 'ccoha2'

method2results = {'BL_PPMI': f'../output/results/ppmi/{c1}_{c2}_CD.csv',
                  'BL_SGNS': f'../output/results/sgns/{c1}_{c2}_CD.csv',
                  'CR_BERT_PT': f'../output/results/bert-base-uncased-PT/{c1}_{c2}_APD.csv',
                  'CR_BERT_FT': f'../output/results/bert-base-uncased-FT_ccoha/{c1}_{c2}_APD.csv',
                  'CR_XLMR_PT': f'../output/results/xlm-roberta-base-PT/{c1}_{c2}_APD.csv',
                  'CR_XLMR_FT_WSD': f'../output/results/xlm-roberta-base-FT_WSD/{c1}_{c2}_APD.csv',
                  'CR_XLMR_FT_SENT': f'../output/results/xlm-roberta-base-FT_SENT/{c1}_{c2}_APD.csv',
                  'MP_BERT_PT': f'../output/results/bert-base-uncased-PT/{c1}_{c2}_JSD.csv',
                  'MP_BERT_FT': f'../output/results/bert-base-uncased-FT_ccoha/{c1}_{c2}_JSD.csv'}

#method2results = {'test': '../output/results/bert-base-uncased-PT/different_topk/ccoha1_ccoha2_JSD-k=20.csv'}

# AGGREGRATE RESULTS
all_results = []
for method, file in method2results.items():
    results_df = pd.read_csv(file)
    for target, distance in zip(results_df['target'], results_df['distance']):
        all_results.append({'method': method, 'target': target, 'distance': distance})    
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(output_file, index=False)

# PERFORMANCE
with open('../../../Data/SemEval2020/ulscd_eng/graded.txt', 'r') as infile:
    gold = [line.replace('\n', '').split('\t') for line in infile.readlines()]
    targets = [g[0] for g in gold]
    graded_gold = [float(g[1]) for g in gold]
    ranked_gold = [r[0] for r in sorted([(t, g) for t, g in zip(targets, graded_gold)], key=lambda x: x[1], reverse=True)]

with open('../../../Data/SemEval2020/ulscd_eng/binary.txt', 'r') as infile:
    binary_gold = [int(line.replace('\n','').split('\t')[1]) for line in infile.readlines()]

grouped = all_results_df.groupby('method')
performance_results = []
for method, results in grouped:
    preds = results['distance']
    # Accuracy
    thres = preds.mean()
    binary_preds = [1 if pred > thres else 0 for pred in preds]
    accuracy = accuracy_score(binary_gold, binary_preds)
    # Correlation
    pearsonsr, p = pearsonr(preds, graded_gold)
    performance_results.append({'method': method, 'pearsonsr': pearsonsr, 'accuracy': accuracy})
performance_results_df = pd.DataFrame(performance_results)
#print(performance_results_df)
performance_results_df.to_csv(output_file.replace('_results.csv', '_performance.csv'), index=False)