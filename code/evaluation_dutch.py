import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

output_file = '../output/results/dutch_results.csv'
c1 = 'FD1'
c2 = 'FD2'
c3 = 'PS'
performance = True

name2results = {'BL_PPMI-between': f'../output/results/ppmi/{c1}_{c3}_CD.csv',
                'BL_PPMI-within': f'../output/results/ppmi/{c1}_{c2}_CD.csv',
                'BL_SGNS-between': f'../output/results/sgns/{c1}_{c3}_CD.csv',
                'BL_SGNS-within': f'../output/results/sgns/{c1}_{c2}_CD.csv',
                'CR_BERT_PT-between': f'../output/results/bert-base-dutch-cased-PT/{c1}_{c3}_APD.csv',
                'CR_BERT_PT-within': f'../output/results/bert-base-dutch-cased-PT/{c1}_{c2}_APD.csv',
                'CR_BERT_FT-between': f'../output/results/bert-base-dutch-cased-FT_UNION/{c1}_{c3}_APD.csv',
                'CR_BERT_FT-within': f'../output/results/bert-base-dutch-cased-FT_UNION/{c1}_{c2}_APD.csv',
                'CR_XLMR_PT-between': f'../output/results/xlm-roberta-base-PT/{c1}_{c3}_APD.csv',
                'CR_XLMR_PT-within': f'../output/results/xlm-roberta-base-PT/{c1}_{c2}_APD.csv',
                'CR_XLMR_FT_WSD-between': f'../output/results/xlm-roberta-base-FT_WSD/{c1}_{c3}_APD.csv',
                'CR_XLMR_FT_WSD-within': f'../output/results/xlm-roberta-base-FT_WSD/{c1}_{c2}_APD.csv',
                'CR_XLMR_FT_SENT-between': f'../output/results/xlm-roberta-base-FT_SENT/{c1}_{c3}_APD.csv',
                'CR_XLMR_FT_SENT-within': f'../output/results/xlm-roberta-base-FT_SENT/{c1}_{c2}_APD.csv',
                'MP_BERT_PT-between': f'../output/results/bert-base-dutch-cased-PT/{c1}_{c3}_JSD.csv',
                'MP_BERT_PT-within': f'../output/results/bert-base-dutch-cased-PT/{c1}_{c2}_JSD.csv',
                'MP_BERT_FT-between': f'../output/results/bert-base-dutch-cased-FT_UNION/{c1}_{c3}_JSD.csv',
                'MP_BERT_FT-within': f'../output/results/bert-base-dutch-cased-FT_UNION/{c1}_{c2}_JSD.csv'}

# AGGREGRATE RESULTS
all_results = []
for name, file in name2results.items():
    method = name.split('-')[0]
    label = name.split('-')[1]
    results_df = pd.read_csv(file)
    for target, distance in zip(results_df['target'], results_df['distance']):
        all_results.append({'method': method, 'label': label, 'target': target, 'distance': distance})    
all_results_df = pd.DataFrame(all_results)
#all_results_df.to_csv(output_file, index=False)


# PERFORMANCE
if performance:
    grouped = all_results_df.groupby('method')
    performance_results = []
    for method, results in grouped:

        binary_gold = [1 if label == 'between' else 0 for label in results['label']]
        preds = results['distance']
        pearsonsr, p = pearsonr(preds, binary_gold)

        binary_preds = []
        for i, row in results.iterrows():
            t = row['target']
            l = 'between' if row['label'] == 'within' else 'within'
            other_dist = results[results['target']==t][results[results['target']==t]['label'] == l]['distance']
            binary_pred = row['distance'] > other_dist
            binary_preds.append(binary_pred)

        accuracy = accuracy_score(binary_gold, binary_preds)

        performance_results.append({'method': method, 'pearsonsr': pearsonsr, 'accuracy': accuracy})
    performance_results_df = pd.DataFrame(performance_results)
    performance_results_df.to_csv(output_file.replace('_results.csv', '_performance.csv'), index=False)