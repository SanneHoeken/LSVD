import pandas as pd

output_file = '../output/results/english_experiments_results.csv'
c1 = 'TheDonald1'
c2 = 'TheDonald2'
c3 = 'HillaryC'
correlation = True

#name2results = {'BERT_PT-donaldVSrandom': f'../output/results/bert-base-uncased-PT/{c1}_{c3}_APD.csv',
                #'BERT_PT-hillaryVSrandom': f'../output/results/bert-base-uncased-PT/{c2}_{c3}_APD.csv',
                #'BERT_PT-donaldVShillary': f'../output/results/bert-base-uncased-PT/{c1}_{c2}_APD.csv'}

name2results = {'PPMI-between': f'../output/results/ppmi/{c1}_{c3}_CD.csv',
                'PPMI-within': f'../output/results/ppmi/{c1}_{c2}_CD.csv',
                'SGNS-between': f'../output/results/sgns/{c1}_{c3}_CD.csv',
                'SGNS-within': f'../output/results/sgns/{c1}_{c2}_CD.csv',
                'BERT_PT-between': f'../output/results/bert-base-uncased-PT/{c1}_{c3}_APD.csv',
                'BERT_PT-within': f'../output/results/bert-base-uncased-PT/{c1}_{c2}_APD.csv',
                'BERT_FT-between': f'../output/results/bert-base-uncased-FT_RandomR/{c1}_{c3}_APD.csv',
                'BERT_FT-within': f'../output/results/bert-base-uncased-FT_RandomR/{c1}_{c2}_APD.csv',
                'XLMR_PT-between': f'../output/results/xlm-roberta-base-PT/{c1}_{c3}_APD.csv',
                'XLMR_PT-within': f'../output/results/xlm-roberta-base-PT/{c1}_{c2}_APD.csv'}

# AGGREGRATE RESULTS
all_results = []
for name, file in name2results.items():
    method = name.split('-')[0]
    label = name.split('-')[1]
    results_df = pd.read_csv(file)
    for target, distance in zip(results_df['target'], results_df['distance']):
        all_results.append({'method': method, 'label': label, 'target': target, 'distance': distance})    
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(output_file, index=False)


# CORRELATION
if correlation:
    grouped = all_results_df.groupby('method')
    correlation_results = []
    for method, results in grouped:
        results['label'] = [1 if label == 'between' else 0 for label in results['label']]
        correlation = results['distance'].corr(results['label'], method='pearson')
        correlation_results.append({'method': method, 'correlation': correlation})
    correlation_results_df = pd.DataFrame(correlation_results)
    correlation_results_df.to_csv(output_file.replace('_results.csv', '_correlation.csv'), index=False)