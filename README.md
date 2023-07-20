# LSVD

Repository for the paper: Sanne Hoeken, Özge Alaçam, Antske Fokkens and Pia Sommerauer. 2023. Methodological Insights in Detection of Subtle Semantic Shifts with Contextualized and Static Language Models. [Manuscript submitted for publication].
Questions can be directed to sanne.hoeken@uni-bielefeld.de

## Running the experiments

For each of the following scripts, set the parameters inside of the code first. In a future version of this repository the parsing of command line arguments will be incorporated, so all scripts can be directly run from the command line. 

### Static representations
```
cd code/static
```
#### PPMI 

1.
```
python3 preprocess.py
```
2. 
```
python3 ppmi.py
```
3. 
```
python3 ppmi_ci+cd.py
```
#### SGNS 

1. 
```
python3 preprocess.py
```
2. 
```
python3 sgns.py
```
3. 
```
python3 op_align.py
```
4. 
```
python3 sgns_aligned_cd.py
```
### Contextualized representations
```
cd code/contextualized
```
#### Pipeline for BERT, XLM-R, and SENT XLM-R:

1. to execute both for target words and contexts:
```
python3 encode_data.py
```
2. 
```
python3 targets2usages.py
```
3. 
```
python3 apd.py
```
#### Pipeline for D-BERT: 

1. to execute both for target words and contexts
```
python3 encode_data.py
```
2. 
```
python3 finetune_mlm.py
```
3. 
```
python3 targets2usages.py
```
4. 
```
python3 apd.py
```
#### Pipeline for WSD XLM-R

1. to execute both for target words and contexts
```
python3 encode_data.py
```

2. 
[to explain]

3. 
```
python3 targets2usages.py
```
4. 
```
python3 apd.py
```
#### Masked target prediction (with BERT or DBERT)

1. to execute both for target words and contexts
```
python3 encode_data.py
```
2. 
```
python3 targets2usages.py
```
3. 
```
python3 topk_substitutes.py
```
4. 
```
python3 jsd.py
```

### Evaluation of all systems 
```
cd code/evaluation
```

on English use-case:
```
python3 evaluation_english.py
```
on Dutch use-case:
```
python3 evaluation_dutch.py
```
on SemEval-2020 test set:
```
python3 evaluation_semeval.py
```

### Manipulation experiments (with BERT or DBERT):

1. 
```
python3 encode_data.py
```
2. 
```
python3 encode_with_swap.py
```
3. 
```
python3 targets2usages.py
```
4. 
```
python3 apd_custom.py
```
