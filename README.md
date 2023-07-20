# LSVD

Repository for the paper: Sanne Hoeken, Özge Alaçam, Antske Fokkens and Pia Sommerauer. 2023. Methodological Insights in Detection of Subtle Semantic Shifts with Contextualized and Static Language Models. [Manuscript submitted for publication].
Questions can be directed to sanne.hoeken@uni-bielefeld.de

## Running the experiments

For each of the following scripts, set the parameters inside of the code first. In a future version of this repository the parsing of command line arguments will be incorporated, so all scripts can be directly run from the command line. 

### Static representations
```
cd code/static
```
PPMI 
1. 
    ```
    preprocess.py
    ```
2. 
    ```
    ppmi.py
    ```
3. 
    ```
    ppmi_ci+cd.py
    ```
SGNS 
1. 
    ```
    preprocess.py
    ```
2. 
    ```
    sgns.py
    ```
3. 
    ```
    op_align.py
    ```
4. 
    ```
    sgns_aligned_cd.py
    ```
### Contextualized representations
```
cd code/contextualized
```
Pipeline for BERT, XLM-R, and SENT XLM-R:

1. 
    ```
    encode_data.py
    ```
    to execute both for target words and contexts
2. 
    ```
    targets2usages.py
    ```
3. 
    ```
    apd.py
    ```
Pipeline for D-BERT: 
1. 
    ```
    encode_data.py
    ```
    to execute both for target words and contexts
2. 
    ```
    finetune_mlm.py
    ```
3. 
    ```
    targets2usages.py
    ```
4. 
    ```
    apd.py
    ```
Pipeline for WSD XLM-R
0. [to explain]

Masked target prediction (with BERT or DBERT)
1. 
    ```
    encode_data.py
    ```
2. 
    ```
    targets2usages.py
    ```
3. 
    ```
    topk_substitutes.py
    ```
4. 
    ```
    jsd.py
    ```

Evaluation of all systems 
```
cd code/evaluation
```
on English use-case:
1. 
    ```
    evaluation_english.py
    ```
on Dutch use-case:
1. 
    ```
    evaluation_dutch.py
    ```
on SemEval-2020 test set:
1. 
    ```
    evaluation_semeval.py
    ```

Manipulation experiments (with BERT or DBERT):
1. 
    ```
    encode_data.py
    ```
2. 
    ```
    encode_with_swap.py
    ```
3. 
    ```
    targets2usages.py
    ```
4. 
    ```
    apd_custom.py
    ```