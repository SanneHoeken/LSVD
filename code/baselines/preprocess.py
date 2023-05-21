import spacy, re, json
from collections import Counter
from tqdm import tqdm
import pandas as pd

def preprocess(text, nlp):
    
    text = re.sub(r"[^a-zA-Z\u00C0-\u00FF]"," ",text)
    text = text.replace('  ', ' ').replace('  ', ' ').rstrip().lstrip()
    lemmas = [t.lemma_.lower() for t in nlp(text)]

    # todo: split compounds? (https://github.com/dtuggener/CharSplit)
    # todo: apply spell checker ? (https://pypi.org/project/pyspellchecker/)

    return lemmas


def main(input_path, output_dir, spacy_pipeline):
    
    # Load post texts
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
        texts = data['text'].astype(str)
    elif input_path.endswith('.txt'):
        with open(input_path, 'r') as infile:
            data = infile.readlines()
        texts = [line.replace('\n', '') for line in data]
    
    # Lemmatize posts 
    print('Preprocessing posts...')
    nlp = spacy.load(spacy_pipeline, disable=['morphologizer', 'parser', 'senter', 'attribute_ruler', 'ner'])
    preprocessed = [preprocess(text, nlp) for text in tqdm(texts)]

    # initialize vocabulary and save index2word dict
    print('Saving vocabulary and counter...')
    all_lemmas = [w for p in preprocessed for w in p]
    print('Total number of tokens in corpus after preprocessing:', len(all_lemmas))
    counter = Counter(all_lemmas)

    with open(output_dir+'/counter.json', 'w') as outfile:
        json.dump(counter, outfile, ensure_ascii=False)
    
    vocabulary = sorted(list(set(all_lemmas)))
    print('Vocabulary size:', len(vocabulary))
    with open(output_dir+'/vocab.txt', 'w') as outfile:
        for word in vocabulary:
            outfile.write(word+'\n')
    
    # Write preprocessed posts to outputfile
    print('Saving pre-processed texts...')
    preprocessed_texts = [' '.join(l) for l in preprocessed]
    with open(output_dir+'/data_preprocessed.txt', 'w') as outfile:
        for post in preprocessed_texts:
            outfile.write(post+'\n')


if __name__ == '__main__':

    input_path = '../../../../Data/SemEval2020/ulscd_eng/ccoha2_lemma.txt'
    output_dir = '../../output/data/ccoha24baselines'
    spacy_pipeline = "en_core_web_sm" #"nl_core_news_sm"
    
    main(input_path, output_dir, spacy_pipeline)