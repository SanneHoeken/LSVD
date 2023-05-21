import gensim
from gensim.models.word2vec import LineSentence

def main(input_path, output_path, dim, k, window, iters, min_count):
         
    # Initialize model
    model = gensim.models.Word2Vec(sg=1, hs=0, negative=k, sample=None, vector_size=dim, window=window, min_count=min_count, epochs=iters, workers=40)

    # Initialize vocabulary
    vocab_sentences = LineSentence(input_path)
    model.build_vocab(vocab_sentences)

    # Train
    sentences = LineSentence(input_path)
    print('Training Word2Vec model...')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    #l2_normalize = False
    #if l2_normalize:
        #model.init_sims(replace=True)

    # Save the vectors and the model
    model.wv.save_word2vec_format(output_path)
    #model.save(output_path + '.model')


if __name__ == '__main__':

    dim = 300
    k = 1
    window = 5
    iters = 5
    min_count = 10

    for c in ['ccoha1', 'ccoha2']:
        print(c)
        input_path = f'../../output/data/{c}4baselines/data_preprocessed.txt'
        output_path = f'../../output/data/{c}4baselines/sgns'
        
        main(input_path, output_path, dim, k, window, iters, min_count)
