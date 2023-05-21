import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, spdiags, save_npz
from tqdm import tqdm

def cooccurence_matrix(sentences, w2i, window_size):

    matrix = dict()

    for sentence in tqdm(sentences):
        for i, word in enumerate(sentence):
            lowerWindowSize = max(i-window_size, 0)
            upperWindowSize = min(i+window_size, len(sentence))
            window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1]
            if len(window)==0: # Skip one-word sentences
                continue
            windex = w2i[word]
            for contextWord in window:
                if (windex,w2i[contextWord]) not in matrix:
                    matrix[(windex,w2i[contextWord])] = 0
                matrix[(windex, w2i[contextWord])] += 1

    return matrix


def main(output_path, input_path, vocab_path, window_size, alpha, k):
    
    # Load posts
    with open(input_path, "r") as infile:
        posts = [line.strip('\n') for line in infile.readlines()]
    lemmatized = [post.split() for post in posts]

    # Load vocabulary
    with open(vocab_path, "r") as infile:
        vocabulary = [line.strip('\n') for line in infile.readlines()]
    w2i = {w: i for i, w in enumerate(vocabulary)}
    
    # Get co-occurence counts of whole corpus
    print('Computing co-occurence matrix...')
    matrix_dict = cooccurence_matrix(lemmatized, w2i, window_size)

    # Convert dictionary to sparse matrix
    matrix = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
    matrix._update(matrix_dict) 
    
    print('Computing ppmi matrix...')
    # Get probabilities
    row_sum = matrix.sum(axis = 1)
    col_sum = matrix.sum(axis = 0)

    # Compute smoothed P_alpha(c)
    smooth_col_sum = np.power(col_sum, alpha)
    col_sum = smooth_col_sum/smooth_col_sum.sum()

    # Compute P(w)
    row_sum = row_sum.astype(np.double)
    row_sum[row_sum != 0] = np.array(1.0/row_sum[row_sum != 0]).flatten()
    col_sum = col_sum.astype(np.double)
    col_sum[col_sum != 0] = np.array(1.0/col_sum[col_sum != 0]).flatten()

    # Apply epmi weighting (without log)
    diag_matrix = spdiags(row_sum.flatten(), [0], row_sum.flatten().size, row_sum.flatten().size, format = 'csr')
    matrix = csr_matrix(diag_matrix * matrix)
    
    diag_matrix = spdiags(col_sum.flatten(), [0], col_sum.flatten().size, col_sum.flatten().size, format = 'csr')
    matrix = csr_matrix(matrix * diag_matrix)

    # Apply log weighting and shift values
    matrix.data = np.log(matrix.data)
    matrix.data -= np.log(k)

    # Eliminate negative and zero counts
    matrix.data[matrix.data <= 0] = 0.0
    matrix.eliminate_zeros()

    # Save ppmi matrix
    print('Saving ppmi matrix...')
    save_npz(output_path, csr_matrix(matrix))
    

if __name__ == '__main__':

    window_size = 10
    alpha = 0.75
    k = 5

    for c in ['ccoha1', 'ccoha2']:
        print(c)
        input_path = f'../../output/data/{c}4baselines/data_preprocessed.txt'
        vocab_path = f'../../output/data/{c}4baselines/vocab.txt'
        output_path = f'../../output/data/{c}4baselines/ppmi.npz' #.npz-file

        main(output_path, input_path, vocab_path, window_size, alpha, k)

