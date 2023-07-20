# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>

import numpy as np

def read_embeddings(file):
    header = file.readline().split(' ')
    count = int(header[0]) 
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim)) 
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ')
        
    return (header, words, matrix) 

def normalize(matrix):

    # length normalize
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]

    # mean centre
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        mask = np.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main(src_input, trg_input, src_output, trg_output):
    
    np.random.seed(0)
    dtype = 'float64'

    batch_size = 10000
    csls_neighborhood=10
    threshold = 0.000001
    stochastic_multiplier = 2.0
    stochastic_interval = 50

    # Read input embeddings
    with open(src_input, errors='surrogateescape') as srcfile:
        src_header, src_words, x = read_embeddings(srcfile)
    
    with open(trg_input, errors='surrogateescape') as trgfile:
        trg_header, trg_words, z = read_embeddings(trgfile)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Normalization
    normalize(x)
    normalize(z)

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    
    identical = set(src_words).intersection(set(trg_words))
    for word in identical:
        src_indices.append(src_word2ind[word])
        trg_indices.append(trg_word2ind[word])

    # Allocate memory
    xw = np.empty_like(x)
    zw = np.empty_like(z)
    src_size = x.shape[0] 
    trg_size = z.shape[0] 
    simfwd = np.empty((batch_size, trg_size), dtype=dtype)
    simbwd = np.empty((batch_size, src_size), dtype=dtype)
    
    best_sim_forward = np.full(src_size, -100, dtype=dtype)
    src_indices_forward = np.arange(src_size)
    trg_indices_forward = np.zeros(src_size, dtype=int)
    best_sim_backward = np.full(trg_size, -100, dtype=dtype)
    src_indices_backward = np.zeros(trg_size, dtype=int)
    trg_indices_backward = np.arange(trg_size)
    knn_sim_fwd = np.zeros(src_size, dtype=dtype)
    knn_sim_bwd = np.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = 0.1
    end = False
    
    while True:

        # Increase the keep probability if we have not improved in stochastic_interval iterations
        if it - last_improvement > stochastic_interval:
            print(it)
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, stochastic_multiplier*keep_prob)
            last_improvement = it

        # Update the embedding (orthogonal) mapping
        u, s, vt = np.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
        w = vt.T.dot(u.T)
        x.dot(w, out=xw)
        zw[:] = z

        # Self-learning
        if end:
            break
        
        # Update the training dictionary
        else:
            
            # forward 
            for i in range(0, trg_size, simbwd.shape[0]):
                j = min(i + simbwd.shape[0], trg_size)
                zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=csls_neighborhood, inplace=True)
            for i in range(0, src_size, simfwd.shape[0]):
                j = min(i + simfwd.shape[0], src_size)
                xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            
            # backward
            for i in range(0, src_size, simfwd.shape[0]):
                j = min(i + simfwd.shape[0], src_size)
                xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=csls_neighborhood, inplace=True)
            for i in range(0, trg_size, simbwd.shape[0]):
                j = min(i + simbwd.shape[0], trg_size)
                zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            
            # union
            src_indices = np.concatenate((src_indices_forward, src_indices_backward))
            trg_indices = np.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            objective = (np.mean(best_sim_forward) + np.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= threshold:
                last_improvement = it
                best_objective = objective

            it += 1

    print(best_objective)

    # Write mapped embeddings
    with open(src_output, mode='w', errors='surrogateescape') as srcfile:
        print(' '.join(src_header), file=srcfile, end='')
        for i in range(len(src_words)):
            print(src_words[i] + ' ' + ' '.join(['%.6g' % x for x in xw[i]]), file=srcfile)
    
    with open(trg_output, mode='w', errors='surrogateescape') as trgfile: 
        print(' '.join(trg_header), file=trgfile, end='')
        for i in range(len(trg_words)):
            print(trg_words[i] + ' ' + ' '.join(['%.6g' % x for x in zw[i]]), file=trgfile)

    
if __name__ == '__main__':
    
    src_input = '[path to directory]' +'/sgns'
    trg_input = '[path to directory]' +'/sgns'
    src_output = '[path to directory]' +'/sgns-aligned'
    trg_output = '[path to directory]' +'/sgns-aligned'

    main(src_input, trg_input, src_output, trg_output)
