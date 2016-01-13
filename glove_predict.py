import numpy as np
import operator
import pandas as pd
from scipy import linalg
from nltk.corpus import stopwords
import argparse
from utils import tokenize


def predict_weights(data, word2vec, N):

    stop = stopwords.words('english')

    pred_weights = []
    for i in range(data.shape[0]):
        #calculate word2vec for question
        q_vec = np.zeros(N)
        for w in tokenize(data['question'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                q_vec += word2vec[w.lower()]
        q_vec = q_vec / linalg.norm(q_vec)
    
        #calculate word2vec for answers
        A_vec = np.zeros(N)
        B_vec = np.zeros(N)
        C_vec = np.zeros(N)
        D_vec = np.zeros(N)
        for w in tokenize(data['answerA'][i]):
            if w.lower() in word2vec  and w.lower() not in stop:
                A_vec += word2vec[w.lower()]
    
        for w in tokenize(data['answerB'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                B_vec += word2vec[w.lower()]
            
        for w in tokenize(data['answerC'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                C_vec += word2vec[w.lower()]
    
        for w in tokenize(data['answerD'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                D_vec += word2vec[w.lower()]

        if linalg.norm(A_vec) < 1e-6:
            A_vec = np.ones(N)
        if linalg.norm(B_vec) < 1e-6:
            B_vec = np.ones(N)
        if linalg.norm(C_vec) < 1e-6:
            C_vec = np.ones(N)
        if linalg.norm(D_vec) < 1e-6:
            D_vec = np.ones(N)
        A_vec = A_vec / linalg.norm(A_vec) 
        B_vec = B_vec / linalg.norm(B_vec)
        C_vec = C_vec / linalg.norm(C_vec)
        D_vec = D_vec / linalg.norm(D_vec)
        
        #choose question based on cosine distance
        weights = np.concatenate((A_vec, B_vec, C_vec, D_vec)).reshape(4, N).dot(q_vec)
        pred_weights.append(weights)
        
    return pred_weights

if __name__ == '__main__':
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--N', type=int, default= 300, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--result_file', type=str, default='glove_result.csv')
    args = parser.parse_args()
    
    #read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )
    
    #read glove
    word2vec = {}
    with open("data/glove/glove.6B." + str(args.N) + "d.txt") as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
    
    #predict
    pred_answs = predict_weights(data, word2vec, args.N)
    
    #save prediction
    pd.DataFrame({
        'id': list(data['id']),
        'weightA': map(operator.itemgetter(0), pred_answs),
        'weightB': map(operator.itemgetter(1), pred_answs),
        'weightC': map(operator.itemgetter(2), pred_answs),
        'weightD': map(operator.itemgetter(3), pred_answs)
    })[['id', 'weightA', 'weightB', 'weightC', 'weightD']].to_csv(args.result_file, index=False)
