from __future__ import division
import argparse, sys, re, pickle
import math, nltk, sklearn
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit as sigmoid
from nltk import FreqDist
import pandas as pd

__authors__ = ['Tailai ZHANG','Pankaj Patil','kalyani DESHMUKH', 'Kimaya DHADE']
__emails__  = ['tailai.zhang@essec.edu','pankaj.patil@essec.edu' , 'kalyani.deshmukh@essec.edu' , 'kimaya.dhade@essec.edu']

def text2sentences(path):
    # tokenize sentence and drop all punctuations, and numbers
    sentences = []
    with open(path, encoding='utf8') as f:
        for l in f:
            # split sentences
            words = re.split(r' *[\.\?!][\'"\)\]]* *', l.lower())
            # split words
            words = [w for token in words for w in token.split() \
                    if len(w) > 1 and w.isalnum() and not w.isdigit()]
            sentences.append(words)
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t') # read the csv file
    pairs = zip(data['word1'],data['word2'],data['similarity']) 
    return pairs

class mySkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.winSize = winSize
        self.minCount = minCount
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.sentences = sentences
        sample = 0.001 # randomly drop high frequency items

        # -------------------- Freq + Count -------------------- #
        # vcount: frequency counts of sentences
        # vcount_freq: drop words for counts < minCount
        # word_index: from word to its index
        # total_words: sum of all counts
        # vcount_prob: probabilities for each word, raised to 3/4 power
        # vocab_size: vocabulary size

        vcount = FreqDist([x for sentence in sentences for x in sentence])
        vcount_freq = {wrd:count for wrd, count in vcount.most_common() \
                      if count > self.minCount}

        # create probabilities for negative sampling
        total_words = sum((a**(0.75) for a in vcount_freq.values()))
        vcount_prob = {wrd:count**0.75/total_words \
                       for wrd, count in vcount_freq.items()}
        vocab_size = len(vcount_prob)
        word_index = {wrd:ind for ind, wrd in enumerate(vcount_prob)}

        self.vcount, self.vcount_freq, self.total_words, self.vcount_prob, \
        self.vocab_size, self.word_index = vcount, vcount_freq, total_words, \
        vcount_prob, vocab_size, word_index
        self.error = 0

        # -------------------- Sub Sampling -------------------- #

        # sub-sampling: remove randomly words with high frequency
        # inspired by http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        sen = [] # prepared for word tokens
        total_words = sum(vcount_freq.values()) # real total words
        for sent in sentences:
            # drop non-existing words, more often less frequent words
            sent = [w for w in sent if w in vcount_prob.keys()]
            for word in sent:
                last_word = word_index[word]
                # get frequency
                cn = vcount_freq[word]
                # drop high frequency words
                ran = (math.sqrt(cn / (sample * total_words + 1))) * (sample * total_words) / cn
                if ran < np.random.random():
                    continue
                # add pack to sen
                sen.append(last_word)

        self.sen = sen
        self.vcount, self.vcount_freq, self.total_words, self.vcount_prob, \
        self.vocab_size, self.word_index = vcount, vcount_freq, total_words, \
        vcount_prob, vocab_size, word_index


    def train(self, stepsize = 0.025, epochs = 6):

        # -------------------- Parameters -------------------- #
        # back propagation insights credited to
        # https://ckmarkoh.github.io/blog/2016/08/29/neural-network-word2vec-part-3-implementation/
        #
        nEmbed = self.nEmbed # no. of units for hidden layer
        negativeRate = self.negativeRate # no. words for negative sampling
        avg_err = 0
        err_count = 0
        global_word_count = 0
        vocab_size = self.vocab_size
        vcount_prob = self.vcount_prob
        word_index = self.word_index

        # w1: random weights between input layer and hidden layer
        # w2: random weights between hidden layer and output layer
        w0 = np.random.uniform(low=-0.5/nEmbed, high=0.5/nEmbed,
        size=(vocab_size, nEmbed))
        w1 = np.zeros((nEmbed, vocab_size))

        p_count = 0 # how many times we already trained
        avg_err = 0 # to be printed
        err_count = 0 # calculate average count
        alpha = stepsize # we tried to adjust alpha after each training
        vcount_freq = self.vcount_freq
        batchSize = 20
        sen = self.sen

        # -------------------- Model training -------------------- #

        for local_iter in range(epochs):
            print("Epoch {}/{}".format(local_iter + 1, epochs))

            for word_pos, token in enumerate(sen):
                # randomize window winSize
                winSize = np.random.randint(1, self.winSize)
                context_start = max(word_pos - winSize, 0)
                context_end = min(word_pos + winSize + 1, len(sen))
                context = sen[context_start:word_pos] + \
                sen[word_pos + 1: context_end]

                # context_word: all words near token within window size
				# o_err: error between predicted value and actual value
				# h_err: hidden layer errors
				
                for context_word in context:
                    # initialize hidden layer errors
                    h_err = np.zeros((nEmbed))
                    p_count += 1
                    # randomly select negative samples
                    negative = np.random.choice(list(vcount_prob.keys()),
                                                size = negativeRate,
                                                p=list(vcount_prob.values()))
					# negative indices							
                    neg_ind = [word_index[w] for w in negative]
                    # classifiers: context and target with labels
                    classifiers = [(context_word, 1)] + \
                    [(target, 0) for target in neg_ind]

					# start training the model
                    for target, label in classifiers:
                        # inspired by http://cpmarkchang.logdown.com/posts/773558-neural-network-word2vec-part-3-implementation
                        pred = sigmoid(np.dot(w0[context_word, :],
                        w1[:, target]))
                        # difference between prediction and label
                        o_err = pred - label
                        # ----- backward propagation ----- #
                        # 1.propagate error to hidden layer
                        h_err += o_err * w1[:, target]
                        # 2.update w1
                        w1[:, target] -= alpha * o_err * w0[context_word]
                        avg_err += abs(o_err)
                        err_count += 1
                    # 3.update w0
                    w0[context_word, :] -= alpha * h_err

        # -------------------- Print Out -------------------- #

                    if p_count % 10000 == 0:
                        # print out progress per 10000
                        print("Epochs: {}, Err: {}, Progress: {}/{}, alpha: {}".\
                        format(local_iter + 1, avg_err/ float(err_count),
                        word_pos, len(sen), alpha))
                        avg_err = 0
                        err_count = 0
        self.w0 = w0
        self.w1 = w1
    # save the model by using pickle 
    def save(self,path):
        if '.pickle' not in path:
            path = path + '.pickle'
        with open(path,'wb') as handle:  # write in the file
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def similarity(self, word1, word2):
        # cosine similarity = u*v / |u|*|v|
		# v1, v2: index of word1, word2
        v1 = self.word_index[word1]
        v2 = self.word_index[word2]
        return (1+np.dot(self.w0[v1, :],self.w0[v2, :])/ \
                (np.linalg.norm(self.w0[v1, :]) * np.linalg.norm(self.w0[v2,:])))/2

    @staticmethod
    def load(path):
        if '.pickle' not in path:
            path = path + '.pickle'
        with open(path,'rb') as handle: # read the file 
            sg = pickle.load(handle)
        return sg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mySkipGram.load(opts.model)
        for a,b,_ in pairs:
            print (sg.similarity(a,b))
