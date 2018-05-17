# NLP - Assignment 1 Catch up

The purpose of this project is to train a word2vec model and calculate the cosine similarity between words based on a text file.

## Authors: 
- Tailai ZHANG 
- Pankaj Patil
- Kalyani DESHMUKH
- Kimaya DHADE

## Usage:

#### Train model:

	python main.3.py --text input-10000.txt --model model.pickle

#### Test model & generate output:
    
    python main.3.py –-text test.txt --model model.pickle --test

#### Inputs: 

1. `input-10000.txt` file
2. `test.txt` file: test file with header of `word1`, `word2`, and `similarity`. 

#### Outputs: 

1. `model.pickle`: pickle file, saves the whole model


## Result:

- We first tried to implement matrix calculation instead of a for loop in "negative samples"

	    for context_word in context:
	        h_err = np.zeros((nEmbed))
	        p_count += 1
	        # initial error with zeros
	        negative = np.random.choice(list(vcount_prob.keys()),
	                                    size = negativeRate,
	                                    p=list(vcount_prob.values()))
	        neg_ind = [word_index[w] for w in negative]
	
	        classifiers = [(token, 1)] + \
	        [(target, 0) for target in neg_ind]
	
	        #for target, label in classifiers:
	        targets, labels = zip(*classifiers)
	        # shape --> (6,)
	        o_pred = sigmoid(np.dot(w0[context_word,:],
	        w1[:, labels]))
	        # shape --> (6,)
	        o_err = o_pred - labels
	        # o_err --> [-0.5, 0.5, 0.5, -0.5, 0.5, 0.5]
	        # backprogation
	        # h_err (100,) o_err(6,) w1[:, targets] (100,6)
	        h_err += np.dot(w1[:, targets], o_err)
	
	        # update w0, w1 every batchSize
	        if p_count % batchSize == 0:
	
	            w1[:, targets] -=  alpha * np.dot(w0[context_word][:, np.newaxis]
	            , o_err[np.newaxis, :])
	            avg_err += np.average(abs(o_err))
	            err_count += 1
	            w0[context_word, :] -= alpha * h_err

However, we found that the similarity is almost all 1, such as. `0.999854`, etc.	

- Then, isnpired by [word2vec (part 3 : Implementation)](http://cpmarkchang.logdown.com/posts/773558-neural-network-word2vec-part-3-implementation), we implemented a for loop in `context_word` and another for loop in `negative samples`


## Reference: 

1. [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
2. [Word2vec (part 3 : Implementation)](http://cpmarkchang.logdown.com/posts/773558-neural-network-word2vec-part-3-implementation)


## Requirements:

- nltk, sklearn, scipy, pandas


