from gensim.models import Word2Vec
import gensim.downloader as downloader

import numpy as np

class Model():    
    def get_model(self):
        model=None
        try:
            print("\nLoading w2v Model \n")
            model = gensim.models.Keyedvectors.load('./w2vecmodel.mod')
            print("\nw2v Model Successfully loaded \n")
        except:
            model = downloader.load('word2vec-google-news-300')
            model.save("./w2vecmodel.mod")
            print("\n w2v Model Saved \n")
            
        return model
    
    def get_word_vec(self, word, model):
        samp = model['pc']
        vec = [0]*len(samp)
        try:
            vec = model[word]
        except:
            vec = [0]*len(samp)
        return (vec)
    
    def get_phrase_embedding(self, phrase, embeddingmodel):
        samp = Model.get_word_vec(self, 'computer', embeddingmodel)
        vec = np.array([0]*len(samp))
        den = 0;
        for word in phrase.split():
            den = den+1
            vec = vec+np.array(Model.get_word_vec(self, word, embeddingmodel))
        return vec.reshape(1, -1)
        