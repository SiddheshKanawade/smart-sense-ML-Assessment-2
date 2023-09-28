import re
import numpy
import gensim
import nltk

from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora

from PyPDF2 import PdfReader

class DataPreprocessing():
    def __init__(self, path):
        self.path=path
    
    def load_data(self, path):
        pdf_text=""
        reader = PdfReader(path)
        for page in reader.pages:
            pdf_text+= page.extract_text()
        return pdf_text
    
    def clean_sentence(self, sentence, stopwords=False):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        if stopwords:
            sentence = remove_stopwords(sentence)
        return sentence
        
    def get_cleaned_sentences(self, tokens, stopwords=False):
        cleaned_sentences = []
        for row in tokens:
            cleaned = DataPreprocessing.clean_sentence(self,row, stopwords)
            cleaned_sentences.append(cleaned)
        return cleaned_sentences
    
    def get_tokens(self, pdf_text):
        nltk.download('punkt')
        tokens = nltk.sent_tokenize(pdf_text)
        return tokens
    
    def get_word2vec(self, cleaned_sentences_with_stopwords):
        sentences = cleaned_sentences_with_stopwords
        sentence_words = [[word for word in document.split()]
                  for document in sentences]
        
        dictionary = corpora.Dictionary(sentence_words)
        bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
        return sentences, dictionary, bow_corpus
    
    
    
    
    
    