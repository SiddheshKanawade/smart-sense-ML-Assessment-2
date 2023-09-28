import os
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

from datapreprocessing import DataPreprocessing
from model import Model

from config import DATASET_PATH, QUESTION


def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, sentences):
  max_sim = -1
  index_sim = -1
  for index, embedding in enumerate(sentence_embeddings):
    sim = cosine_similarity(embedding, question_embedding)[0][0]
    if sim > max_sim:
      max_sim = sim
      index_sim = index
  
  return index_sim


model_processor=Model()
model=model_processor.get_model()

for file in os.listdir(DATASET_PATH):
    filename=os.path.join(DATASET_PATH, file)
    if filename.endswith(".pdf"):
        processor=DataPreprocessing(filename)
        print(filename)
        pdf_text=processor.load_data(filename)
        tokens=processor.get_tokens(pdf_text)
        cleaned_sentences=processor.get_cleaned_sentences(tokens, stopwords=True)
        cleaned_sentences_with_stopwords = processor.get_cleaned_sentences(tokens, stopwords=False)
        sentences, dictionary, bow_corpus=processor.get_word2vec(cleaned_sentences_with_stopwords)
        
        # question
        question = processor.clean_sentence(QUESTION, stopwords=False)
        question_embedding = dictionary.doc2bow(question.split())
        
        sent_embeddings=[]
        for sent in sentences:
            sent_embeddings.append(model_processor.get_phrase_embedding(sent, model))
        
        question_embedding = model_processor.get_phrase_embedding(question, model)
        index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)
        print("Question: ", question)
        try:
          
          print("Answer: ", cleaned_sentences_with_stopwords[index])
        except:
          print("Try another question")
        print('\n')
        break
    else:
        print("\n")
        print(f"{filename} is not a pdf file")
        print("\n")
        continue

