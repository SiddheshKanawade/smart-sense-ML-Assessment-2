# smart-sense-ML-Assessment-2

## EDA on Dataset

I have been given a dataset with pdf files as object. Every pdf file represents a brochure a motor-vechicle(car). There are mostly sedan, hatch-backs, SUVs of the common brands used in India. 

Length of brochure is roughly 20-25 pages
Color: RGB

## Task

I have to implement the knowledge of Deep Learning for this task. My script should be capable of providing answers to the questions asked by the user. 

Initial Estimate: I may use fine-tuned BERT for this task or [Word-Vec]('https://spotintelligence.com/2023/02/15/word2vec-for-text-classification/) conversion technique

## Pre-trained Model:

Pre-trained model used for this task is: **word2vec**

Reference: https://spotintelligence.com/2023/02/15/word2vec-for-text-classification/

## Steps followed:

1. Load the data from the given PDFs
2. Pre-process the data: Preprocess the text data by removing stop words, converting all text to lowercase, and removing punctuation using NLTK package
3. Download the Word2Vec model
4. Convert the preprocessed text data to a vector representation using the Word2Vec model.
5. Ask question and get input

## Instructions to Run code

Follow the following steps in sequence
1. `make build`: This will build all the required dependencies
2. `make run`: This will run the model