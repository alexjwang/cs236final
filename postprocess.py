from sentence_transformers import SentenceTransformer
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
import re
import nltk
import torch
import numpy as np
import tensorflow as tf
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

class Comparer():
    def __init__(self, l=1.0):
        self.l = l
        self.pattern = r'[0-9][0-9]*'

    def convert(self, s):
        nums = list(filter(lambda elem: elem.isnumeric(), s.split(' ')))
        new_string = re.sub(self.pattern, 'Var', s)
        return new_string, nums

    def lossy_numbers(self, l1, l2):
        loss = 0
        for i in range(min(len(l1), len(l2))):
            if l1[i] != l2[i]:
                loss += 1
        loss += max(len(l1), len(l2)) - min(len(l1), len(l2))
        return loss


    def loss_sentences(self, s1, s2):
        m1, l1 = self.convert(s1)
        m2, l2 = self.convert(s2)
        #print(fuzz.ratio(m1,m2))
        return fuzz.ratio(m1,m2)/100.0 + self.l*(self.lossy_numbers(l1, l2))



# v  = embedder.encode(['Hey!!', 'Heyy. How are you? How is life? everything gooood???'])
# print(v[0])
# print(v[1])
def get_val(s):
	return list(filter(lambda elem: elem.isnumeric(), s.split(' ')))

def get_label(s, l):
	return list(map(lambda elem: 2*int(elem in l)-1, s.split(' ')))

def generate_numbers(data, c):
    data['labels'] = data.apply(lambda r: c.loss_sentences(r['clean'], r['Prob_clean']) , axis=1)
    #data['labels'] = np.log(data['labels'])
    data['labels'] = data['labels']/data['labels'].max()
    return data['labels']




# Build a model
inputs = layers.Input(shape=(384,))
layer1 = layers.Dense(128, activation='relu')(inputs)
layer2 = layers.Dense(128)(layer1)
layer3 = keras.layers.LeakyReLU(alpha=0.3)(layer2)
predictions = layers.Dense(1, activation='sigmoid')(layer3)
model = keras.Model(inputs=inputs, outputs=predictions)
print('Model Built')
# Define custom loss
    
# Compile the model
model.compile(optimizer='adam',
              loss='mse', # Call the loss function with the selected layer
              metrics=['accuracy'])
print('Model  compiled')
train = pd.read_json('MathQA/train.json')
train['soln'] = 'Category : '+ train.category + '.# ' + train.Rationale.str.replace('"', '')
train['text'] = 'Category : '+ train.category + '.# ' + train.Rationale.str.replace('"', '') + ' # ' + train.Problem
train['clean'] = train[['soln']].applymap(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,lst_stopwords=nltk.corpus.stopwords.words("english")))
train['Prob_clean'] = train[['Problem']].applymap(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,lst_stopwords=nltk.corpus.stopwords.words("english")))
labels = generate_numbers(train, Comparer())
print(len(labels))
#print(labels)
#print(labels.type)
# train
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(embedder.encode(train['clean']))
model.fit(embedder.encode(train['clean'].values), np.array(labels))