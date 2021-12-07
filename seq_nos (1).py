from sentence_transformers import SentenceTransformer
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
import re
import nltk
import torch


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



# v  = embedder.encode(['Hey!!', 'Heyy. How are you? How is life? everything gooood???'])
# print(v[0])
# print(v[1])
def get_val(s):
	return list(filter(lambda elem: elem.isnumeric(), s.split(' ')))

def get_label(s, l):
	return list(map(lambda elem: int(elem in l), s.split(' ')))

def generate_labels(data):
	data['nos'] = data[['Prob_clean']].applymap(lambda s : get_val(s))
	data['labels'] = data.apply(lambda r: get_label(r.clean, r.nos), axis=1)
	data['size'] = data[['clean']].applymap(lambda s : len(s.split(' ')))
	N = 30
	data['labels'] = data.apply(lambda r: r['labels'] + [0]*(max(0,N-r['size'])), axis=1)
	#data['labels'] = data[['labels']].applymap(lambda r: torch.Tensor(r[:N]))
	data['labels'] = data[['labels']].applymap(lambda r: r[:N])
	data['size'] = data[['labels']].applymap(lambda s : len(s))
	#print(data['size'].quantile(0.99))
	print('Checker')
	print(data['labels'][0])
	return data['labels']




# Build a model
inputs = layers.Input(shape=(384,))
layer1 = layers.Dense(128, activation='relu')(inputs)
layer2 = layers.Dense(128, activation='relu')(layer1)
predictions = layers.Dense(30, activation='softmax')(layer2)
model = keras.Model(inputs=inputs, outputs=predictions)
print('Model Built')
# Define custom loss
def custom_loss():

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        l = 0
        t = 10.0
        print('Here')
        for i in range(len(y_true)):
        	if y_true[i] > y_pred[i]:
        		l += 1
        	if y_true[i] < y_pred[i]:
        		l += t
        return l
   
    # Return a function
    return loss
    
# Compile the model
model.compile(optimizer='adam',
              loss=custom_loss(), # Call the loss function with the selected layer
              metrics=['accuracy'])
print('Model  compiled')
train = pd.read_json('MathQA/train.json')[0:100]
train['soln'] = 'Category : '+ train.category + '.# ' + train.Rationale.str.replace('"', '')
train['text'] = 'Category : '+ train.category + '.# ' + train.Rationale.str.replace('"', '') + ' # ' + train.Problem
train['clean'] = train[['soln']].applymap(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,lst_stopwords=nltk.corpus.stopwords.words("english")))
train['Prob_clean'] = train[['Problem']].applymap(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,lst_stopwords=nltk.corpus.stopwords.words("english")))
labels = generate_labels(train)
print(labels)
# train
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(embedder.encode(train['clean']))
model.fit(embedder.encode(train['clean'].values), labels)  