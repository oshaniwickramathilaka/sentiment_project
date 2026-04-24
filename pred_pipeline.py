import pandas as pd
import numpy as np
import re
import pickle
import string

from nltk.stem import PorterStemmer
ps=PorterStemmer()

#load model
vocab=pd.read_csv('static/model/vocabulary.txt',header=None)
tokens=vocab[0].tolist()

with open('static/model/model.pickle','rb') as f:
    model=pickle.load(f)

with open('static/model/corpora/stopwords/english','r') as file:
    sw=file.read().splitlines()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text=text.replace(punctuation,'')
    return text



#preprocessing
def preprocessing(text):
    data=pd.DataFrame([text],columns=["tweet"])
    
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*','',x,flags=re.MULTILINE) for x in x.split()))
    data["tweet"]=data["tweet"].apply(remove_punctuations)
    data["tweet"]=data["tweet"].str.replace('\d+','',regex=True)
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split() ))

    return data["tweet"].tolist()




#vectorizer
def vectorizer(ds):
    vectorized_list = []

    for sentence in ds:
        sentence_list = np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_list[i] = 1

        vectorized_list.append(sentence_list)

    return np.asarray(vectorized_list, dtype=np.float32)

#prediction
def get_prediction(vectorized_txt):
    prediction = model.predict(vectorized_txt)
    pred_value = prediction[0]

    print("DEBUG prediction:", pred_value)  # keep for testing

    
    if pred_value == 1:
        return 'positive'
    else:
        return 'negative'
    