# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:14:25 2018

@author: User
"""
import re
import nltk
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import numpy as np
import itertools
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
lemmetizer = lemmetizer = FrenchLefffLemmatizer()

def tokenize(text):
    text = text.replace("'"," ")
    text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
    text = re.sub(r'\@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove tweeter names
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'\(Crédits :.+\)\r\n', ' ', text, flags=re.MULTILINE) # remove credits from start of article
    text = re.sub(r'\r\n.+\r\nVoir les réactions', '', text, flags=re.MULTILINE) # remove credits from start of article
    text = text.replace("/ REUTERS", "").replace("/ AFP", "")
    s = text.split("(Reuters) - ",1)
    if len(s)>1:
        text = s[1]
    s = text.split(" - ",1)
    if len(s[0]) < 35:
        text = s[1]
    text = re.sub(r'\r\npar .+\r\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\r\n\(.+\)', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nÀ LIRE AUSSI\r\n.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE AUSSI\r\n.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE AUSSI >>.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\n» LIRE AUSSI -.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLE FIGARO. -.+ - ', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'www.\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'\r\n» Vous pouvez également suivre.+.', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE NOTRE DOSSIER COMPLET\r\n.+\r\n', '', text, flags=re.MULTILINE) 
    
    text = text.replace("\r\nLIRE AUSSI :\r\n»", "")
    text = text.replace("(Reuters)", "").replace("Article réservé aux abonnés", " ")
    text = text.translate({ord(ch): None for ch in '0123456789'})
    text = text.translate({ord(ch): " " for ch in '-•“’!"#$%&()*+,./:;<=>?@[\\]^_`{|}~«»–…‘'}) # lower + suppress numbers
    text = re.sub(r' +', ' ', text, flags=re.MULTILINE) # remove autor name
    text = re.sub(r' \b[a-zA-Z]\b ', ' ', text, flags=re.MULTILINE) ### mono letters word
    tokens = nltk.word_tokenize(text.lower(), language='french')  
    tokens = " ".join([lemmetizer.lemmatize(lemmetizer.lemmatize(lemmetizer.lemmatize(word, "v"), "n"), "a") for word in tokens])
    return tokens


def load_doc2vec_model():
    fname = get_tmpfile(os.path["DIR_PATH"] + "/data/doc2vec/v2/doc2vec_articles_181030")
    model = Doc2Vec.load(fname) 
    return model


def from_output_to_classe(y, classes):
    yp = np.where(y >= 0.75, 1, 0)
    pred = [list(itertools.chain(*l.tolist())) for l in list(map(lambda x : np.argwhere(x == np.amax(x)), yp))]
    
    predictions = []
    for a in pred:
        sub = []
        if len(a) < 4:
            for id_ in a:
                sub.append(classes[id_])
            predictions.append(sub)
        else:
            predictions.append(None)

    return predictions