# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:18:10 2018

@author: User
"""

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import gensim
import tqdm
import re
import logging
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r"\w+")
stopword_set = set(stopwords.words("french")) 
porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def nlp_clean(data):
   data = re.sub(r'\S*@\S*\s?', '', data, flags=re.MULTILINE) # remove email
   data = re.sub(r'http\S+', '', data, flags=re.MULTILINE) # remove web addresses
   new_str = data.lower().translate({ord(ch): None for ch in '0123456789'}) # lower + suppress numbers
   dlist = tokenizer.tokenize(new_str) # punctuation
   dlist = list(set(dlist).difference(stopword_set)) # stopwords
   dlist = [wordnet_lemmatizer.lemmatize(wordnet_lemmatizer.lemmatize(word, "v"), "n") for word in dlist] ## stemming
   return dlist


class LabeledLineSentence(object):
    def __init__(self, doc_list, id_list):
        self.doc_list = doc_list
        self.id = id_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.TaggedDocument(nlp_clean(doc), [self.id[idx]])
            

if __name__ == "__main__":
    full = pd.read_csv(r"D:\data\articles_journaux\all_articles.csv")
    articles = full["article"].tolist()
    id_list = range(full.shape[0])
    
    train_corpus = LabeledLineSentence(articles, id_list)
    
    model = Doc2Vec(size = 300, 
                    alpha = 0.025,
                    min_alpha=0.00025,
                    min_count=5,
                    dm =1, workers=8)
    
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs = 15)
    
    
#    model.infer_vector(gensim.utils.simple_preprocess(full["article"][99]))
#    
#    for i in [0, 110, 10554, 406,1203,54064]:
#        a = nlp_clean(full["article"].iloc[i])
#        new_vector = model.infer_vector(a)
#        print(model.docvecs.most_similar([new_vector]))
        
    
    
    
    