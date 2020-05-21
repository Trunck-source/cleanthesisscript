#%%
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
import gensim.downloader as api
import torch
import os
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from scipy.spatial import distance
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
stop_words = set(stopwords.words('english')) 
print(torch.cuda.is_available())

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

#%%
path = "C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/clean data/"

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

#%%

import os

filenames= os.listdir ("C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/clean data/") # get all files' and folders' names in the current directory

filesresult = []
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(os.path.abspath("C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/clean data/"), filename)): # check whether the current object is a folder or not
        filesresult.append(filename)
        
filesresult.sort()

fullpaths = []

for i in filesresult: 
    fullpaths.append(path+i) 

#filesresult = filesresult[45:56]
#fullpaths = fullpaths[45:56]
#%%

for it, i in enumerate(fullpaths):
    files = os.listdir(i)
    textfiles = []
    for texts in files: 
        if texts.endswith('.txt'):
            textfiles.append(texts)

    d = {}
    for t in textfiles:
        organisation = re.match('.+?(?=\.)', t)
        d[organisation[0]] = pd.DataFrame(index=[organisation[0]])

    for f in textfiles: 
        with open(i + "/" + f, encoding="latin1") as file_in:
            print("working on" + " " + f)
            organisation = re.match('.+?(?=\.)', f)
            text = file_in.read()
            lines = text.split('.')

            corpus = lines
            corpus = list(filter(lambda x: x != "", corpus))
            corpus = list(filter(lambda x: len(str(x)) > 12, corpus))
        
            corpus_embeddings = embedder.encode(corpus)
            corpus_embeddings = np.asarray(corpus_embeddings)
            corpus_scaled = scale(corpus_embeddings, -1,1)
            corpmean = np.mean(corpus_scaled, axis=0)
        
            for m in textfiles: 
                with open(i+ "/" + m, encoding="latin1") as file_in_2:
                    print("Also working on" + " " + m)
                    organisation2 = re.match('.+?(?=\.)', m)
                    text = file_in_2.read()
                    lines = text.split('.')

                    queries = lines
                    queries = list(filter(lambda x: x != "", queries))
                    queries = list(filter(lambda x: len(str(x)) > 12, queries))

                    query_embeddings = embedder.encode(queries)
                    query_embeddings = np.asarray(query_embeddings)
                    query_scaled = scale(query_embeddings, -1,1)
                    querymean = np.mean(query_scaled, axis=0)

                    results = distance.euclidean(corpmean, querymean)
                    d[organisation[0]].loc[:,organisation2[0]] = results

    df = pd.DataFrame()
    for row in d: 
        df = df.append(pd.DataFrame(d[row]))

    directory = "C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/matrices/"

    os.chdir(directory)
    os.mkdir(filesresult[it])
    os.chdir(directory+filesresult[it])

    df.to_csv(filesresult[it] + ".csv")


# %%
