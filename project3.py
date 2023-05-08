
from nltk.tokenize import word_tokenize
import PyPDF2
import glob
import os
import argparse
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from nltk import ngrams
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--document")
   args = parser.parse_args()

   doc = args.document
   pdf = open(doc, "rb")
   pdfReader = PyPDF2.PdfReader(pdf)

   text = ''
   for page in pdfReader.pages:
      text += page.extract_text() + "\n"

   city = ''.join(args.document)
   print(city)
   df = pd.DataFrame()
   df['city'] = city
   df['raw_text'] = [text]


   # Tokenize words
   tokens = word_tokenize(text)

   # removing punctuation
   words = [word for word in tokens if word.isalpha()]

   # removing stop words
   no_stops = [w for w in words if not w in stopwords]


   # stemming of words
   porter = PorterStemmer()
   stemmed = [porter.stem(word) for word in no_stops]


   remove = ['smart','citi','baton','roug','east','parish','jersey','new','newark','mobil',
          'miami','detroit','dc','de','moin','rochest','birmingham', 'montgomeri', 'alabama',
          'tucson','smart citi','fremont','pki','ca','the','a','i','t','f','as','in','get',
          'by','abirmingham','sacramento','question','ii','iii','iv','of','m','o','oakland','california',
          'washington','dc','san','josé','s','haven','st','ct', 'lincoln','nebraska', 'use',
          'project','data','louisvil','chattanooga','petersburg','transport','system','x','ü',
          'area', 'oper','inlud','saint','paul','minneapoli']




   supe_clean = []
   supe_clean_string = ''
   for word in stemmed:
      if word not in remove:
         supe_clean.append(word)
         supe_clean_string += word
         supe_clean_string += ' '


   #df['clean_text'] = supe_clean_string


   vec = pickle.load(open('vec.pickle', 'rb'))
   km = pickle.load(open('kmeans.pickle', 'rb'))

   l = [supe_clean_string]
   X = vec.transform(l)
   preds = km.predict(X.toarray())

   df['clean_text'] = l
   df['clusterid'] = preds
   df.to_csv('smartcity_predict.tsv', sep='\t')







if __name__ == "__main__":
    main()










