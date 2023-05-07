
import PyPDF2
import glob
import os
import argparse
import pickle
import pandas as pd
#from sklearn.cluster import KMeans
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--document")
   args = parser.parse_args()






if __name__ == "__main__":
    main()
