import pandas as pd
import numpy as np
from pyemd import emd
import gensim
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from texttable import Texttable
from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preProcess(line):
	#each_line = re.sub(r'\b[0-9]+\b\s*', '', re.sub(r'[^\w\s]',' ',BeautifulSoup(data['Body'][each],'lxml').text.replace('\n', '')))
	each_line = re.sub(r"[^a-zA-Z0-9]+", " ", BeautifulSoup(data['Body'][each],'lxml').text.replace('\n', ''))
	each_line = each_line.lower()
	each_line = re.sub(r"\s*[^A-Za-z]+\s*", " ", each_line)
	each_line = re.sub(r"\'s", " 's ", each_line )
	each_line = re.sub(r"\'re", " 're ", each_line)
	each_line = re.sub(r"\'ve", " 've", each_line)
	each_line = re.sub(r"\'d", " 'd ", each_line )
	each_line = re.sub(r"\'ll", " 'll ", each_line )
	each_line = each_line.split()
	each_line = [w for w in each_line if not w in stop_words]
	each_line = [w for w in each_line if w.isalpha()]
	stemmed_words = [stemmer.stem(word) for word in each_line]
	stemmed_line = " ".join(stemmed_words)
	return stemmed_line
	
def preProcessQuery(ipquery):
	#each_line = re.sub(r'\b[0-9]+\b\s*', '', re.sub(r'[^\w\s]',' ',ipquery.replace('\n', '')))
	each_line = re.sub(r"[^a-zA-Z0-9]+", " ",re.sub(r'[^\w\s]',' ',ipquery.replace('\n', '')))
	each_line = each_line.lower()
	each_line = re.sub(r"\s*[^A-Za-z]+\s*", " ", each_line)	
	each_line = re.sub(r"\'s", " 's ", each_line )
	each_line = re.sub(r"\'re", " 're ", each_line)
	each_line = re.sub(r"\'ve", " 've", each_line)
	each_line = re.sub(r"\'d", " 'd ", each_line )
	each_line = re.sub(r"\'ll", " 'll ", each_line )
	each_line = each_line.split()
	each_line = [w for w in each_line if not w in stop_words]
	each_line = [w for w in each_line if w.isalpha()]
	stemmed_words = [stemmer.stem(word) for word in each_line]
	stemmed_line = " ".join(stemmed_words)
	return stemmed_line

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
#file = "D1.csv"
files = ["D1.csv","D2.csv"]

for file in files:
	if file == "D2.csv":
		data = pd.read_csv(file)
	else:
		data = pd.read_csv(file,nrows=3000)
	query_post = int(input("Enter the query post # in file " + file + " : "))
	print(" ")
	print("Printing related posts for below query: ")
	print("--------------------------------------")
	print(BeautifulSoup(data['Body'][query_post],'lxml').text)
	print(" ")
	data_stats = []
	cnt = len(data['Body'])
	for each in list(range(cnt)):
		data_stats.append(preProcess(each))

	tfidf = TfidfVectorizer()
	doc_tfidf = tfidf.fit_transform(data_stats)

	avg_tfidf_list = []
	matching_strings_list = []
	best_tfidf_score_list = []
	best_tfidf_arg_list = []

	query_list = []
	query_list.append(preProcessQuery(BeautifulSoup(data['Body'][query_post],'lxml').text))	
	query_tfidf = tfidf.transform(query_list)

	tfidf_matrix = cosine_similarity(doc_tfidf,query_tfidf)
	tfidf_scores = [item for sublist in tfidf_matrix.tolist() for item in sublist]
	matching_strings = np.argsort(tfidf_scores)[::-1][1:6]
	matching_strings_list.append(matching_strings)
	best_tfidf_score_list.append(sorted(tfidf_scores,reverse=True)[1])

	t = Texttable()
	t.add_row(["Record Type", "Post #","Post"])

	## Print query post	
	t.add_row(["Query Post",query_post,BeautifulSoup(data['Body'][query_post],'lxml').text])

	## Print related posts
	for ix1 in matching_strings:
		t.add_row(["Related Post",ix1,BeautifulSoup(data['Body'][ix1],'lxml').text])
	print(t.draw())
	print(" ")
	print(" ")