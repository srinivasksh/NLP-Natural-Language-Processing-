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

	for ix in range(cnt):
		query_list = []
		query_list.append(preProcessQuery(BeautifulSoup(data['Body'][ix],'lxml').text))	
		query_tfidf = tfidf.transform(query_list)

		tfidf_matrix = cosine_similarity(doc_tfidf,query_tfidf)
		tfidf_scores = [item for sublist in tfidf_matrix.tolist() for item in sublist]
		matching_strings = np.argsort(tfidf_scores)[::-1][1:6]
		matching_strings_list.append(matching_strings)
		best_tfidf_score_list.append(sorted(tfidf_scores,reverse=True)[1])

	## Best Performing Posts
	printed_posts = []
	printed_cnt = 0
	print("Best performing posts for file: " + str(file.split(".")[0]))
	for best_post_ix in range(100):
		t = Texttable()
		t.add_row(["Record Type", "Post #","Post"])

		## Print query post	
		best_matching_post = np.argsort(best_tfidf_score_list)[::-1][best_post_ix]
		if best_matching_post not in printed_posts:
			t.add_row(["Query Post",best_matching_post,BeautifulSoup(data['Body'][best_matching_post],'lxml').text])
			printed_posts.append(best_matching_post)

			## Print related posts
			for ix1 in matching_strings_list[best_matching_post]:
				printed_posts.append(ix1)
				t.add_row(["Related Post",ix1,BeautifulSoup(data['Body'][ix1],'lxml').text])
			print(t.draw())
			print(" ")
			print(" ")
			printed_cnt += 1
			if printed_cnt == 2:
				break
		else:
			continue
			
	## worst Performing Posts
	printed_posts = []
	printed_cnt = 0
	print("Bad performing posts for file: " + str(file.split(".")[0]))
	for best_post_ix in range(100):
		t = Texttable()
		t.add_row(["Record Type", "Post #","Post"])

		## Print query post	
		best_matching_post = np.argsort(best_tfidf_score_list)[best_post_ix]
		if best_matching_post not in printed_posts:
			t.add_row(["Query Post",best_matching_post,BeautifulSoup(data['Body'][best_matching_post],'lxml').text])
			printed_posts.append(best_matching_post)

			## Print related posts
			for ix1 in matching_strings_list[best_matching_post]:
				printed_posts.append(ix1)
				t.add_row(["Related Post",ix1,BeautifulSoup(data['Body'][ix1],'lxml').text])
			print(t.draw())
			print(" ")
			print(" ")
			printed_cnt += 1
			if printed_cnt == 2:
				break
		else:
			continue