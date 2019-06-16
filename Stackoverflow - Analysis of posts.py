%matplotlib inline

import pandas as pd
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

files = ["D1.csv","D2.csv"]
#files = ["D1.csv"]

for file in files:
	data = pd.read_csv(file)

	## Word vocabulary
	data_stats1 = []
	
	## Hold each record 
	data_stats2 = []
	
	## Hold count of words in each line
	data_stats3 = []
	
	data_stats4 = []
	
	cnt = len(data['Body'])
	for each in list(range(cnt)):
		each_text = re.sub(r'\b[0-9]+\b\s*', '', re.sub(r'[^\w\s]',' ',BeautifulSoup(data['Body'][each],'lxml').text.replace('\n', ''))).split()
		data_stats1 = data_stats1 + each_text
		data_stats2.append(BeautifulSoup(data['Body'][each],'lxml').text.replace('\n', ''))
		data_stats3.append(len(each_text))
		
	count_data = Counter(data_stats1)
	sorted_count_data = sorted(count_data.items(), key=lambda x: x[1],reverse=True)
	sorted_words = [word[0] for word in sorted_count_data]

	## Remove stop words
	stop_words = set(stopwords.words('english'))
	filtered_words = [w for w in sorted_words if not w in stop_words]	
		
	## Identify Nouns only
	filtered_words_tags = nltk.pos_tag(filtered_words)
	filtered_nouns = [word for word,pos in filtered_words_tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
	filtered_verbs = [word for word,pos in filtered_words_tags if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ')]
	filtered_adj = [word for word,pos in filtered_words_tags if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS')]
	filtered_adv = [word for word,pos in filtered_words_tags if (pos == 'RB' or pos == 'RBR' or pos == 'RBS')]
		
	t = PrettyTable(['Top words', 'Top words (Stopwords excluded)','Top Nouns'])

	for i in range(20):
		t.add_row([sorted_words[i],filtered_words[i],filtered_nouns[i]])
	
	## POS of all words (excluding stop words)
	#words_tags = nltk.pos_tag(filtered_words)
	data_stats4_xaxis = ["Noun","Verb","Adective","Adverbs"]
	data_stats4.append(len(filtered_nouns))
	data_stats4.append(len(filtered_verbs))
	data_stats4.append(len(filtered_adj))
	data_stats4.append(len(filtered_adv))		
	
	## Print Results
	print("    Statistics for file: " + str(file))
	print(t)
	print(" ")
	
	wordcloud_string = " ".join(filtered_words)
	wordcloud = WordCloud().generate(wordcloud_string)
	
	# plot the WordCloud image                        
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0)
	img_name = 'CS17EMDS11028_T1_' + file.split(".")[0] + '_01' + '.png'
	plt.savefig(img_name)
	plt.show()
				
	x_axis = list(range(len(data_stats3)))
	
	## Plot of count of words for each line
	plt.plot(x_axis,data_stats3,'r-')
	plt.xlabel('Line#', fontsize=18)
	plt.ylabel('# of words', fontsize=16)
	title_name = "Plot of number of words in each line"
	plt.title(title_name)
	img_name = 'CS17EMDS11028_T1_' + file.split(".")[0] + '_02' + '.png'
	plt.savefig(img_name)
	plt.show()
	
	## Bar chart of various words
	plt.bar(data_stats4_xaxis,data_stats4)
	plt.xlabel('POS Tag', fontsize=18)
	plt.ylabel('Count', fontsize=16)
	title_name = "Plot of POS tag vs Count"
	plt.title(title_name)
	img_name = 'CS17EMDS11028_T1_' + file.split(".")[0] + '_03' + '.png'
	plt.savefig(img_name)
	plt.show()