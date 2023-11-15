import pandas as pd
import numpy as np
import seaborn as sns
import nltk
nltk.download('punkt')
import re
import json
import pickle
import matplotlib.pyplot as plt

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
more_stopword = ['co', 'id','republika','oleh']


data = pd.read_csv("Data_latih.csv")
#filter label = 0 adalah fake news
#filter label = 1 adalah true
fake = data.loc[data['label'] == 0]
true = data.loc[data['label'] == 1]



data['judul'] = data['judul'].apply(lambda x: x.lower())
for index, row in data.iterrows():
  text = row['judul'].split(' ')
  if text[0] == 'jakarta':
    del text[0]
  data.iloc[index, data.columns.get_loc('judul')] = ' '.join(text)

# token for unormalize data
# stop word removed
unnormalize = []
# fakeunormalize = []
# trueunormalize = []
for index, row in data.iterrows():
  text = row['judul']
  # text = stopword.remove(text)
  # text = text.encode("utf-8")
  # text_decode = str(text.decode("utf-8"))
  unnormalize += nltk.word_tokenize(text)


#remove Duplicates
unnormalize_clean = list( dict.fromkeys(unnormalize) )


def textEncode(text):
  text = text.encode("utf-8")
  text_decode = str(text.decode("utf-8"))
def casefolding(review):
  review = review.lower()
  return review
def tokenize(review):
  token = nltk.word_tokenize(review)
  return token
def filtering(review):
  # Remove link web
  review = re.sub(r'http\S+', '', review)
  # Remove @username
  review = re.sub('@[^\s]+', '', review)
  # Remove #tagger
  review = re.sub(r'#([^\s]+)', '', review)
  # Remove angka termasuk angka yang berada dalam string
  # Remove non ASCII chars
  review = re.sub(r'[^\x00-\x7f]', r'', review)
  review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
  review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
  review = re.sub(r'\\u\w\w\w\w', '', review)
  # Remove simbol, angka dan karakter aneh
  review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
  return review
def replaceThreeOrMore(review):
  # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
  pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
  return pattern.sub(r"\1\1", review)
def tokenize(review):
  token = nltk.word_tokenize(review)
  return token
def removeDoubleSpaces(review):
  while '  ' in review:
    review = review.replace('  ', ' ')
  return review
def convertToSlangword(review):
  kamus_slangword = open("slangwords_dict.txt").read() # Membuka dictionary slangword
  # pattern = re.compile(r'\b( ' + ':'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
  kamus_slangword = json.loads(kamus_slangword)
  review = review.split(' ')
  content = []
  for kata in review:
    if kata in kamus_slangword:
      kata = kamus_slangword[kata]
    content.append(kata)
  return ' '.join(content)
def removeStopword(review):
    # stopwords = open(stopwords_Reduced.txt', 'r').read().split()
    # content = []
    # filteredtext = [word for word in review.split() if word not in stopwords]
    # content.append(" ".join(filteredtext))
    # review = content
    review = stopword.remove(review)
    return review
def process_text(s):

    # Check string to see if they are a punctuation
    nopunc = [char for char in s if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('indonesian')]
    return clean_string


  # test = data.iloc[0, data.columns.get_loc('text')]
  # # test = 'km apa khabar'
normalize = []
fake = []
truth = []
for index, row in data.iterrows():
    test= row['judul']
    test = casefolding(test)
    # print(test)
    test = filtering(test)
    # print(test)
    test = replaceThreeOrMore(test)
    # print(test)
    test = removeDoubleSpaces(test)
    # print(test)
    test = convertToSlangword(test)
    # print(test)
    test = removeStopword(test)
    # print(test)
    test = tokenize(test)
    # print(test)
    normalize += test
    if row['label'] == 1:
      truth += test
    else:
      fake += test

#remove Duplicates
normalize_clean = list( dict.fromkeys(normalize) )

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Unnormalize', 'Normalize']
students = [len(unnormalize_clean),len(normalize_clean)]
normalpercent = (len(normalize_clean)/len(unnormalize_clean))*100

from wordcloud import WordCloud
all_words = ' '.join(normalize)
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

all_words = ' '.join(fake)
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)


all_words = ' '.join(truth)
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)


token_phrase = normalize
frequency = nltk.FreqDist(token_phrase)
df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
#jika ingin menghitung kata yang sering muncul gabungan antara fake & real
#df_frequency = df_frequency.nlargest(columns = "Frequency", n = 20)
#plt.figure(figsize=(12,8))
#ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
#ax.set(ylabel = "Count")
#plt.xticks(rotation='vertical')
#plt.title("Freq Word")
#plt.show()

token_phrase = fake
frequency = nltk.FreqDist(token_phrase)
df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
df_frequency = df_frequency.nlargest(columns = "Frequency", n = 20)

token_phrase = truth
frequency = nltk.FreqDist(token_phrase)
df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
df_frequency = df_frequency.nlargest(columns = "Frequency", n = 20)


kamus_slangword = open("slangwords_dict.txt").read() # Membuka dictionary slangword
kamus_slangword = json.loads(kamus_slangword)
def process_text(review):
  review = review.lower()
  review = re.sub(r'http\S+', '', review)
  # Remove @username
  review = re.sub('@[^\s]+', '', review)
  # Remove #tagger
  review = re.sub(r'#([^\s]+)', '', review)
  # Remove angka termasuk angka yang berada dalam string
  # Remove non ASCII chars
  review = re.sub(r'[^\x00-\x7f]', r'', review)
  review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
  review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
  review = re.sub(r'\\u\w\w\w\w', '', review)
  # Remove simbol, angka dan karakter aneh
  review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
  # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
  pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
  review = pattern.sub(r"\1\1", review)

  while '  ' in review:
    review = review.replace('  ', ' ')


  # pattern = re.compile(r'\b( ' + ':'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)

  review = review.split(' ')
  content = []
  for kata in review:
    if kata in kamus_slangword:
      kata = kamus_slangword[kata]
    content.append(kata)
  review=' '.join(content)

  # stopwords = open(stopwords_Reduced.txt', 'r').read().split()
  # content = []
  # filteredtext = [word for word in review.split() if word not in stopwords]
  # content.append(" ".join(filteredtext))
  # review = content
  review = stopword.remove(review)
  token = nltk.word_tokenize(review)
  return token
def rebrand(review):
  if review['label'] ==0:
    review['label']='Hoax'
  else:
    review['label']='Fakta'
  return review

data_rebrand = data.apply(rebrand, axis=1)
dataProcesed = data_rebrand.sample(frac = 1)


import string
def process_text2(s):
  nopunc = [char for char in s if char not in string.punctuation]

  # Join the characters again to form the string.
  nopunc = ''.join(nopunc)

  # Convert string to lowercase and remove stopwords
  clean_string = [word for word in nopunc.split() if word.lower() not in new_stopword]
  return clean_string

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Ambil Stopword bawaan
stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopword = ['dengan', 'bahwa', 'ia','oleh']

# Merge stopword
new_stopword = stop_factory + more_stopword

dictionary = ArrayDictionary(new_stopword)
stopword = StopWordRemover(dictionary)

dataProcesed['token'] = dataProcesed['judul'].apply(process_text2)

from sklearn.feature_extraction.text import CountVectorizer
import string

dataProcesed.sample(40)
bow_transformer = CountVectorizer(analyzer=process_text2).fit(dataProcesed['token'])

news_bow = bow_transformer.transform(dataProcesed['token'])
sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)

from sklearn.model_selection import train_test_split

news_train, news_test, status_train, status_test = train_test_split(dataProcesed['judul'], dataProcesed['label'], test_size=0.4)

#untuk random forest, xgboost, dan algoritma lainnya silahkan searching cara penggunaan algoritma tersebut pada pipline
from sklearn import svm #svc : svm.SVC()
from sklearn.naive_bayes import MultinomialNB #naive bayes
from sklearn.pipeline import Pipeline

modelsvm = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', svm.SVC(probability=True)),  # train on TF-IDF vectors w/ SVM Support Vector Classifier
])
tmodelsvm = modelsvm.fit(news_train,status_train)

testsvm = 'polisi tangkap perekayasa chat hoax kapolri dan kapolda jabar'
presvm = tmodelsvm.predict([testsvm])
presvm



import pkgutil

for module in pkgutil.iter_modules():
    print(module.name)