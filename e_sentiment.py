import streamlit as st
import numpy as np
import pandas as pd
import re #regular expression
import string
import nltk #membantu pengolahan teks
import swifter #untuk membantu tahap stemming
import os
import os.path
import sys
import ast
import gensim
nltk.download('stopwords')
nltk.download('punkt')

from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

st.title('E-Sentiment')
st.header('Upload Data')
uploaded_file = st.file_uploader("Choose a file")

with st.spinner('Mohon Tunggu'):
    if uploaded_file is not None:
      data = pd.read_csv(uploaded_file)
      print("Data :")
      data

from PIL import Image
image = Image.open('chart.png')
st.image(image, caption='Visualization with Chart')
image2 = Image.open('stylecloud.png')
st.image(image2, caption='Visualization with Word Cloud')


#baca file dr local drive untuk tahap pre-processing - normalisasi
normalization = pd.read_csv('normalization.csv')
normalization_dict = {}

#inisialisasi variabel sw diisi dengan stopwords word bahasa inggris
sw = stopwords.words('english')

#inisialisasi dictionary untuk stemming
stemmer = SnowballStemmer(language='english')
term_dict = {}

class preprocessing :
  ##FILTERING
  def filtering(text):
    #hapus tab, baris baru
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    #hapus unsur unsur non ASCII (emoticon, chinese word, dll)
    text = text.encode('ascii', 'replace').decode('ascii')
    #hapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #hapus URL
    return text.replace("http://", " ").replace("https://", " ")
  
  data['tweet_filtering'] = data['tweet'].apply(filtering)

  ##CASE FOLDING -> mengubah huruf kapital ke huruf kecil
  data['tweet_casefolding'] = data['tweet_filtering'].str.lower()

  ##TOKENIZING -> memisahkan kalimat menjadi toke kata-kata
  #hapus punctuation atau tanda baca
  def remove_punct(text):
    return text.translate(str.maketrans("", "", string.punctuation))
  
  data['tweet_tokens'] = data['tweet_casefolding'].apply(remove_punct)

  #ubah multiple whitespace ke single whitespace
  def remove_ws_mult(text):
    return re.sub('\s+', ' ', text)
  
  data['tweet_tokens'] = data['tweet_tokens'].apply(remove_ws_mult)

  #hapus single char
  def remove_sc(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
  
  data['tweet_tokens'] = data['tweet_tokens'].apply(remove_sc)

  #nltk word tokenize
  def word_tokenize_wrapper(text):
    return word_tokenize(text)
  
  data['tweet_tokens'] = data['tweet_tokens'].apply(word_tokenize_wrapper)

  ##NORMALISASI
  for index, row in normalization.iterrows():
    if row[0] not in normalization_dict:
      normalization_dict[row[0]] = row[1]
  
  def normalized_term(document):
    return [normalization_dict[term] if term in normalization_dict else term for term in document]  
  data['tweet_tokens_normalized'] = data['tweet_tokens'].apply(normalized_term)

  ##STOPWORD ->hapus stopwords atau kata-kata yang tidak perlu
  #penambahan kata-kata lain ke sw
  sw.extend(['i', 'ive', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'youre', "you're", 'youve', "you've", 
             'youll', "you'll", 'youd', "you'd", 'your', 'yours', 'yourself', 'yourselves', 'girls', 'girl', 'boys', 'boy', 'he', 'him', 'his', 'himself', 
             'she', 'shes', "she's", 'her', 'hers', 'herself', 'it', 'its', "it's", 'its', 'itself', 'they', 'em', 'them', 'their', 
             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'thatll', "that'll", 'these', 
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
             'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
             'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
             'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
             'same', 'so', 'than', 'too', 'very', 's', 't', 'cant', 'can', 'will', 'just', 'don', 'dont', "don't", 'should', 
             'shouldve', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'arent', "aren't", 'couldn', 
             'couldnt', "couldn't", 'didn', 'didnt', "didn't", 'doesn', 'doesnt', "doesn't", 'hadn', 'hadnt', "hadn't", 'hasn', 'hasnt',
             "hasn't", 'haven', 'havent', "haven't", 'isn', 'isnt', "isn't", 'ma', 'mightn', 'mightnt', "mightn't", 'mustn', 'mustnt', "mustn't",
             'needn', 'neednt', "needn't", 'shan', 'shant', "shan't", 'shouldn', 'shouldnt', "shouldn't", 'wasn', 'wasnt', "wasn't", 'weren', 'werent', 
             "weren't", 'won', 'wont', "won't", 'wouldn', 'wouldnt', "wouldn't"])
  
  #convert list to dictionary
  sw = set(sw)

  def remove_stopwords(words):
    return [word for word in words if word not in sw]

  data['tweet_tokens_sw'] = data['tweet_tokens_normalized'].apply(remove_stopwords)

  ##STEMMING -> mengubah setiap kata menjadi kata dasar
  def stemmed_wrapper(term):
    return stemmer.stem(term)
  
  for document in data['tweet_tokens_sw']:
    for term in document :
      if term not in term_dict:
        term_dict[term] = ' '
  
  for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)

  #apply stemming to data
  def stemmed_term(document):
    return [term_dict[term] for term in document]
  
  data['tweet_tokens_stem'] = data['tweet_tokens_sw'].swifter.apply(stemmed_term)

  data['tweet_cleaned'] = data['tweet_tokens_stem']

#duplikat df data ke df datacleaned
datacleaned = data
datacleaned.to_csv('datacleaned.csv')

##panggil class preprocessing
pre = preprocessing()
st.header('Pre-processing')
st.subheader('Tahap Pre-processing : Filtering')
data['tweet_filtering']
st.subheader('Tahap Pre-processing : Case Folding')
data['tweet_casefolding']
st.subheader('Tahap Pre-processing : Tokenizing')
data['tweet_tokens']
st.subheader('Tahap Pre-processing : Normalization')
data['tweet_tokens_normalized']
st.subheader('Tahap Pre-processing : Stopwords')
data['tweet_tokens_sw']
st.subheader('Tahap Pre-processing : Stemming')
data['tweet_tokens_stem']

#----------------------------------------------------------

class word2vec:
  w2vec_model = gensim.models.Word2Vec(datacleaned['tweet_cleaned'], vector_size=50, window=2, workers=4, min_count=1, sg=0)
  w2vec_model.train(datacleaned['tweet_cleaned'], total_examples = len(datacleaned['tweet_cleaned']), epochs=20)

  #vectorization word w2v
  w2v_words = list(w2vec_model.wv.index_to_key)
  vector=[]
  for sent in tqdm(datacleaned['tweet_cleaned']):
    sent_vec=np.zeros(50)
    count = 0
    for word in sent:
      if word in w2v_words:
        vec = w2vec_model.wv[word]
        sent_vec += vec
        count += 1
    if count != 0:
      sent_vec /= count #normalize
    vector.append(sent_vec)

#panggil class word2vec
w2v = word2vec()
st.header('Word2Vec')
st.subheader('Tahap Word2Vec : Words Vectorization')
vec_w2v = w2v.vector
vec_w2v

#----------------------------------------------------------

class dtree:
  #token word into one string
  def join_text_list(texts):
    texts = ast.literal_eval(str(texts))
    return ' '.join([text for text in texts])
  
  datacleaned["tweet_join"] = datacleaned["tweet_cleaned"].apply(join_text_list)
  datacleaned['tweet_join'] = datacleaned['tweet_join'].astype(str)

  vectorizer = CountVectorizer()
  corpus = vectorizer.fit_transform(datacleaned['tweet_join'])
  
  #split data tweet_join (dtree without w2v)
  x = corpus.toarray()
  y = datacleaned['sentiment'].values

  data_split = x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

  #split data w2v
  X = vec_w2v
  Y = datacleaned['sentiment'].values

  data_split_w2v = X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2)

  ##BUILD MODEL TANPA WORD2VEC
  dt = DecisionTreeClassifier(max_depth=23, criterion='entropy', random_state=1)
  
  #fit dt to the training set
  dt.fit(x_train, y_train)

  #predict test set labels
  y_pred = dt.predict(x_test)
  acc_dt = accuracy_score(y_test, y_pred)

  ##BUILD MODEL TANPA WORD2VEC
  dtw2v = DecisionTreeClassifier(max_depth=23, criterion='entropy', random_state=1)
  
  #fit dt to the training set
  dtw2v.fit(X_train, Y_train)

  #predict test set labels
  Y_pred = dtw2v.predict(X_test)
  acc_dt_w2v = accuracy_score(Y_test, Y_pred)

#panggil class dtree
classification = dtree()
st.header('Classification')
st.subheader('Accuracy : Decision Tree without Word2Vec')
acc = classification.acc_dt
st.write(acc)
st.subheader('Accuracy : Decision Tree with Word2Vec')
acc_w2v = classification.acc_dt_w2v
st.write(acc_w2v)