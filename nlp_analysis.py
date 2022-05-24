#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import csv
import string
import re
from collections import Counter
from nltk import ngrams
nltk.download('punkt')
get_ipython().system('pip3 install transformers sentencepiece')
from transformers import MBartTokenizer, MBartForConditionalGeneration


# In[2]:


df = pd.read_excel('/Users/natalyakrauze/Desktop/диплом/with_tags_binary — копия.xlsx')
df


# In[3]:


def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[13]:


common_words = get_top_n_words(df.Title, 20)
for word, freq in common_words:
    print(word, freq)


# In[14]:


non_speaker = re.compile('[A-Za-z]+: (.*)')

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            if all(word not in string.punctuation for word in phrase):
                phrase_counter[phrase] += 1

phrase_counter = Counter()


# In[15]:


extract_phrases(" ".join(df.Title.values), phrase_counter, 2)

most_common_phrases = phrase_counter.most_common(50)
for k,v in most_common_phrases:
    print('{0: <5}'.format(v), k)


# # Title reduction

# In[16]:


df['Title_new'] = df['Title'].str.lower()
df['Title_new'] = df['Title_new'].str.replace("-", " ", regex=False)
df['Title_new'] = df['Title_new'].str.replace("/", " ", regex=False)
df['Title_new'] = df['Title_new'].str.replace(",", "", regex=False)
df['Title_new'] = df['Title_new'].str.replace(".", "", regex=False)
df['Title_new'] = df['Title_new'].str.replace("(", "", regex=False)
df['Title_new'] = df['Title_new'].str.replace(")", "", regex=False)
df['Title_new'] = df['Title_new'].str.replace("&", " ", regex=False)
df['Title_new'] = df['Title_new'].str.replace(" по ", " ", regex=False)
df['Title_new'] = df['Title_new'].str.replace(" с ", " ", regex=False)


# In[17]:


df.Title_new = df.Title_new.replace('analyst', 'аналитик')


# ## 2-gram

# In[19]:


tokens = nltk.word_tokenize(" ".join(df.Title_new.values))
#Create your bigrams
bgs = nltk.bigrams(tokens)
#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
fdist = dict(sorted(fdist.items(), key=lambda item: -item[1]))
for k,v in list(fdist.items())[:15]:
    print(k, v)


# ## 3-gram

# In[131]:


tokens = nltk.word_tokenize(" ".join(df.Title_new.values))
#Create your bigrams
n = 3
onegrams = ngrams(sentence.split(), n)
#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(onegrams)
fdist = dict(sorted(fdist.items(), key=lambda item: -item[1]))
for k,v in list(fdist.items())[:10]:
    print(k, v)


# No such ngrams

# In[132]:


popular_titles = ['аналитик',
                  'системный администратор', 
                  'финансовый аналитик', 
                  'бизнес аналитик', 
                  'business analyst', 
                  'data scientist',
                  'аналитик bi',
                  'junior',
                  'программист',
                  'developer',
                  'project manager',
                  'teamlead',
                  'product manager',
                  'технический директор',
                  'начальник отдела',
                  'руководитель проектов',
                  'начальник отдела',
                  'руководитель отдела',
                  'it',
                  'web']
for tit in popular_titles:
    df[tit] = df['Title_new'].str.contains(tit).astype(int)


# In[133]:


df


# In[134]:


df.to_excel('Title_reduction.xlsx')


# # Uni analysis

# In[2]:


df = pd.read_excel('/Users/natalyakrauze/Desktop/диплом/Title_reduction.xlsx')


# In[3]:


df.head()


# In[4]:


for uni in df['LastUni'].unique():
    print(uni)


# In[5]:


def uni_unification(text):
    if type(text) is not str:
        return text
    
    text = text.lower()
    if ('высшая школа экономики' in text) or ('higher school of economics' in text)     or ('вшэ' in text) or ('hse' in text):
        return 'ВШЭ'
    
    elif ('ельцин' in text) or ('yeltsin' in text):
        return 'Уральский федеральный университет имени первого Президента России Б.Н. Ельцина'
    
    elif (('ломонос' in text) and ('север' not in text) and ('орден' not in text))     or ('мгу ' in text) or ('lomonos' in text):
        return 'МГУ'
    
    elif ('баум' in text) or ('мгту ' in text) or ('bauman' in text):
        return 'МГТУ'
    
    elif (('физико-технический институт' in text) and ('мифи' not in text))     or ('moscow institute of physics and technology' in text):
        return 'Физтех'
    
    elif ('мифи' in text) or ('mephi' in text):
        return 'МИФИ'
    
    elif ('мисис' in text) or ('moscow institute for steel and alloys' in text) or ('misa' in text):
        return 'МИСИС'
    
    elif ('горный университет' in text) and ('петер' in text) or ('спгу ' in text):
        return 'Санкт-Петербургский государственный горный университет'
    
    elif ('политех' in text) and ('томск' in text):
        return 'Национальный исследовательский Томский политехнический университет'
    
    elif ('плехан' in text) or ('plekh' in text):
        return 'РЭА Плеханова'
    
    return text
    
def set_other_uni(value):
    if type(value) is not int:
        return 11
    return value
    
df['fixed_uni_name_1'] = df.apply(lambda x: uni_unification(x['LastUni']), axis = 1)
df['fixed_uni_name_2'] = df.apply(lambda x: uni_unification(x['LastUni_2']), axis = 1) 

rename_dict = {'ВШЭ' : 1,
               'МГУ' : 2,
               'МГТУ' : 3,
               'Физтех' : 4,
               'МИФИ' : 5,
               'МИСИС' : 6,
               'Санкт-Петербургский государственный горный университет': 7,
               'РЭА Плеханова': 8,
               'Уральский федеральный университет имени первого Президента России Б.Н. Ельцина': 9,
               'Национальный исследовательский Томский политехнический университет': 10}
    
    
df['uni_1_code'] = df['fixed_uni_name_1'].replace(rename_dict)
df['uni_2_code'] = df['fixed_uni_name_2'].replace(rename_dict)

df['uni_1_code'] = df.apply(lambda x: set_other_uni(x['uni_1_code']), axis = 1)
df['uni_2_code'] = df.apply(lambda x: set_other_uni(x['uni_2_code']), axis = 1) 
df


# In[6]:


# ВШЭ
df[df['uni_1_code'] == 2][['LastUni', 'fixed_uni_name_1', 'uni_1_code']].drop_duplicates()


# In[7]:


# МГУ
df[df['uni_1_code'] == 2]


# In[8]:


# Бауманка
df[df['uni_1_code'] == 3]


# In[9]:


# Физтех
df[df['uni_1_code'] == 4]


# In[10]:


df.to_excel('final_clean_dataset.xlsx')


# In[ ]:




