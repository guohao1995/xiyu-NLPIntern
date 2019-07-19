import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set(color_codes=True)
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
"""
x=np.arange(-2.0,6.0,0.1)
scores=np.stack([x,np.ones_like(x),0.2*np.ones_like(x)])
plt.plot(x,softmax(scores).T,linewidth=2)
"""


def print_sentence(df, sentence_id=None):
    if not sentence_id:
        sentence_ids = df_test.SentenceId.unique()
        sentence_id = np.random.choice(sentence_ids)
    print("Sentence ID = {}".format(sentence_id))
    return df[df.SentenceId == sentence_id].iloc[:].Phrase

def text_to_num():
    print('ok')

if __name__ == '__main__':
    train= pd.read_csv("./input/train.tsv", sep="\t")
    test = pd.read_csv("./input/test.tsv", sep="\t")

    print(train.head())
    print(train.shape)
    print(train.Phrase)
    train_list = nltk.word_tokenize(train.Phrase)
    #print(train.iloc[0]['Phrase'],'Sentiment - ',train.iloc[0]['Sentiment'])
    
    #freq.sort_values('frequency', ascending=False)
    #print (freq)
    pass
