# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:41:58 2021

@author: killskill
"""

import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

def cleaningText(text):
    text_clean = re.sub(r'@[A-Za-z0-9]+', '', str(text)) # Hapus Mention
    text_clean = re.sub(r'#[A-Za-z0-9]+', '', text_clean) # Hapus hashtag
    text_clean = re.sub(r'RT[\s]', '', text_clean) # Hapus RT
    text_clean = re.sub(r"http\S+", '', text_clean) # Hapus link
    text_clean = re.sub(r'[0-9]+', '', text_clean) # Hapus angka
    text_clean = text_clean.replace('\n', ' ') # Ganti enter ke spasi
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
    text_clean = text_clean.strip(' ') # Hapus spasi tdk jelas
    return text_clean

def casefoldingText(text): # merubah huruf menjadi lower
    lwr = text
    map(str.lower, lwr)
    text = lwr
    return text


tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
def tokenizingText(text): # memisah kata per kata
    text = tokenizer.tokenize(text)
                               
    return text

def filteringText(text): # Menghilangkan kata-kata yang sering muncul dalam dokumen, akan tetapi tidak memberikan arti yang signifikan
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

def stemmingText(text): # membuang kata kata berimbuhan dan mengambil makna 
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text