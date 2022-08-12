# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:43:00 2021

@author: killskill
"""

import pickle
import numpy as np
import pandas as pd


Pkl_Filename = "pickle/X_train.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    X_train = pickle.load(file)

Pkl_Filename = "pickle/y_train.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    y_train = pickle.load(file)

# Navie Bayes Classifier 
class NBClassifier:

    def __init__(self, X_train, y_train, size):  # fungsi yang akan dijalankan pertama kali
      self.X_train = X_train
      self.y_train = y_train
      self.size = size

    def createDictionary(self): # Membuat bag of words (ubah tabel frekwensi)
      dictionary = dict()
    
      for sampel in  X_train:
        for token in sampel:
          dictionary[token] = dictionary.get(token, 0) + 1 # handle null input
      daftar_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
      return dict(daftar_dict)
    
    def train(self): # training dataset
      X_train_dict = self.createDictionary() # bikin tabel frekwensi
      if self.size == 'full':
        self.daftar_kata = list(X_train_dict.keys())
        self.jumlah_kata = dict.fromkeys(self.daftar_kata, None)
        
      else:
        self.daftar_kata = list(X_train_dict.keys())[:int(self.size)]
        self.jumlah_kata = dict.fromkeys(self.daftar_kata, None)
      
      train = pd.DataFrame(columns = ['X_train', 'y_train']) # mulai menghitung prior probability
      train['X_train'] = X_train
      train['y_train'] = y_train

      train_0 = train.copy()[train['y_train'] == 0]
      train_1 = train.copy()[train['y_train'] == 1]
      train_2 = train.copy()[train['y_train'] == 2]

      Prior_0 = train_0.shape[0]/train.shape[0]
      Prior_1 = train_1.shape[0]/train.shape[0]
      Prior_2 = train_2.shape[0]/train.shape[0] # selesai menghitung prior probability
      
      self.Prior = np.array([Prior_0, Prior_1, Prior_2])
        
      def flat(listOfList): # list output to 1 list
        jadi = []
        for elemen in listOfList:
          jadi.extend(elemen)
        return jadi
  
      X_train_0 = flat(train[train['y_train'] == 0]['X_train'].tolist())
      X_train_1 = flat(train[train['y_train'] == 1]['X_train'].tolist())
      X_train_2 = flat(train[train['y_train'] == 2]['X_train'].tolist())
    
      self.X_train_len = np.array([len(X_train_0), len(X_train_1), len(X_train_2)]) # mulai perhitungan kemunculan kata

      for token in self.daftar_kata:
        res = []
        res.insert(0, X_train_0.count(token))
        res.insert(1, X_train_1.count(token))
        res.insert(2, X_train_2.count(token))
        self.jumlah_kata[token] = res
      return self # selesai perhitungan kemunculan kata

    def predict(self, X_test):  # hasil dari training
      pred = []
      for sampel in X_test:
            
        mulai = np.array([1,1,1])
        
        for tokens in sampel: # Hitung conditional probability
          jumlah_vocab = len(self.daftar_kata)
          if tokens in self.daftar_kata:
            prob = ((np.array(self.jumlah_kata[tokens])+1) / (self.X_train_len + jumlah_vocab))
          else:
            prob = ((np.array([0,0,0])+1) / (self.X_train_len + jumlah_vocab))
          mulai = mulai * prob # Akhir Hitung conditional probability
        pos = mulai * self.Prior # Hitung Posterior Probability
        pred.append(np.argmax(pos)) # Akhir hitung Posterior Probability
      return pred, pos
    
    def score(self, pred, labels): #menghitung score hasil dari klasifikasi
      correct = (np.array(pred) == np.array(labels)).sum()
      accuracy = correct/len(pred)
      return correct, accuracy
