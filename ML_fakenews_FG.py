"""
Created on Thu Oct 14 07:07:05 2021
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
@author: florence
"""

# Import des packages

import os #import des fichiers
import pandas as pd # création de df
import matplotlib.pyplot as plt # plots
import re # nettoyage des données
# repartition des echantillon train et test
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # tokenization
import tensorflow as tf # word to seq
# Méthode de machine learning : random forest
from sklearn.ensemble import RandomForestClassifier
# evaluation des modèles
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import cross_val_score
import seaborn as sns # representation graphique de la matrice de confusion
import numpy as np #vecteurs

# Renseignement du répertoire de travail
os.chdir('E:\M2\machine_learning\projet')

# paramètres graphiques
plt.style.use('ggplot')

# -----------------------------------------------------------------------------------------

# definition des fonctions

# normalisation on enlève les url, les espaces en trop, etc. 
def normalize(df):
    normalized = []
    for i in df:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

# démarche de machine learning

def test(data, y_res, mod): # data = le X à tester, res = la var réponse à comparer, model = modèle entrainé
    data = normalize(data)
    # tokenization
    tokenizer.fit_on_texts(data)
    # création du vecteur
    data = tokenizer.texts_to_sequences(data)
    # tensorflow
    data = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post', maxlen=256)
    # prediction 
    predicted_value = mod.predict(data)
    #Matrice de confusion
    matrix = confusion_matrix(predicted_value, y_res, normalize='all')
    ## Valeur d'accuracy et de précision      
    print('Accuracy on testing set:', accuracy_score(predicted_value, y_res))
    print('Precision on testing set:', precision_score(predicted_value, y_res))
    return(matrix)

# création des graphs (matrice de confusion)

def graph(conf_matrix): # conf_matrix = matrice de confusion
    # params graphiques
    plt.figure(figsize=(16, 10))
    ax= plt.subplot()
    # labels, title and ticks
    ax.set_xlabel('Predicted Labels', size=20)
    ax.set_ylabel('True Labels', size=20)
    ax.set_title('Confusion Matrix', size=20) 
    ax.xaxis.set_ticklabels([0,1], size=15)
    ax.yaxis.set_ticklabels([0,1], size=15)
    # heatmap
    sns.heatmap(conf_matrix, annot=True, ax = ax)

# -------------------------------------------------------------------------------------------

# import des données
Fausses = pd.read_csv("Fake.csv")
Fausses.drop(Fausses.index[5000:23481],0,inplace=True)
Vraies = pd.read_csv("True.csv")
Vraies.drop(Vraies.index[5000:21417],0,inplace=True)

# ajout de la colonne catégorie (à prédire)
Fausses['category'] = 0
Vraies['category'] = 1

# dataframe de travail
df = pd.concat([Fausses,Vraies]) 

# Concatenation des colones text et titre
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

# -------------------------------------------------------------------------------------------

# repartition des echantillon train et test
X_train, X_test, y_train, y_test = train_test_split(df.text, df.category, test_size = 0.2,random_state=2)

##### entrainement d'un modèle
# normalisation
X_train = normalize(X_train)

# tokenization
max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)

# passer de texte à un vecteur numérique
X_train = tokenizer.texts_to_sequences(X_train)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)

# Méthode de machine learning : random forest
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)

# par cv
print('Accuracy du modèle par CV: ', cross_val_score(RandomForest, X_train, y_train, scoring="accuracy", cv = 5).mean())
# accuracy par cv de 0.88


##### Tests du modèle -----------------------------------------------------------------------

# test du modèle sur un échantillon issu du jeu de données 
test_dataset = test(X_test,y_test,RandomForest)
graph(test_dataset)

# utilisation des données aléatoires pour prouver que le jeu de données
# n'est pas biaisé
alea = np.random.choice(a=[False, True], size=(10000, 1), p=[0.5, 1-0.5])  
df["category2"] = alea

test_alea = test(df.text,df.category2,RandomForest)
graph(test_alea)

# test sur un jeu de données personnel
own_data = pd.read_excel(io="notre_base2.xlsx", sheet_name="notre_base2",header=None)
categorie = [0] * 16 + [1] * 15
own_data['category'] = categorie
own_data = own_data.rename(columns={0: 'text'})
own_data = own_data.sample(frac = 1)

test_own_data = test(own_data.text,own_data.category,RandomForest)
graph(test_own_data)













