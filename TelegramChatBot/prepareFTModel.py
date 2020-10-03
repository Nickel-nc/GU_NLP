import os
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import FastText
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re

# Settings
DATA_PATH = "data/Otvety.txt"
ANSWERS_PATH = 'data/prepared_answers.txt'
RAW_SENTENCES_PATH = 'data/raw_sentences.pkl'
FASTTEXT_100_MODEL_PATH = 'data/ft_100_model.h5'
# FASTTEXT_200_MODEL_PATH = 'data/ft_200_model.h5'
FT_INDEX_PATH = 'data/ft_index'
FT_VECTOR_MAX_LEN = 100
morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)

# Предобработка и токенизация текста
def preprocess_txt(line):
    
    """Разделение текста на токены. Приведение к нижнему регистру, обработка морфорлогии"""
    
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls
    
    
def get_sentences(input_file, len_limit=1500000):
    
    """Предобработка текста. Удаление стоп-слов, токенизация """
    
    # Preprocess for models fitting

    sentences = []

    morpher = MorphAnalyzer()
    sw = set(get_stop_words("ru"))
    exclude = set(string.punctuation)
    c = 0

    with open(input_file, "r", encoding='utf-8') as fin:
        for line in tqdm(fin):
            spls = preprocess_txt(line, exclude, morpher, sw)
            sentences.append(spls)
            c += 1
            if c > len_limit:
                break
                
    return sentences
    
    
# Создание индексов FastText
ft_index = annoy.AnnoyIndex(FT_VECTOR_MAX_LEN ,'angular')

index_map = {}
counter = 0

# Обучение Fast text с максимальными размерами 100 и 200
sentences = [i for i in sentences if len(i) > 2]
modelFT_100 = FastText(sentences=sentences, size=100, min_count=1, window=5)
modelFT_100.save(FASTTEXT_100_MODEL_PATH)

# modelFT_200 = FastText(sentences=sentences, size=200, min_count=1, window=5)
# modelFT_200.save(FASTTEXT_200_MODEL_PATH)

modelFT = modelFT_100

with open(ANSWERS_PATH, "r", encoding='utf-8') as f:
    for line in tqdm(f):
        n_ft = 0
        spls = line.split("\t")
        index_map[counter] = spls[1]
        question = preprocess_txt(spls[0])
        vector_ft = np.zeros(FT_VECTOR_MAX_LEN)
        for word in question:
            if word in modelFT:
                vector_ft += modelFT[word]
                n_ft += 1
        if n_ft > 0:
            vector_ft = vector_ft / n_ft
        ft_index.add_item(counter, vector_ft)
            
        counter += 1

    
sentences = get_sentences(DATA_PATH)

# Сохрание сырых данных
with open(RAW_SENTENCES_PATH, 'wb') as f:
    pickle.dump(sentences, f)
    

# Сохраняем индексы
ft_index.build(10)
ft_index.save(FT_INDEX_PATH)



