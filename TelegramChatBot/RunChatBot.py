# Телеграм бот-болталка, обученная на dialogflow и FastText-эмбеддинги на ответы-mailru



import os
from telegram.ext  import Updater, CommandHandler, MessageHandler, Filters
import dialogflow
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import Word2Vec, FastText
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import regex as re
import random

from data.keyvault import keyvault


# Settings
DATA_PATH = "data/Otvety.txt"
ANSWERS_PATH = 'data/prepared_answers.txt'
FASTTEXT_MODEL_PATH = 'data/ft_100_model.h5'
RAW_SENTENCES_PATH = 'data/raw_sentences.pkl'
INDEX_MAP_PATH = 'data/index_map.pkl'
FT_INDEX_PATH = 'data/ft_index'
FT_VECTOR_MAX_LEN = 100

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)


def preprocess_txt(line):
    
    """Строковый препроцессинг,  текстовой строки строки"""
    
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls

def clear_sent(text):
    
    """Очистка текста от тегов, обработка лишних знаков препинаний, смайлов и прочего мусора"""
    
    target_sent = random.choice(text)
    target_sent = re.sub(r'<[^>]*>','', target_sent)
    target_sent = re.sub(r'[\xa0]',' ', target_sent)
    target_sent = re.sub(r'.{1,2} [\n]','', target_sent)
    target_sent = re.sub(r'[\n]','', target_sent)
    target_sent = re.sub(r'\s[.]\s','', target_sent)
    target_sent = re.sub(r"\)\)\)\)", "))", target_sent)
    target_sent = re.sub(r"\)\)", ")", target_sent)
    target_sent = re.sub(r"\) \)", "))", target_sent)
    target_sent = re.sub(r',{2,}','', target_sent)
    return target_sent

def get_response(question, index, model, index_map):
    
    """Получение ответа из модели fast text"""
    
    question = preprocess_txt(question)
    vector = np.zeros(FT_VECTOR_MAX_LEN)
    norm = 0
    for word in question:
        if word in model:
            vector += model[word]
            norm += 1
    if norm > 0:
        vector = vector / norm
    answers = index.get_nns_by_vector(vector, 5) 
    return [index_map[i] for i in answers if len(index_map[i]) < 300] # 

# Загрузка заготовок: Fast text модель, индексы, индексированная карта ответов
modelFT = FastText.load(FASTTEXT_MODEL_PATH)
ft_index = annoy.AnnoyIndex(FT_VECTOR_MAX_LEN, 'angular')
ft_index.load(FT_INDEX_PATH)
with open(INDEX_MAP_PATH, 'rb') as f:
    index_map = pickle.load(f)
    
    
    
    
updater = Updater(token=keyvault['telegram_token'])
dispatcher = updater.dispatcher
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyvault['google_api_key'] # Ключ к google API dialogflow

DIALOGFLOW_PROJECT_ID = keyvault['dialogflow_project_id'] # PROJECT ID DialogFlow 
DIALOGFLOW_LANGUAGE_CODE = 'ru' # языковая группа
SESSION_ID = 'SetMyTasksBot'  # ID телеграмм бота

start_msg = 'R1D1 машет клешнёй! Бот может поддержать беседу веселой болтовнёй и отвечать на глупые вопросы'


def startCommand(bot, update):
    
    bot.send_message(chat_id=update.message.chat_id, text=start_msg)
    logging.info("Start Bot")

def textMessage(bot, update):
    
    # Начало: Поиск интентов на DialogFlow
    input_txt = update.message.text
    print(f"input_txt is: {input_txt}")
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=input_txt, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
         raise
    text = response.query_result.fulfillment_text
    if text and text != 'null':
        bot.send_message(chat_id=update.message.chat_id, text= response.query_result.fulfillment_text)
    else:
        # Интент DialogFlow на найден --> включение режима болталки на "ответах mail.ru"
        print(f'Entering to answers routine')
        answers_response = get_response(input_txt, ft_index, modelFT, index_map)
        answers_response = clear_sent(answers_response)
        bot.send_message(chat_id=update.message.chat_id, text=answers_response)

        
if __name__ == "__main__":

    # Хендлеры
    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, textMessage)
    # Добавляем хендлеры в диспатчер
    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)
    # Начинаем поиск обновлений
    updater.start_polling(clean=True)
    # Останавливаем бота на Ctrl + C
    updater.idle()