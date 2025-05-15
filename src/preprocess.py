import pandas as pd
from langdetect import detect
from googletrans import Translator
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

translator = Translator()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, src=lang, dest='en').text
    except:
        pass
    return text

def preprocess_text(text):
    text = detect_and_translate(text)
    text = clean_text(text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_dataframe(df, text_column='text'):
    df['clean_text'] = df[text_column].apply(preprocess_text)
    return df
