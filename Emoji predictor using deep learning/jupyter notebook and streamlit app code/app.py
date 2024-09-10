import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow.keras
import streamlit as st
from keras.preprocessing.sequence import pad_sequences

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered-title">Emoji Predictor</h1>', unsafe_allow_html=True)

df = pd.read_csv("../data/emoji_uncleaned.csv").sample(n=30)
df.dropna(inplace=True)

X = df["Tweet"].values
y = df["Label"].values

emoji_raw = open('../data/us_mapping.txt','r',encoding="utf8")

emojis=[]
for sentence in emoji_raw:
    sentence = sentence.rstrip()
    emojis.append(sentence)

    
emoji_dict={}

for e in emojis:
    idx = int(e.split()[0])
    emoji = e.split()[1]
    emoji_dict[idx] = emoji


tokenizer = pickle.load(open("../Model/tweet_tokenizer",'rb'))

def preprocess_text(X):
    max_len=40
    X_seqs = tokenizer.texts_to_sequences(X)
    X_seqs_pd = pad_sequences(X_seqs, truncating="pre", padding="pre", maxlen=max_len)
    return X_seqs_pd


import string
import re

from tensorflow import keras
emoji_predict_model = keras.models.load_model("../Model/BLSTM.h5", compile=False)


def tweet_clean(tweet):
    tweet = str(tweet).lower()
    rm_mention = re.sub(r'@[A-Za-z0-9]+', '', tweet)                       
    rm_rt = re.sub(r'RT[/s]+', '', rm_mention)                             
    rm_links = re.sub(r'http\S+', '', rm_rt)                               
    rm_links = re.sub(r'https?:\/\/\S+','', rm_links)
    rm_nums = re.sub('[0-9]+', '', rm_links)                               
    rm_punc = [char for char in rm_nums if char not in string.punctuation] 
    rm_punc = ''.join(rm_punc)
    cleaned = rm_punc
    
    return cleaned


def predict_emoji(text, model=emoji_predict_model):
    text = tweet_clean(text)
    X_sequences = preprocess_text([text])
    predictions = np.argmax(model.predict(X_sequences), axis=1)
    emoji_idx = predictions[0]
    emoji = emoji_dict[emoji_idx]
    
    return emoji



text=st.text_input('Enter the tweet')
if st.button('Predict'):
    st.markdown(f"<h3>Original text</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:24px;'>{text}</p>", unsafe_allow_html=True)
    if text:
        with st.spinner('Predicting Emoji'):
            predicted_emoji = predict_emoji(text)
            st.markdown(f"<h3>Text with the most probabilistic emoji</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'>{text} {predict_emoji(text)}</p>", unsafe_allow_html=True)
    else:
        st.write('Please enter a tweet to predict the emoji.')