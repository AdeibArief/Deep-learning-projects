import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import string
import re
import pandas as pd

FEATURES_PATH = r'C:\Users\HP\Desktop\2 assignment projects\caption generator\Models\features.pkl'
MODEL_PATH = r'C:\Users\HP\Desktop\2 assignment projects\caption generator\Models\best_model.h5'
TOKENIZER_PATH = r'C:\Users\HP\Desktop\2 assignment projects\caption generator\Models\tokenizer.pkl'

model = load_model(MODEL_PATH)
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 35  
vocab_size = len(tokenizer.word_index) + 1

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        if word not in in_text:  
            in_text += " " + word
        if word == 'endofsen':
            break
    final_caption = in_text.replace('startseq ', '').strip()
    if final_caption.endswith('endofsen'):
        final_caption = final_caption[:-5]  
    return final_caption

EMOJI_MODEL_PATH = r"c:\Users\HP\Desktop\2 assignment projects\Emoji predictor using deep learning\Model\BLSTM.h5"
EMOJI_TOKENIZER_PATH = r"c:\Users\HP\Desktop\2 assignment projects\Emoji predictor using deep learning\Model\tweet_tokenizer"
EMOJI_MAPPING_PATH = r"c:\Users\HP\Desktop\2 assignment projects\Emoji predictor using deep learning\data\us_mapping.txt"

emoji_predict_model = load_model(EMOJI_MODEL_PATH)
tokenizer_emoji = pickle.load(open(EMOJI_TOKENIZER_PATH, 'rb'))

emoji_raw = open(EMOJI_MAPPING_PATH,'r',encoding="utf8")
emojis=[]
for sentence in emoji_raw:
    sentence = sentence.rstrip()
    emojis.append(sentence)
emoji_dict={}

for e in emojis:
    idx = int(e.split()[0])
    emoji = e.split()[1]
    emoji_dict[idx] = emoji

def preprocess_text(X):
    max_len=40
    X_seqs = tokenizer_emoji.texts_to_sequences(X)
    X_seqs_pd = pad_sequences(X_seqs, truncating="pre", padding="pre", maxlen=max_len)
    return X_seqs_pd

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

def predict_emoji(text):
    text = tweet_clean(text)
    X_sequences = preprocess_text([text])
    predictions = np.argmax(emoji_predict_model.predict(X_sequences), axis=1)
    emoji_idx = predictions[0]
    emoji = emoji_dict[emoji_idx]
    return emoji

st.title('Funny Image Caption Generator 😂')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_id = uploaded_file.name.split('.')[0]
    if image_id in features:
        feature = features[image_id]
        with st.spinner('Generating caption...'):
            caption = predict_caption(model, feature, tokenizer, max_length)
            emoji_prediction = predict_emoji(caption)
        st.write(f'{caption}{emoji_prediction}')
        
    else:
        st.write('No features found for this image.')
