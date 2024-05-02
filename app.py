import streamlit as st 
import pandas as pd
import tensorflow as tf
import pickle as pk
from Wav_to_CSV import voice_to_csv

# Loading Models
with open('model_sc.pkl', "rb") as f:
    new_sc = pk.load(f)

model = tf.keras.models.load_model("model.h5")

# Uploaded Audio
audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "opus"])

if st.button("Click"):

    df = voice_to_csv(audio, "acoustics.csv", start=0, end=20)

    st.write(":blue[DataFrame Format of the audio is :]")
    st.dataframe(df)

    X = new_sc.transform(df)
    st.write(":blue[DataFrame Format after standard scaler is :]")
    st.dataframe(X)

    output = model.predict(X)
    if output[0][0] == 0:
        st.write(f"The voice is of :- :red[Male]")
    elif output[0][0] == 1:
        st.write(f"The voice is of :- :red[Female]")