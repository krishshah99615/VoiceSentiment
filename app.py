import streamlit as st
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
import sklearn 
import pickle as pk
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import sounddevice as sd
import wavio
def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    
    return result
def draw_embed(embed, name, which):
    """
    Draws an embedding.
    Parameters:
        embed (np.array): array of embedding
        name (str): title of plot
    Return:
        fig: matplotlib figure
    """
    fig, embed_ax = plt.subplots()
    plot_embedding_as_heatmap(embed)
    embed_ax.set_title(name)
    embed_ax.set_aspect("equal", "datalim")
    embed_ax.set_xticks([])
    embed_ax.set_yticks([])
    embed_ax.figure.canvas.draw()
    return fig


def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.
    Parameters:
        voice_sample (str): path to sample of sound
    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=3, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


model_load_state = st.text("Loading pretrained models...")
pickle_off = open("model.pk", "rb")
model = pk.load(pickle_off)
model_load_state.text("Loaded pretrained models!")
st.header("Record your own voice")



if st.button(f"Click to Record"):
    filename = 'sample'
    record_state = st.text("Recording...")
    duration = 3  # seconds
    fs = 48000
    myrecording = record(duration, fs)
    record_state.text(f"Saving sample as {filename}.mp3")

    path_myrecording = f"{filename}.mp3"

    save_record(path_myrecording, myrecording, fs)
    record_state.text(f"Done! Saved sample as {filename}.mp3")

    st.audio(read_audio(path_myrecording))

    fig = create_spectrogram(path_myrecording)
    st.pyplot(fig)

if st.button(f"Click to Predict"):
    p_load_state = st.text("Predicitng...")


    feature=extract_feature('sample.mp3', mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)
    #print(feature.shape)
    p = model.predict(feature)
    st.warning("The predicted sentiment is "+p[0].upper())
    p_load_state.text("Predicted")

