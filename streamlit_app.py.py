import streamlit as st
from transformers import pipeline
import numpy as np
from io import BytesIO
from scipy.io import wavfile

SENTIMENT_REPO = "frangipaninpools/Group2"

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment  = pipeline("sentiment-analysis", model=SENTIMENT_REPO, tokenizer=SENTIMENT_REPO)
    tts        = pipeline("text-to-audio", model="facebook/mms-tts-eng")
    return summarizer, sentiment, tts

summarizer, sentiment_ft, tts = load_models()

st.title("Market News Analyzer")
news = st.text_area("Please enter the news here", height=220)

if st.button("Run"):
    summary = summarizer(news, max_length=130, min_length=40, do_sample=False)[0]["summary_text"]
    sent = sentiment_ft(summary)[0]

    st.subheader("Summary of News")
    st.write(summary)

    st.subheader("Sentiment Analysis")
    st.write(f"[{sent['label']} | {sent['score']:.3f}]")

    spoken = f"Summary. {summary}. Sentiment. {sent['label']}."
    audio_out = tts(spoken)

    audio_arr = np.asarray(audio_out["audio"], dtype=np.float32).squeeze().reshape(-1)
    sr = int(audio_out["sampling_rate"])
    wav_int16 = (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16)

    buf = BytesIO()
    wavfile.write(buf, sr, wav_int16)
    buf.seek(0)
    st.audio(buf.read(), format="audio/wav")
