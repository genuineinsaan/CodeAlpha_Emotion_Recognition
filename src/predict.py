import os
import librosa
import numpy as np
import joblib

from pydub import AudioSegment
from tensorflow.keras.models import load_model

# ==================================
# LOAD MODEL
# ==================================

model = load_model(
    "models/emotion_model.h5"
)

# ==================================
# LOAD LABEL ENCODER
# ==================================

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# ==================================
# LOAD SCALER
# ==================================

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==================================
# CONVERT AUDIO TO WAV
# ==================================

def convert_to_wav(audio_path):

    extension = os.path.splitext(
        audio_path
    )[1].lower()

    if extension == ".wav":

        return audio_path

    wav_path = (
        audio_path.rsplit(".", 1)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        audio_path
    )

    audio.export(
        wav_path,
        format="wav"
    )

    return wav_path

# ==================================
# PREDICT EMOTION
# ==================================

def predict_emotion(audio_path):

    try:

        audio_path = convert_to_wav(
            audio_path
        )

        signal, sample_rate = librosa.load(
            audio_path,
            sr=22050
        )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(
            mfcc.T,
            axis=0
        )

        mfcc_mean = mfcc_mean.reshape(
            1,
            -1
        )

        # ==========================
        # APPLY SCALER
        # ==========================

        mfcc_mean = scaler.transform(
            mfcc_mean
        )

        prediction = model.predict(
            mfcc_mean,
            verbose=0
        )

        emotion_index = np.argmax(
            prediction
        )

        emotion = encoder.inverse_transform(
            [emotion_index]
        )[0]

        confidence = round(
            float(
                np.max(prediction)
            ) * 100,
            2
        )

        return (
            emotion,
            confidence
        )

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return (
            "neutral",
            0.0
        )