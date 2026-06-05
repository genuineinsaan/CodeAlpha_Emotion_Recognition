import os
import librosa
import numpy as np
import pandas as pd

dataset_path = "dataset"

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

for actor_folder in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for audio_file in os.listdir(actor_path):

        if not audio_file.endswith(".wav"):
            continue

        parts = audio_file.split("-")

        if len(parts) < 3:
            continue

        file_path = os.path.join(actor_path, audio_file)

        emotion_code = parts[2]

        if emotion_code not in emotions:
            continue

        emotion = emotions[emotion_code]

        try:

            signal, sample_rate = librosa.load(
                file_path,
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

            features.append(mfcc_mean)
            labels.append(emotion)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

X = np.array(features)

df = pd.DataFrame(X)

df["emotion"] = labels

os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/emotion_features.csv",
    index=False
)

print("\nFeature Extraction Completed")
print("Saved: outputs/emotion_features.csv")
print("Total Samples:", len(df))
print("Total Features:", X.shape[1])