from src.predict import predict_emotion

audio_file = "dataset/Actor_01/03-01-03-01-01-01-01.wav"

emotion, confidence = predict_emotion(
    audio_file
)

print("Emotion :", emotion)
print("Confidence :", confidence, "%")