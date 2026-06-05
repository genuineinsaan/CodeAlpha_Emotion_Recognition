from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename
from datetime import datetime
import os

from src.predict import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "ogg",
    "flac",
    "aac",
    "webm"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "audio_file" not in request.files:

        return (
            "No audio file uploaded."
        )

    file = request.files["audio_file"]

    if file.filename == "":

        return (
            "Please select an audio file."
        )

    if not allowed_file(
        file.filename
    ):

        return (
            "Unsupported audio format."
        )

    filename = secure_filename(
        file.filename
    )

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    emotion, confidence = predict_emotion(
        file_path
    )

    positive_emotions = [
        "happy",
        "calm",
        "surprised"
    ]

    negative_emotions = [
        "sad",
        "angry",
        "fearful",
        "disgust"
    ]

    if emotion.lower() in positive_emotions:

        category = (
            "Positive Emotion"
        )

        emotion_color = (
            "#22c55e"
        )

    elif emotion.lower() in negative_emotions:

        category = (
            "Negative Emotion"
        )

        emotion_color = (
            "#ef4444"
        )

    else:

        category = (
            "Neutral Emotion"
        )

        emotion_color = (
            "#f59e0b"
        )

    if confidence >= 75:

        confidence_level = (
            "High Confidence"
        )

    elif confidence >= 50:

        confidence_level = (
            "Medium Confidence"
        )

    else:

        confidence_level = (
            "Low Confidence"
        )

    prediction_time = datetime.now().strftime(
        "%d %B %Y | %I:%M %p"
    )

    return render_template(
        "result.html",

        emotion=emotion.title(),

        confidence=round(
            confidence,
            2
        ),

        confidence_level=
        confidence_level,

        category=category,

        emotion_color=
        emotion_color,

        filename=filename,

        prediction_time=
        prediction_time,

        audio_path=filename
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )