"""
app.py — Flask Web Application for Speech Emotion Recognition
================================================================
Provides a beautiful, modern web interface where users can:
  1. Upload a .wav or .mp3 audio file
  2. Get the predicted emotion with confidence scores
  3. View a probability bar-chart for all emotion classes

Routes:
  GET  /           → Upload page
  POST /predict    → Process audio and return results
"""

import os
import uuid
import traceback

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

from src.predict import predict_emotion

# ── App Configuration ─────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Emoji map for each emotion ────────────────────────────────────────────
EMOTION_EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle file upload, run prediction, and render results.

    Returns the results page on success, or redirects back with an
    error flash message on failure.
    """
    # ── Validate upload ───────────────────────────────────────────────
    if "audio_file" not in request.files:
        flash("No file uploaded. Please select an audio file.", "error")
        return redirect(url_for("index"))

    file = request.files["audio_file"]

    if file.filename == "":
        flash("No file selected. Please choose a .wav or .mp3 file.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            "Unsupported file type. Only .wav and .mp3 files are accepted.",
            "error",
        )
        return redirect(url_for("index"))

    # ── Save uploaded file ────────────────────────────────────────────
    try:
        # Use a unique filename to avoid collisions
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)
    except Exception:
        flash("Failed to save uploaded file. Please try again.", "error")
        return redirect(url_for("index"))

    # ── Run prediction ────────────────────────────────────────────────
    try:
        result = predict_emotion(filepath)
    except FileNotFoundError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except ValueError as exc:
        flash(f"Audio processing error: {exc}", "error")
        return redirect(url_for("index"))
    except Exception:
        traceback.print_exc()
        flash(
            "An unexpected error occurred during prediction. "
            "Please try a different file.",
            "error",
        )
        return redirect(url_for("index"))

    # ── Prepare template data ─────────────────────────────────────────
    emotion = result["emotion"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    # Sort probabilities descending for the chart
    sorted_probs = sorted(
        probabilities.items(), key=lambda x: x[1], reverse=True
    )

    # Build the URL for audio playback in the browser
    # filepath is like "static/uploads/abc123.wav" — strip the leading "static/"
    relative_path = filepath.replace("\\", "/")
    if relative_path.startswith("static/"):
        relative_path = relative_path[len("static/"):]
    audio_url = url_for("static", filename=relative_path)

    return render_template(
        "result.html",
        emotion=emotion,
        emoji=EMOTION_EMOJI.get(emotion, "🎭"),
        confidence=confidence,
        confidence_pct=f"{confidence * 100:.1f}",
        sorted_probs=sorted_probs,
        filename=secure_filename(file.filename),
        audio_url=audio_url,
        audio_ext=ext,
    )


# ── Error Handlers ────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(error):
    """Handle files exceeding the size limit."""
    flash("File too large! Maximum allowed size is 16 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template("index.html"), 404


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  🎙️  Speech Emotion Recognition — Flask Server")
    print("  📍  http://127.0.0.1:5000")
    print("═" * 55 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
