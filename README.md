# Speech Emotion Recognition

A Flask web application and command-line tool that predicts one of eight emotions from a speech recording. The model is a PyTorch CNN-BiLSTM trained on MFCC audio features from the RAVDESS dataset.

> Predictions classify vocal expression only. They do not reliably determine a person's actual emotional state.

## Features

- Classifies: angry, calm, disgust, fearful, happy, neutral, sad, and surprised.
- Upload `.wav` or `.mp3` files through a local web interface (maximum 16 MB).
- Train, evaluate, and predict from the command line.
- Uses speaker-independent train/validation/test splits, so no actor appears in more than one split.

## Project structure

```text
.
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── src/
│   ├── preprocess.py      # RAVDESS feature extraction
│   ├── train.py           # Model training
│   ├── evaluate.py        # Test-set evaluation
│   ├── predict.py         # Command-line inference
│   ├── features.py        # Audio/MFCC helpers
│   ├── model.py           # CNN-BiLSTM architecture
│   └── splits.py          # Actor-independent split logic
├── templates/             # Flask HTML templates
├── static/                # Styles and temporary user uploads
└── tests/                 # Automated tests
```

`data/` is created locally when you download or prepare the dataset. It is deliberately excluded from Git because it can contain large source audio, NumPy arrays, model checkpoints, and evaluation images.

## Publish to GitHub

After creating an empty repository on GitHub, run these commands from the project root, replacing the URL with your repository URL:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

The included `.gitignore` prevents the RAVDESS dataset, model files, temporary uploads, virtual environments, and Python cache files from being uploaded.

## Quick start: run the web app

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows PowerShell**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Obtain a trained checkpoint. Either train one using the instructions below, or place a compatible checkpoint at `data/models/best_ravdess_pytorch.pt`.

5. Start the application:

   ```bash
   python app.py
   ```

   Then open <http://127.0.0.1:5000> and upload a WAV or MP3 recording.

## Train the model

1. Download and extract RAVDESS **Audio_Speech_Actors_01-24** from [Zenodo](https://zenodo.org/records/1188976). Arrange it like this:

   ```text
   data/
   └── Audio_Speech_Actors_01-24/
       ├── Actor_01/
       ├── Actor_02/
       └── ...
   ```

2. Preprocess audio into MFCC arrays:

   ```bash
   python -m src.preprocess --data-dir data/Audio_Speech_Actors_01-24 --output-dir data
   ```

3. Train the CNN-BiLSTM:

   ```bash
   python -m src.train --data-dir data --model-dir data/models
   ```

   The best checkpoint is saved to `data/models/best_ravdess_pytorch.pt`.

## Evaluate and predict

Evaluate the trained model on held-out speakers:

```bash
python -m src.evaluate --data-dir data --model-dir data/models
```

Run a single prediction from the command line:

```bash
python -m src.predict path/to/audio.wav
```

Run automated tests:

```bash
python -m pytest
```

## Notes

- The application uses Flask's development server and is intended for local use.
- The application accepts WAV and MP3 uploads. MP3 decoding may require FFmpeg on some systems.
- RAVDESS is distributed separately; follow its licence and citation requirements when using the dataset.

## License

Add a licence file before publishing if you want to specify reuse terms for this project.
