from flask import Flask, render_template, request, jsonify
from models.load_models import load_all_models
import torch
import whisper
import os
from pydub import AudioSegment

# Ajouter le chemin pour ffmpeg (Windows)
os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"

# Flask setup avec bons chemins
app = Flask(
    __name__,
    static_folder="assets",         # 📁 Dossier des fichiers CSS/JS/images
    template_folder="template"      # 📁 Dossier contenant index.html
)

# Chemins des fichiers temporaires
UPLOAD_FOLDER = "temp.wav"
CONVERTED_FILE = "converted.wav"

# Chargement des modèles
print("Chargement des modèles IA...")
whisper_model, emotion_model, processor = load_all_models()
print(">> Modèles chargés avec succès ✅")

# Mapping des labels
id2label = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
    6: "surprise"
}
emotion_model.config.id2label = id2label

# Page principale (tout le site)
@app.route("/")
def index():
    return render_template("index.html")

# Upload de l’audio
@app.route("/upload", methods=["POST"])
def upload_audio():
    audio = request.files["audio_data"]
    audio.save(UPLOAD_FOLDER)
    return jsonify({"message": "Audio reçu avec succès."})

# Transcription avec Whisper
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({"transcription": "Fichier audio non trouvé."})

    audio = AudioSegment.from_file(UPLOAD_FOLDER)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(CONVERTED_FILE, format="wav")

    result = whisper_model.transcribe(CONVERTED_FILE)
    print(">> Transcription :", result["text"])
    return jsonify({"transcription": result["text"]})

# Prédiction de l’émotion
@app.route("/predict", methods=["POST"])
def predict_emotion():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({"emotion": "Fichier audio non trouvé."})

    audio = AudioSegment.from_file(UPLOAD_FOLDER)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)

    inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()

    predicted_emotion = id2label.get(predicted_id, "unknown")
    print(">> Émotion détectée :", predicted_emotion)
    return jsonify({"emotion": predicted_emotion})

# Démarrage de l'app Flask
if __name__ == "__main__":
    app.run(debug=True)
