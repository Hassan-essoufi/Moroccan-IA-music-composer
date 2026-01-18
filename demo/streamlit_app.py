import streamlit as st
import requests
import tempfile
import os
from pathlib import Path
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
API_URL = "http://localhost:8000/api/generate"

st.set_page_config(page_title="Moroccan Music Transformer", page_icon="🎵")

st.title("🎵 Moroccan Music Transformer")
st.write("Générez de la musique à partir d'un prompt et écoutez le résultat !")

# -------------------------
# Formulaire utilisateur
# -------------------------
with st.form(key="generate_form"):
    prompt = st.text_input("Seed MIDI / Prompt (optionnel)", "")
    length = st.number_input("Length (number of tokens/events)", min_value=32, max_value=1024, value=128, step=16)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    top_k = st.number_input("Top-k", min_value=1, max_value=50, value=5, step=1)
    
    submit_button = st.form_submit_button("Generate Music 🎶")

# -------------------------
# Appel API
# -------------------------
if submit_button:
    payload = {
        "prompt": prompt,
        "length": int(length),
        "temperature": float(temperature),
        "top_k": int(top_k)
    }

    with st.spinner("Génération en cours..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            data = response.json()

            if data.get("success"):
                st.success(data.get("message", "Music generated!"))

                midi_path = data["midi_file_path"]
                wav_path = data["audio_file_path"]

                # Affichage fichiers
                st.write("**MIDI file:**", midi_path)
                st.write("**WAV file:**", wav_path)

                # Lecture audio
                if os.path.exists(wav_path):
                    audio_file = open(wav_path, "rb")
                    st.audio(audio_file.read(), format="audio/wav")
                else:
                    st.warning("Le fichier WAV n'a pas été trouvé.")

            else:
                st.error(data.get("message", "Erreur pendant la génération"))

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API: {e}")
