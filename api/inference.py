from fastapi import APIRouter
from api.schemas import GenerateRequest, GenerateResponse
from api.metrics import track_request
from pathlib import Path
import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.generation.generate import generate_music
from src.evaluation.compare_audio import midi_to_wav
from src.preprocessing import tokenizer
from src.monitoring.latency import measure_latency
from src.utils import load_config

router = APIRouter()

# Outputs
output_dir = Path("/content/drive/MyDrive/Moroccan-IA-music-composer/outputs/generated_midi")
output_dir.mkdir(parents=True, exist_ok=True)

config = load_config("/content/drive/MyDrive/Moroccan-IA-music-composer/config/generation.yaml")
model_path = "/content/drive/MyDrive/Moroccan-IA-music-composer/models/final_model.keras"

# Generation endpoint
@router.post("/generate", response_model=GenerateResponse)
@measure_latency
@track_request
def generate_music_api(request: GenerateRequest):
    """
    Generate a MIDI sequence using the pre-trained Transformer model.
    """

    try:
        # # unique file name generation
        file_id = uuid.uuid4().hex
        midi_filename = f"generated_{file_id}.midi"
        wav_filename = f"generated_{file_id}.wav"

        # User parametres
        config["generation"]["length"] = request.length
        config["generation"]["temperature"] = request.temperature
        config["generation"]["top_k"] = request.top_k

        if request.prompt:
            config["generation"]["seed_midi_path"] = request.prompt

        # Midi generation
        generate_music(
            model_path=model_path,
            config=config,
            gen_file=midi_filename
        )

        midi_path = os.path.join("/content/drive/MyDrive/Moroccan-IA-music-composer","generated_midi",midi_file)
        wav_path = os.path.join("/content/drive/MyDrive/Moroccan-IA-music-composer", "audio", wav_filename)
        # Conversion: MIDI to WAV
        midi_to_wav(
            midi_path,
            wav_path
        )

        return GenerateResponse(
            midi_file_path=midi_path,
            audio_file_path=wav_path,
            success=True,
            message="Music generated successfully."
        )

    except Exception as e:
        return GenerateResponse(
            midi_file_path="",
            audio_file_path="",
            success=False,
            message=str(e)
        )
