from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: Optional[str] = ""      
    length: Optional[int] = 128     # Number of events/tokens to generate 
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 5

class GenerateResponse(BaseModel):
    midi_file_path: str             # Path to generated MIDI
    audio_file_path: str            # Path to generated WAV
    success: bool
    message: Optional[str] = None
