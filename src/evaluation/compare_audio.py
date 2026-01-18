import pretty_midi
import numpy as np
import librosa
import soundfile as sf

def midi_to_wav(midi_path, wav_path, fs=22050):
    """
    Convert a MIDI file to WAV format.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio = midi_data.fluidsynth(fs=fs)  # Generate a numpy float32 array
    sf.write(wav_path, audio, fs)
    return wav_path

def load_audio(file_path, sr=22050):
    """
    Load a WAV audio file and return a numpy array.
    """
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def compute_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512):
    """
    Compute the Mel spectrogram of an audio signal.
    It returns: np.ndarray: Mel spectrogram in dB.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def compare_midi_files(midi1, midi2, tmp_wav1, tmp_wav2):
    """
    Compare two MIDI files by converting them to WAV and computing
    the cosine similarity between their Mel spectrograms.
    """
    # Convert MIDI files to WAV
    midi_to_wav(midi1, tmp_wav1)
    midi_to_wav(midi2, tmp_wav2)
    
    # Load audio
    y1, sr1 = load_audio(tmp_wav1)
    y2, sr2 = load_audio(tmp_wav2)
    
    # Compute Mel spectrograms
    mel1 = compute_mel_spectrogram(y1, sr1)
    mel2 = compute_mel_spectrogram(y2, sr2)
    
    # Adjust length to match
    min_frames = min(mel1.shape[1], mel2.shape[1])
    mel1 = mel1[:, :min_frames]
    mel2 = mel2[:, :min_frames]
    
    # Flatten and normalize
    vec1 = mel1.flatten() / np.linalg.norm(mel1.flatten())
    vec2 = mel2.flatten() / np.linalg.norm(mel2.flatten())
    
    # Cosine similarity
    similarity = np.dot(vec1, vec2)
    return similarity