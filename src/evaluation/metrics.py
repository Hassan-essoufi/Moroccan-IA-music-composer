import numpy as np
from collections import Counter
from midi_neural_processor import processor 

def token_entropy(tokens):
    """
    Measure diversity of generated tokens (entropy).
    Works with integer tokens.
    """
    counts = Counter(tokens)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    return entropy

def pitch_range(tokens):
    """
    Compute pitch range from note_on tokens.
    """
    pitches = [processor.get_pitch(t) for t in tokens if processor.is_note_on(t)]
    if not pitches:
        return 0
    return max(pitches) - min(pitches)

def note_density(tokens):
    """
    Compute approximate note density: notes per time_shift token.
    """
    note_count = sum(1 for t in tokens if processor.is_note_on(t))
    time_shift_count = sum(1 for t in tokens if processor.is_time_shift(t))
    return note_count / max(1, time_shift_count)

def evaluate_tokens(tokens):
    """
    Aggregate metrics for a token sequence.
    """
    return {
        "entropy": token_entropy(tokens),
        "pitch_range": pitch_range(tokens),
        "note_density": note_density(tokens),
        "num_tokens": len(tokens)
    }
