import os
import librosa
import pretty_midi
import numpy as np

def wav_to_midi(wav_path, output_dir, segment_duration_seconds=30):
    """
    Converts a WAV file into MIDI files using fixed-duration segments.
    Returns ALL segments, not just the first one.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    try:
        y, sr = librosa.load(wav_path, sr=22050)
    except Exception as e:
        print(f"Audio loading error {wav_path}: {e}")
        return []
    
    total_duration = len(y) / sr
    num_segments = int(np.ceil(total_duration / segment_duration_seconds))
    
    print(f"\nProcessing: {os.path.basename(wav_path)}")
    print(f"   Duration: {total_duration:.1f}s → {num_segments} segments of {segment_duration_seconds}s")
    
    midi_paths = []
    total_notes = 0
    
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    for seg_num in range(num_segments):
        # Compute segment indices
        start_time = seg_num * segment_duration_seconds
        end_time = min((seg_num + 1) * segment_duration_seconds, total_duration)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment
        y_segment = y[start_sample:end_sample]
        
        if len(y_segment) == 0:
            continue
        
        # Create MIDI for this segment
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=32, name="Gnawa")
        
        # Onset detection for this segment
        onset_times = librosa.onset.onset_detect(
            y=y_segment, 
            sr=sr, 
            units='time',
            backtrack=True
        )
        
        note_count = 0
        
        # Add notes based on onsets
        for onset_time in onset_times:
            # Use a random pitch within the Gnawa range
            gnawa_notes = [36, 38, 40, 41, 43, 45, 47, 48] 
            pitch = np.random.choice(gnawa_notes)
            
            note = pretty_midi.Note(
                velocity=np.random.randint(70, 100),
                pitch=pitch,
                start=onset_time,
                end=onset_time + np.random.uniform(0.2, 0.8)
            )
            instrument.notes.append(note)
            note_count += 1
        
        # Only save if we have notes
        if note_count > 0:
            midi.instruments.append(instrument)
            
            # Save MIDI
            midi_filename = f"{base_name}_seg{seg_num+1}_{start_time:.0f}s-{end_time:.0f}s.mid"
            midi_path = os.path.join(output_dir, midi_filename)
            
            midi.write(midi_path)
            midi_paths.append(midi_path)
            total_notes += note_count
            
            print(f"   Segment {seg_num+1}: {note_count} notes ({start_time:.0f}-{end_time:.0f}s)")
        else:
            print(f"   Segment {seg_num+1}: No notes detected")
    
    # Return ALL created MIDI paths
    return midi_paths, total_notes


if __name__ == "__main__":
    raw_dir = 'data/raw/moroccan_midi/gnawa'
    output_dir = 'data/raw/moroccan_midi/gnawa_midi'
    
    os.makedirs(output_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"Processing {len(wav_files)} audio files...")
    
    total_segments_created = 0
    total_notes_created = 0
    
    for i in range(len(wav_files)):
        wav_path = os.path.join(raw_dir, wav_files[i])
        
        # Convert to MIDI - get ALL segments
        midi_paths, notes_count = wav_to_midi(wav_path, output_dir)
        
        if midi_paths:
            total_segments_created += len(midi_paths)
            total_notes_created += notes_count
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Audio files processed: {len(wav_files)}")
    print(f"MIDI segments created: {total_segments_created}")
    print(f"Total notes generated: {total_notes_created}")
    print(f"Output directory: {os.path.abspath(output_dir)}")