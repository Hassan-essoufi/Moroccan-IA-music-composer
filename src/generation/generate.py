import os
import numpy as np
import tensorflow as tf
import time
from src.models.transformer_decoder import TransformerDecoder
import midi_neural_processor.processor as midi_tokenizer
from src.generation.sampler import sample_next_token

def generate_music(model_path, config, gen_file, max_duration=30.0):
    """
    Generate a MIDI file.
    """
    start_time = time.time()
    
    # Config
    output_midi_dir = config.get("output", {}).get("midi_dir", "generated_midis")
    max_seq_len = config["data"]["max_seq_len"]
    
    vocab_size = config["data"].get("vocab_size", max_seq_len + 1)
    
    seed_midi_path = config["generation"]["seed_midi_path"]
    
    model_load_start = time.time()
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'TransformerDecoder': TransformerDecoder}
        )
    except:
        model = tf.keras.models.load_model(model_path, compile=False)
    
    model.trainable = False
    
    generated = []
    
    if seed_midi_path and os.path.exists(seed_midi_path):
        print(f"Encoding seed MIDI: {seed_midi_path}")
        try:
            seed_tokens = midi_tokenizer.encode_midi(seed_midi_path)
            generated = seed_tokens[-max_seq_len:].copy() if len(seed_tokens) > max_seq_len else seed_tokens.copy()
            print(f"   Using {len(generated)} seed tokens")
        except Exception as e:
            print(f"error encoding seed MIDI: {e}")
            generated = [np.random.randint(1, vocab_size - 1)]
    else:
        generated = [np.random.randint(1, vocab_size - 1)]
        print("No seed MIDI, starting with random token")
    
    # Generation parametres
    temperature = config["generation"].get("temperature", 1.0)
    top_k = config["generation"].get("top_k", 50)
    top_p = config["generation"].get("top_p", 0.9)
    
    @tf.function
    def predict_step(input_tensor):
        return model(input_tensor, training=False)
    
    print("🎹 Generating music...")
    
    tokens_per_second = 15
    max_tokens = int(max_duration * tokens_per_second)
    tokens_generated = 0
    
    estimated_time_per_token = 0.1
    
    for i in range(max_tokens):
        
        if len(generated) > max_seq_len:
            input_seq = generated[-max_seq_len:]
        else:
            input_seq = generated
        
        input_tensor = tf.constant([input_seq], dtype=tf.int32)
        
        # Prediction
        logits = predict_step(input_tensor)
        next_logits = logits[0, -1].numpy()
        
        # Sampling
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Token validation
        next_id = int(next_id)
        if next_id >= vocab_size or next_id < 0:
            next_id = np.random.randint(1, vocab_size - 1)
        
        generated.append(next_id)
        tokens_generated += 1
    
    # 6. Decoding & Saving
    print("🎼 Decoding to MIDI...")
    try:
        midi_data = midi_tokenizer.decode_midi(generated)
        
        os.makedirs(output_midi_dir, exist_ok=True)
        output_path = os.path.join(output_midi_dir, gen_file)
        
        if not output_path.lower().endswith(('.mid', '.midi')):
            output_path += '.mid'
        
        midi_data.write(output_path)
        
        total_time = time.time() - start_time
        print(f"Music generated and saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"error saving MIDI: {e}")
        return None