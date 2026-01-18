import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config
from src.generation.generate import generate_music

model_path = "models/final_model.h5"
generated_file = "test.midi"

# Config
config = load_config("config/generation.yaml")

if __name__ == "__main__":
    try: 
        generate_music(model_path, config, generated_file)
        print("Generation tests passed!")
    except Exception as e:
        print("Something wrong")


