import os
import shutil
from pathlib import Path

def copy_maestro_files(base_dir="maestro-v3.0.0", output_dir="data/raw/maestro/files"):
    """
    Copie tous les fichiers MIDI de MAESTRO vers un dossier de sortie.
    
    Args:
        base_dir: Dossier de base contenant MAESTRO
        output_dir: Dossier de destination pour les fichiers MIDI
    """
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Dossier source: {base_dir}")
    print(f"📁 Destination: {output_dir}")
    
    # 1. Trouver tous les fichiers MIDI
    all_midi_files = []
    total_size = 0
    
    for root, dirs, files in os.walk(base_dir):
        # Ignorer les dossiers cachés
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith(('.midi', '.mid')):
                full_path = os.path.join(root, file)
                all_midi_files.append(full_path)
                total_size += os.path.getsize(full_path)
    
    print(f"✅ {len(all_midi_files)} fichiers MIDI trouvés")
    print(f"📊 Taille totale: {total_size / (1024*1024):.1f} MB")
    
    if not all_midi_files:
        print("❌ Aucun fichier MIDI trouvé")
        return []
    
    # 2. Copier tous les fichiers
    print(f"\n📋 Copie des fichiers...")
    
    copied_files = []
    skipped_files = []
    
    for i, midi_path in enumerate(all_midi_files):
        try:
            # Nom de fichier source
            filename = os.path.basename(midi_path)
            dest_path = os.path.join(output_dir, filename)
            
            # Gérer les doublons
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                dest_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            # Copier le fichier
            shutil.copy2(midi_path, dest_path)
            copied_files.append(dest_path)
            
            # Afficher la progression
            if (i + 1) % 100 == 0:
                print(f"  Copié {i+1}/{len(all_midi_files)} fichiers...")
                
        except Exception as e:
            print(f"❌ Erreur avec {os.path.basename(midi_path)}: {str(e)[:50]}")
            skipped_files.append((os.path.basename(midi_path), str(e)[:50]))
    
    # 3. Résumé
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DE LA COPIE")
    print(f"{'='*60}")
    print(f"Fichiers source trouvés: {len(all_midi_files)}")
    print(f"Fichiers copiés avec succès: {len(copied_files)}")
    print(f"Fichiers ignorés/échoués: {len(skipped_files)}")
    print(f"Dossier de destination: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    copy_maestro_files()