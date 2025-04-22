import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # pentru o bară de progres frumoasă

# Calea spre folderul care conține folderele genurilor (fiecare gen are 100 fișiere .wav)
GENRE_DIR = 'genres/'

# Funcție care extrage caracteristicile audio pentru un fișier
def extract_features(file_path):
    try:
        # Încarcă fișierul audio, doar primele 30 secunde
        y, sr = librosa.load(file_path, duration=30)
        
        # Inițializăm un dicționar în care vom pune caracteristicile extrase
        features = {}

        # Rata de zero-crossing (cât de des semnalul audio trece prin axa 0)
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))

        # Chroma STFT - descrie cât de "armonios" este semnalul
        features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

        # MFCC - o reprezentare importantă a spectrului audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc_{i+1}'] = np.mean(mfcc)

        # Spectral contrast - diferențele între frecvențe
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # Tonnetz - caracteristici armonice, legate de ton și acorduri
        features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))

        # Tempo - estimarea vitezei ritmice a piesei
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        return features
    except Exception as e:
        print(f"Eroare la procesarea fișierului {file_path}: {e}")
        return None

def preprocess_dataset(output_csv='genres/features.csv'):
    """
    Funcție principală care parcurge toate fișierele audio din folderul specificat
    și salvează caracteristicile într-un fișier CSV.
    """
    if os.path.exists(output_csv):
        print(f"Fișierul '{output_csv}' există deja. Îl încărcăm...")
        # Dacă fișierul există, îl încărcăm direct pentru procesare ulterioară
        return pd.read_csv(output_csv)
    else:
        data = []  # Listă în care vom stoca fiecare rând de caracteristici + etichetă (gen muzical)

        # Parcurgem fiecare folder (fiecare gen muzical)
        for genre in os.listdir(GENRE_DIR):
            genre_folder = os.path.join(GENRE_DIR, genre)

            if not os.path.isdir(genre_folder):
                continue  # Dacă nu este folder, sărim peste

            # Parcurgem fiecare fișier .au din folderul genului curent
            for filename in tqdm(os.listdir(genre_folder), desc=f"Prelucrăm genul: {genre}"):
                if filename.endswith('.au'):
                    file_path = os.path.join(genre_folder, filename)
                    features = extract_features(file_path)
                    
                    # Dacă extragerea caracteristicilor a fost cu succes
                    if features:
                        # Adăugăm eticheta (genul muzical) la fiecare set de caracteristici
                        features['label'] = genre
                        data.append(features)

        # Convertim lista de dicționare într-un DataFrame pandas
        df = pd.DataFrame(data)

        # Salvăm DataFrame-ul într-un fișier CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Caracteristicile au fost salvate în {output_csv}")

        return df  # Returnăm DataFrame-ul pentru a putea lucra cu el în continuare