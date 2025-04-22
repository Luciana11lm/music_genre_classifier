import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GENRE_DIR = 'genres/'

def get_genre_distribution(data_path=GENRE_DIR):
  genres = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]     # lista cu numele tuturor folderelor/ genurilor muzicale din directorul specificat
  genre_counts = {genre: len(os.listdir(os.path.join(data_path, genre))) for genre in genres}  # dictionar cu atribuirea gen: nr de fisiere audio din acel gen 
    
  # vizualizare distributie
  plt.figure(figsize=(10,6))
  sns.barplot(x=list(genre_counts.keys()), y=list(genre_counts.values()))
  plt.title("Distributia fisierelor pe genuri muzicale")
  plt.xlabel("Gen muzical")
  plt.ylabel("Numar fisiere")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig('genres/distributia_genurilor.png')
  plt.show()
    
  return genre_counts

def extract_audio_features(file_path):
  try: 
    y, sr = librosa.load(file_path, duration=30) # extragerea vectorului de amplitudini pe intervale de timp discrete si rata de exantionare
    features = {}                                # initializam un dictionar in care vom pune caracteristicile extrase

    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))                         # rata de zero-crossing (cat de des semnalul audio trece prin axa 0)
    features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))               # chroma STFT - descrie cat de "armonios" este semnalul
    features['mfcc'] = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)                   # MFCCs (13 coeficienti default)
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))   # spectral contrast - diferentele intre frecvente
    features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)) # tonnetz - caracteristici armonice, legate de ton si acorduri
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)                                               # tempo - estimarea vitezei ritmice a piesei
    features['tempo'] = float(tempo)
    
    return features
  except Exception as e:
    print(f"Eroare la procesarea fisierului {file_path}: {e}")
    return None

def plot_feature_comparison_across_genres(data_path=GENRE_DIR):
  feature_names = ['zcr', 'chroma_stft', 'mfcc', 'spectral_contrast', 'tonnetz', 'tempo']
  genre_features = {}

  for genre in os.listdir(data_path):                    # pentru fiecare director/ gen muzical se extrage primul fisier pentru care se extrag caracteristicile si se afiseaza un grafic comparativ
    genre_folder = os.path.join(data_path, genre)
    if not os.path.isdir(genre_folder):
      continue

    for filename in os.listdir(genre_folder):
      if filename.endswith('.au'):
        file_path = os.path.join(genre_folder, filename)
        features = extract_audio_features(file_path)
        if features:
          genre_features[genre] = features
          break                                         # iesim dupa primul fisier
                  
  # afisarem caracteristici
  plt.figure(figsize=(12, 6))
  for i, feature in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    values = [
      np.mean(genre_features[genre][feature]) if isinstance(genre_features[genre][feature], np.ndarray)
      else genre_features[genre][feature]
      for genre in genre_features
    ]
    genres = list(genre_features.keys())
    plt.bar(genres, values)
    plt.title(feature)
    plt.xticks(rotation=45)

  plt.tight_layout()
  plt.suptitle("Comparatie intre genuri pe baza caracteristicilor audio", fontsize=16, y=1.05)
  plt.savefig('genres/comparatie_genuri.png')
  plt.show()

def visualize_mfcc_example(file_path):
  y, sr = librosa.load(file_path)
  mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(mfcc, x_axis='time')
  plt.colorbar()
  plt.title('MFCC')
  plt.tight_layout()
  plt.show()

def analyze_example_file(file_path= 'classical/classical.00000.au'):
  example_path = os.path.join(GENRE_DIR, file_path)
  features = extract_audio_features(example_path)
    
  print("Exemplu caracteristici extrase:")
  for key, value in features.items():
    if isinstance(value, np.ndarray):
      print(f"{key}: {value[:5]}...")  # doar primele 5 valori pentru MFCC
    else:
      print(f"{key}: {value}")

  visualize_mfcc_example(example_path)

def run_full_analysis():
  print("[1] Analiza distributiei genurilor")
  genre_counts = get_genre_distribution()

  print("[2] Compararea caracteristicilor audio intre genuri")
  plot_feature_comparison_across_genres()

  print("[3] Analiza unui fisier de exemplu")
  analyze_example_file()