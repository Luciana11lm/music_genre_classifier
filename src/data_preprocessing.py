import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # pentru bara de progres 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_analysis import extract_audio_features

GENRE_DIR = 'genres/'

# functia principala care parcurge toate fisierele audio din folderul specificat si salveaza caracteristicile intr-un fisier CSV
def preprocess_dataset(output_csv='genres/features.csv'):
  if os.path.exists(output_csv):
    print(f"Fisierul '{output_csv}' exista deja. Il incarcam.")
    return pd.read_csv(output_csv)   # daca fisierul există, il incărcam direct pentru procesare ulterioara
  else:
    data = []      # lista in care vom stoca fiecare rand de caracteristici + eticheta (gen muzical)
    
    for genre in os.listdir(GENRE_DIR):              # parcurgem fiecare folder (fiecare gen muzical)
      genre_folder = os.path.join(GENRE_DIR, genre)
      if not os.path.isdir(genre_folder):
        continue  # daca nu este folder, sarim peste
      for filename in tqdm(os.listdir(genre_folder), desc=f"Prelucram genul: {genre}"):     # parcurgem fiecare fisier .au din folderul genului curent
        if filename.endswith('.au'):
          file_path = os.path.join(genre_folder, filename)
          try:
            features = extract_audio_features(file_path)        # se extrag caracteristicile folosind functia din data_analysis.py
            for i in range(len(features['mfcc'])):
              features[f'mfcc_{i+1}'] = features['mfcc'][i]
            features.pop('mfcc', None)
            if features:
              features['label'] = genre                         # adaugam eticheta la fiecare set de caracteristici
              data.append(features)
          except Exception as e:
            print(f"Eroare la fisierul {file_path}: {e}")

    df = pd.DataFrame(data)  # convertim lista de dictionare intr-un DataFrame pandas
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  
    df.to_csv(output_csv, index=False)          # salvam DataFrame-ul intr-un fisier CSV
    print(f"Caracteristicile au fost salvate in {output_csv}")

    return df  # returnam DataFrame-ul pentru a putea lucra cu el in continuare la antrenare

def load_and_split_data(test_size=0.2, random_state=42, save_csv=True, output_csv='genres/features.csv'):
  df = preprocess_dataset(output_csv)
  if df.isnull().values.any() or np.isinf(df.select_dtypes(include=[np.number])).values.any():
    print("!!! Exista valori lipsa sau infinite in setul de date !!!")

  train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42) # impartim initial df-ul in train si test 
  if save_csv:
    train_df.to_csv('genres/train.csv', index=False)
    test_df.to_csv('genres/test.csv', index=False)
    print("Seturile de date au fost salvate:")
    print(f" - Train: {train_df.shape[0]} exemple")
    print(f" - Test: {test_df.shape[0]} exemple")

  # se separa coloana de label de restul caracteristicilor      
  X_train = train_df.drop(columns=["label"])  # caracteristicile audio
  y_train = train_df["label"]                 # etichetele (genurile muzicale)
  X_test = test_df.drop(columns=["label"])
  y_test = test_df["label"]

  scaler = StandardScaler()                   # normalizarea valorilor
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled, y_train, y_test