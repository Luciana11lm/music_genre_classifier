import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------------------
# FUNCȚIA 1: Încărcarea și împărțirea datelor în train/test
# -----------------------------------------------------
def load_and_split_data(test_size=0.2, random_state=42):
    """
    Încarcă datele din features.csv și le împarte în seturi de antrenare și testare.
    
    :param test_size: proporția din date care va fi alocată setului de testare
    :param random_state: valoare fixă pentru reproducibilitate
    :return: tuple cu X_train, X_test, y_train, y_test (datele procesate)
    """
    # Încarcă datele extrase anterior în fișierul CSV
    df = pd.read_csv("genres/features.csv")

    # Separă coloana de label-uri (gen muzical) de restul caracteristicilor
    X = df.drop(columns=["label"])  # Caracteristicile audio
    y = df["label"]                 # Etichetele (genurile muzicale)

    # Normalizează caracteristicile pentru a avea valori comparabile
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Împarte datele în set de antrenament și testare
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------
# FUNCȚIA 2: Antrenarea mai multor modele KNN pentru a alege cel mai bun
# -----------------------------------------------------
def train_knn(X_train, X_test, y_train, y_test, k_values=[3, 5, 7, 9]):
    """
    Antrenează mai multe modele KNN cu diferite valori k și returnează cel mai performant model.
    
    :param X_train: datele de antrenament
    :param X_test: datele de test
    :param y_train: etichetele de antrenament
    :param y_test: etichetele de test
    :param k_values: lista de valori k pentru care testăm KNN
    :return: modelul KNN cu cea mai mare acuratețe și valoarea k corespunzătoare
    """
    best_k = None
    best_accuracy = 0
    best_model = None

    for k in k_values:
        # Creează modelul KNN cu valoarea k curentă
        model = KNeighborsClassifier(n_neighbors=k)

        # Antrenează modelul pe datele de antrenament
        model.fit(X_train, y_train)

        # Face predicții pe datele de test
        y_pred = model.predict(X_test)

        # Calculează acuratețea
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acuratețea pentru KNN cu k={k}: {accuracy:.4f}")

        # Salvează modelul cu cea mai mare acuratețe
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_model = model

    print(f"Cel mai bun model KNN are k={best_k} cu acuratețea: {best_accuracy:.4f}")
    return best_model


# -----------------------------------------------------
# FUNCȚIA 3: Antrenarea modelului Random Forest
# -----------------------------------------------------
def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100):
    """
    Antrenează un model Random Forest și returnează acuratețea și modelul.
    
    :param X_train: datele de antrenament
    :param X_test: datele de test
    :param y_train: etichetele de antrenament
    :param y_test: etichetele de test
    :param n_estimators: numărul de arbori în pădure
    :return: modelul Random Forest antrenat
    """
    # Creează modelul Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Antrenează modelul
    model.fit(X_train, y_train)

    # Face predicții
    y_pred = model.predict(X_test)

    # Calculează acuratețea
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acuratețea pentru Random Forest: {accuracy:.4f}")

    return model


# -----------------------------------------------------
# FUNCȚIA 4: Antrenează ambele modele și alege cel mai bun
# -----------------------------------------------------
def train_best_model():
    """
    Încărcă datele, antrenează modelele KNN și Random Forest și returnează cel mai bun model.
    """
    # Împărțim datele în seturi de train și test
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Antrenăm modelul KNN
    knn_model = train_knn(X_train, X_test, y_train, y_test)

    # Antrenăm modelul Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Poți alege aici manual care model să fie folosit în continuare
    # Sau poți compara acuratețile și returna cel mai bun
    return rf_model  # Sau knn_model, dacă acela e mai bun
