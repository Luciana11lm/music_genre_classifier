# music_genre_classifier
.
├── README.md
├── genres/                   # Folderul cu fișierele audio, imaginile și CSV-urile
│   ├── [blues, classical, ...]  # Subfoldere cu fișiere .au
│   ├── features.csv             # Date extrase pentru antrenare
│   ├── train.csv / test.csv     # Seturi de train și test
│   ├── comparatie_genuri.png   # Grafice de analiză
│   ├── confusion_matrix.png
├── models/                  # Modelele antrenate (joblib)
│   ├── model_genre_classifier_knn.joblib
│   └── model_genre_classifier_random_forest.joblib
├── main.py                  # Script principal de rulare
├── src/                     # Cod sursă organizat pe module
│   ├── data_analysis.py         # Vizualizări și analize pe date
│   ├── data_preprocessing.py    # Curățare, split, procesare
│   ├── model_train.py           # Antrenarea modelelor
│   └── model_test.py            # Testare și evaluare modele
