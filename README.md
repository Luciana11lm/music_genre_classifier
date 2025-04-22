# Music Genre Classification

Un proiect de clasificare a genurilor muzicale folosind învățarea automată, care procesează fișiere audio și le clasifică în diverse genuri pe baza caracteristicilor audio extrase.

## Descrierea Proiectului

Acest proiect utilizează tehnici de învățare automată pentru a clasifica fișiere audio în funcție de genul muzical. Folosind biblioteci precum **Librosa** pentru extragerea caracteristicilor audio și **Scikit-learn** pentru construirea și antrenarea modelelor, proiectul include patru module principale:

1. **Analiza exploratorie a datelor**
2. **Preprocesarea datelor**
3. **Antrenarea modelelor**
4. **Testarea și evaluarea modelelor**

## Funcționalități

- **Preprocesarea datelor audio**: Extrage caracteristici importante din fișierele audio (ex. MFCC, Spectral Contrast, etc.).
- **Crearea și antrenarea modelelor**: Implementarea algoritmilor de învățare automată precum KNN și Random Forest.
- **Testarea și validarea modelelor**: Calcularea metodelor de evaluare (precizie, recall, F1-score, matrice de confuzie).
- **Vizualizarea datelor**: Grafice pentru distribuția genurilor și compararea caracteristicilor audio între genuri.

---

## Fișiere și Descriere

| Fișier                     | Descriere                                                                                                                                                                          | Stare    |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `data_analysis.py`          | Analiza exploratorie a datelor: obținerea numărului și distribuția variabilelor, examinarea caracteristicilor audio.                                                                | Aprobat  |
| `data_preprocessing.py`     | Curățarea/preprocesarea datelor: tratarea valorilor lipsă, curățarea duplicatelor și a outlierilor, eliminarea coloanelor irelevante.                                              | Aprobat  |
| `model_train.py`            | Crearea și antrenarea unui model inițial: testarea diferitelor valori ale parametrilor pentru KNN, SVM, regresia logistică și NN. Împărțirea datelor în seturi de antrenament/testare. | Aprobat  |
| `model_test.py`             | Testarea și validarea modelului: calcularea și determinarea metricilor (precizie, recall, F1-score, SSE/MSE) pe setul de date de testare.                                          | Aprobat  |

---

## Funcții Descriere

### `data_analysis.py`
- **get_genre_distribution(data_path)**: Crează o listă cu toate genurile muzicale și numărul de fișiere audio asociate fiecărui gen. Afișează un grafic cu distribuția genurilor muzicale.
- **extract_audio_features(file_path)**: Extrage caracteristici audio (ZCR, MFCC, Spectral Contrast, Tempo) folosind biblioteca **Librosa**.
- **plot_feature_comparison_across_genres(data_path)**: Compară caracteristicile audio între diferite genuri muzicale.
- **visualize_mfcc_example(file_path)**: Afișează spectrograma MFCC pentru un fișier audio.
- **analyze_example_file()**: Analizează un fișier audio exemplu.

### `data_preprocessing.py`
- **preprocess_dataset(output_csv)**: Verifică dacă fișierul CSV există deja; dacă nu, creează-l extrăgând caracteristici din fișierele audio.
- **load_and_split_data(test_size, random_state, save_csv)**: Împarte datele în seturi de antrenament și testare, normalizând caracteristicile și salvându-le într-un fișier CSV.

### `model_train.py`
- **train_knn(X_train, X_test, y_train, y_test, k_values)**: Antrenează și testează un model KNN pentru diferite valori de k și returnează modelul cu acuratețea cea mai mare.
- **train_random_forest(X_train, X_test, y_train, y_test, n_estimators)**: Antrenează un model Random Forest cu parametri optimi folosind GridSearchCV.
- **train_best_model(X_train, X_test, y_train, y_test)**: Antrenează ambele modele (KNN și Random Forest) și alegerea modelului cu cele mai bune rezultate pentru testare.

### `model_test.py`
- **Testarea modelului**: Calculează și afişează precizia, recall, F1-score și matricea de confuzie pentru modelele antrenate. Afișează cel mai bun set de parametri pentru modelul Random Forest.

---

## Instalare

### 1. Clonarea Repozitoriului

Clonează acest proiect pe mașina ta locală:

```bash
git clone https://github.com/Luciana11lm/music_genre_classifier.git
