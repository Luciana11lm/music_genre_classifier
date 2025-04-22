import pandas as pd
import numpy as np
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# -----------------------------------------------------------------
# Antrenarea mai multor modele KNN pentru a-l alege pe cel mai bun
# -----------------------------------------------------------------
def train_knn(X_train, X_test, y_train, y_test, k_values=[3, 5, 7, 9]):
  best_k = None             # initializare valori definitorii pentru a stabili cel mai bun model
  best_accuracy = 0
  best_model = None

  for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)  # instanta de model KNN
    model.fit(X_train, y_train)                  # antrenare pe datele de antrenament
    y_pred = model.predict(X_test)               # predictie pe datele de test pentru a afla acuratetea
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acuratetea pentru KNN cu k={k}: {accuracy:.4f}")

    if accuracy > best_accuracy:  # se salveaza modelul cu acuratetea cea mai mare
      best_accuracy = accuracy
      best_k = k
      best_model = model

  print(f"Cel mai bun model KNN are k={best_k} cu acuratetea: {best_accuracy:.4f}")
  dump(model, 'models/model_genre_classifier_knn.joblib') # salvare model
  return best_model

# -----------------------------------
# Antrenarea modelului Random Forest
# -----------------------------------
def train_random_forest(X_train, X_test, y_train, y_test):
  param_grid = {
    'n_estimators': [500, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
  }
  model = RandomForestClassifier(random_state=42) # instanta de model Random Forest
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  print(f"Cel mai bun set de parametri: {grid_search.best_params_}")
  print(f"Acuratețea modelului optimizat: {grid_search.best_score_}")
  best_model = grid_search.best_estimator_

  y_pred = best_model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Acuratețea pe setul de test: {accuracy:.4f}")
  #model.fit(X_train, y_train)     # antrenare model
  #y_pred = model.predict(X_test)  # predictii folosite pentru calcularea acuratetii

  #accuracy = accuracy_score(y_test, y_pred)
  #print(f"Acuratetea pentru Random Forest: {accuracy:.4f}")
  dump(best_model, 'models/model_genre_classifier_random_forest.joblib')  # salvare model
  return best_model

# ----------------------------------------------------
# Antrenarea ambelor modele si alegerea celui mai bun
# ----------------------------------------------------
def train_best_model(X_train, X_test, y_train, y_test ):
  knn_model = train_knn(X_train, X_test, y_train, y_test)             # antrenare model knn
  rf_model = train_random_forest(X_train, X_test, y_train, y_test)    # antrenare model random forest

  model_choice = input("Introduceti modelul ales (1 - KNN, 2 - Random Forest): ")
  if model_choice == 1: 
    return knn_model
  else:
    return rf_model  
