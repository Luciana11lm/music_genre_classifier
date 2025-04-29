from src.data_analysis import run_full_analysis
from src.data_preprocessing import load_and_split_data
from src.model_train import train_best_model
from src.model_test import evaluate_model

if __name__ == "__main__":
  print("\nPASUL 1 - Analiza exploratorie -")
  run_full_analysis()
  print("\nPASUL 2 - Preprocesarea datelor si impartirea in seturi de antrenamet/ test -")
  X_train_scaled, X_test_scaled, y_train, y_test = load_and_split_data(output_csv='genres/features.csv')
  step = float(input("Se continua cu antrenarea modelului (0) sau se foloseste un model deja antrenat (1)?\n"))
  if step == 0:
    print("\nPASUL 3 - Antrenarea si salvarea modelului -")
    model, model_path = train_best_model(X_train_scaled, X_test_scaled, y_train, y_test)
    print("\nPASUL 4 - Testarea modelului si evaluarea performantelor -")
    evaluate_model(X_test_scaled, y_test, model_path=model_path)
  else:
    print("\nPASUL 4 - Testarea modelului si evaluarea performantelor -")
    evaluate_model(X_test_scaled, y_test, model_path='models/model_genre_classifier_random_forest.joblib')   # pentru testare fara antrenare, pe un model deja antrenat
