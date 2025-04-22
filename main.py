from src.data_analysis import run_full_analysis
from src.data_preprocessing import preprocess_dataset
from src.model_train import train_best_model

if __name__ == "__main__":
  print("PASUL 1 - Analiza exploratorie -")
  run_full_analysis()
  print("PASUL 2 - Preprocesarea datelor si impartirea in seturi de antrenamet/ test -")
  X_train_scaled, X_test_scaled, y_train, y_test = load_and_split_data(output_csv='genres/features.csv')
  print("PASUL 3 - Antrenarea si salvarea modelului -")
  #train_best_model()
  print("PASUL 4 - Testarea modelului si evaluarea performantelor -")
