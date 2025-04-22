from src.data_analysis import run_full_analysis
from src.data_preprocessing import preprocess_dataset
from src.model_train import train_best_model

if __name__ == "__main__":
  run_full_analysis()
  #df = preprocess_dataset('genres/features.csv')
  #train_best_model()
