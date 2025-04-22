from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

def evaluate_model(X_test, y_test, model_path="best_random_forest_model.pkl"):
  model = joblib.load(model_path)
  print("Modelul a fost incarcat")
  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='macro')
  recall = recall_score(y_test, y_pred, average='macro')
  f1 = f1_score(y_test, y_pred, average='macro')

  print(f"\n Evaluarea modelului:")
  print(f" - Acuratete: {accuracy:.4f}")
  print(f" - Precizie (macro): {precision:.4f}")
  print(f" - Recall (macro): {recall:.4f}")
  print(f" - F1-score (macro): {f1:.4f}")
  print("\n Classification Report:\n")
  print(classification_report(y_test, y_pred))

  # confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
  plt.xlabel('Etichete prezise')
  plt.ylabel('Etichete reale')
  plt.title('Confusion Matrix')
  plt.tight_layout()
  plt.savefig('genres/confusion_matrix.png')
  plt.show()