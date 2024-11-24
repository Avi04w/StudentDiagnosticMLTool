import numpy as np
from sklearn.utils import resample
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
from knn import knn_impute_by_user

# currently all models are using knn for bagging
def knn_bagging_ensemble(matrix, valid_data, test_data, k, num_models=3):
  """
  Implements a bagging ensemble with KNN as the base model.

    :param matrix: 2D sparse matrix (training data).
    :param valid_data: Validation data as a dictionary.
    :param test_data: Test data as a dictionary.
    :param k: Number of neighbors for KNN.
    :param num_models: Number of base models in the ensemble.
    :return: Validation and test accuracies for the ensemble.
  """
  predictions_valid = []
  predictions_test = []

  for model_idx in range(num_models):
    # bootstrapping - sample rows with replacement from original training matrix
    bootstrapped_matrix = resample(matrix, replace=True, n_samples=matrix.shape[0])

    # train the knn model on the bootstrapped training set
    val_pred = knn_impute_by_user(bootstrapped_matrix, valid_data, k)
    test_pred = knn_impute_by_user(bootstrapped_matrix, test_data, k)

    # store predictions
    predictions_valid.append(val_pred)
    predictions_test.append(test_pred)

  # ensemble predictions - average predictions from all models
  avg_valid = np.mean(predictions_valid, axis=0)
  avg_test = np.mean(predictions_test, axis=0)

  # Convert predictions to binary predictions (0 or 1) based on the threshold
  avg_valid_binary = (avg_valid >= 0.5).astype(int)
  avg_test_binary = (avg_test >= 0.5).astype(int)

  # Convert the 1D averaged predictions back to a 2D matrix for evaluation
  valid_matrix = np.zeros_like(matrix)
  test_matrix = np.zeros_like(matrix)
  
  # Assign the averaged predictions back to the corresponding user-question pairs
  for i in range(len(valid_data["user_id"])):
      user_id = valid_data["user_id"][i]
      question_id = valid_data["question_id"][i]
      valid_matrix[user_id, question_id] = avg_valid_binary[i]
  
  for i in range(len(test_data["user_id"])):
      user_id = test_data["user_id"][i]
      question_id = test_data["question_id"][i]
      test_matrix[user_id, question_id] = avg_test_binary[i]

  # evaluate ensemble performance
  val_accuracy = sparse_matrix_evaluate(valid_data, valid_matrix)
  test_accuracy = sparse_matrix_evaluate(test_data, test_matrix)

  print(f"Ensemble Validation Accuracy: {val_accuracy}")
  print(f"Ensemble Test Accuracy: {test_accuracy}")

  return val_accuracy, test_accuracy


def main():
  sparse_matrix = load_train_sparse("./data").toarray()
  val_data = load_valid_csv("./data")
  test_data = load_public_test_csv("./data")

  k = 11  # Use the best k found previously
  n_models = 3

  print("Running bagging ensemble...")
  val_accuracy, test_accuracy = knn_bagging_ensemble(sparse_matrix, val_data, test_data, k, n_models)

  print(f"Final Validation Accuracy: {val_accuracy}")
  print(f"Final Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()