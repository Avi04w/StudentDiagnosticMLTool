# TODO: complete this file.

import numpy as np
from sklearn.impute import KNNImputer
from utils import (
    load_train_sparse,
    load_valid_csv,
    load_public_test_csv,
    sparse_matrix_evaluate,
)

def create_bootstrap_dataset(primary_data: dict, dataset_size: int, m=3) -> list:
    """Using the primary data given for this assignment, generate m bootstrapped datasets.
    
    :param primary_data: Aligns with utils.py _load_csv dictionary representation of the primary data.
    :param dataset_size: The number of samples in each bootstrapped dataset.
    :param m: The number of bootstrapped datasets created.
    :return: A list of m bootstrapped datasets. (As a dictionary.)
    """
    
    dataset_idx = np.arange(len(primary_data["user_id"]))
    bootstrap_datasets = []
    
    for _ in range(m):
        # Sample w/ Repplacement
        bootstrap_idx = np.random.choice(dataset_idx, size=dataset_size, replace=True)

        # Create a new bootstrapped dataset.
        bootstrapped_data = {}
        for key in primary_data.keys():
            bootstrapped_data[key] = np.array([primary_data[key][i] for i in bootstrap_idx])

        # Append the dataset to the list.
        bootstrap_datasets.append(bootstrapped_data)

    return bootstrap_datasets


def knn_predictions(train_data, test_data, sparse_shape, k=11):
    """"Generate the predictions using KNN, where k=11. 
    11 was chosen based on the findings from knn.py.

    TODO: Add more details here. 

    """

    # Using the KNNImputer similar to knn.py.
    nbrs = KNNImputer(n_neighbors=k)

    train_matrix = np.empty(sparse_shape)
    train_matrix[:] = np.nan
    train_matrix[train_data["user_id"], train_data["question_id"]] = train_data["is_correct"]
    
    new_matrix = nbrs.fit_transform(train_matrix)

    predictions = []
    # for user_id, question_id in zip(test_data["user_id"], test_data["question_id"]):
    #     predictions.append(new_matrix[user_id, question_id])

    
    for user_id, question_id in zip(test_data["user_id"], test_data["question_id"]):
        if user_id < new_matrix.shape[0] and question_id < new_matrix.shape[1]:
            predictions.append(new_matrix[user_id, question_id])
        else:
            predictions.append(np.nan)  # Handle invalid indices

    return predictions


def evaluate_ensemble(bootsrapped_matrices: list, primary_data: dict) -> float:
    """Evaluate the ensemble of bootstrapped matrices.
    
    :param bootsrapped_matrices: A list of bootstrapped matrices.
    :param primary_data: The primary data used for the bootstrapped matrices.
    :return: The average accuracy of the ensemble of bootstrapped matrices.
    """
    # accuracies = []
    # for matrix in bootsrapped_matrices:
    #     acc = sparse_matrix_evaluate(primary_data, matrix)
    #     accuracies.append(acc)

    # return np.mean(accuracies)

    ensemble_matrix = np.mean(bootsrapped_matrices, axis=0)
    accuracy = sparse_matrix_evaluate(primary_data, ensemble_matrix)
    return accuracy


def main():
    """Main function for the ensemble.py file. """

    # Load data.
    train_sparse = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Creating the bootstrapped datasets.
    dataset_size = len(valid_data["user_id"])
    bootstrapped_datasets = create_bootstrap_dataset(valid_data, dataset_size, m=3)

    # Generate the predictions for each bootstrapped dataset.
    bootstrapped_predictions = []
    for dataset in bootstrapped_datasets:
        predictions = knn_predictions(dataset, test_data, train_sparse.shape, k=11)
        bootstrapped_predictions.append(predictions)

    # Evaluate the ensemble of bootstrapped predictions.
    ensemble_accuracy = evaluate_ensemble(bootstrapped_predictions, test_data)
    print(f"Ensemble Accuracy: {ensemble_accuracy}")

    test_ensamble = np.mean(bootstrapped_predictions, axis=0)
    test_accuracy = sparse_matrix_evaluate(test_data, test_ensamble)
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()