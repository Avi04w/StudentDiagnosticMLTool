"""This file contains all code necessary for the Ensemble portion of the Final Project.

CSC311H1 - Fall 2024
"""

import numpy as np
from sklearn.impute import KNNImputer
import torch
from torch.autograd import Variable
from utils import (
    load_train_csv,
    load_train_sparse,
    load_valid_csv,
    load_public_test_csv,
    sparse_matrix_evaluate,
    evaluate
)
import item_response as ir
import neural_network as nn

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


def knn_predictions(train_data, test_data, k=11):
    """"Generate the predictions using KNN, where k=11. 
    11 was chosen based on the findings from knn.py.

    :param train_data: dictionary with training data
    :param test_data: dictionary with testing data
    :param sparse_shape: tuple indicating the shape of the full matrix
    :param k: k-value from knn
    :return: list of predictions for the test data 

    """

    # train_matrix = create_train_matrix(train_data, test_data)
    train_matrix = load_train_sparse("./data").toarray()

    # Using the KNNImputer similar to knn.py.
    nbrs = KNNImputer(n_neighbors=k)

    new_matrix = nbrs.fit_transform(train_matrix)

    predictions = []
    for user_id, question_id in zip(test_data["user_id"], test_data["question_id"]):
        # This check is necessary to avoid out of bound errors we were experiencing.
        if user_id < new_matrix.shape[0] and question_id < new_matrix.shape[1]:
            predictions.append(new_matrix[user_id, question_id])
        else:
            # Out of Bound Indices: Default to 0.5
            predictions.append(0.5)

    return predictions


def irt_predictions(train_data, test_data, lr=0.01, iterations=50):
    """
    Generate the predictions using IRT. "lr" and "iterations" is 
    set from values from item_response.py. 

    :param train_data: Dictionary in the format, {user_id: list, question_id: list, is_correct: list}
    :param test_data: Similar to train_data, but for testing (Note: used as input into IRT.)
    :param lr: Learning rate for IRT. (Based on item_response.py)
    :param iterations: Number of iterations for IRT. (Based on item_response.py)
    :return: List of predictions for the test data.
    """

    # Use the "irt" function in item_response.py to get the theta and beta values.
    theta, beta, _, _, _ = ir.irt(
        data=train_data, val_data=test_data, lr=lr, iterations=iterations
    )
    
    num_users = len(theta)
    num_questions = len(beta)

    predictions = []
    for i, question_id in enumerate(test_data["question_id"]):
        user_id = test_data["user_id"][i]
        if user_id >= num_users or question_id >= num_questions:
            # Again, out of Bound Indices: Default to 0.5
            predictions.append(0.5)
        else:
            probability = ir.sigmoid(theta[user_id] - beta[question_id])
            predictions.append(probability)
    
    return predictions


def nn_predictions(train_data, test_data, max_questions, num_epoch=10, k=100, lr=0.05, lamb=0.001):
    """" 
    Generate the predictions using a neural network model. 
    The hyperparameters are set based on the investigations and values from neural_network.py.

    :param train_data: Dictionary in the format: {user_id: list, question_id: list, is_correct: list}
    :param test_data: Similar to train_data
    :param max_questions: Total number of questions in the dataset.
    :param num_epoch: Number of epochs for the neural network model. (Based on neural_network.py)
    :param k: Number of hidden layer size. (Based on neural_network.py)
    :param lr: Learning rate. (Based on neural_network.py)
    :param lamb: Regularization parameter. (Based on neural_network.py)
    :return: List of predictions for the test data.
    """
    
    # Code mimics the load_data function in neural_network.py.
    train_matrix = load_train_sparse("./data").toarray()

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    # Create and train the model.
    model = nn.AutoEncoder(max_questions, k)
    nn.train(model=model, 
             lr=lr, 
             lamb=lamb, 
             train_data=train_matrix, 
             zero_train_data=zero_train_matrix, 
             valid_data=train_data, 
             num_epoch=num_epoch)
    
    predictions = []
    for i, question_id in enumerate(test_data["question_id"]):
        user_id = test_data["user_id"][i]
        inputs = Variable(train_matrix[user_id]).unsqueeze(0)
        output = model(inputs)
        probability = output[0][question_id].item()
        predictions.append(probability)

    predictions = np.array(predictions)
    print(f"NN Predictions: {predictions}")
    print(f"NN Predictions (NaN count): {np.isnan(predictions).sum()}")
    predictions = np.nan_to_num(predictions, nan=0.5)

    return predictions


def bagging_evaluation(bootstrap_predictions: dict, test_data: dict, weights: list, threshold=0.5) -> float:
    """Given the predictions derived from the bootstrapped datasets, evaluate them
    by averaging the predictions and calculating the accuracy from the given data.

    Note: Predictions are weighted (see main()). 

    This implementation relies on "evaluate" from utils.py.
    
    :param bootstrap_predictions: Predictions based off of bootstrapped datasets.
    :param test_data: Data used for the evalution.
    :param threshold: Threshold for the evaluation, defaulted to 0.5.
    :return: The average accuracy of the ensemble.
    """

    # Average the weighted predictions across all the bootstapped samples.
    ensemble_predictions = sum(
        weight * np.mean(bootstrap_predictions[model], axis=0)
        for model, weight in zip(bootstrap_predictions.keys(), weights)
    )
    
    # Use "evaluate" to calculate the accuracy.
    accuracy = evaluate(test_data, ensemble_predictions, threshold)

    return accuracy


def main():
    """Main function for the ensemble.py file. 
    
    NOTE: To obtain validation accuracy, please replace all instances of "test_data" 
    with "valid_data." Or vise versa if you want to evaluate the test data. Due to time 
    constraints, I wasn't able to implement a better solution to switch between accuracies
    (of course without copy and pasting the code twice).
    """

    # Load data.
    train_data = load_train_csv("./data")
    train_sparse = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Creating the bootstrapped datasets.
    dataset_size = len(train_data["user_id"])
    bootstrapped_datasets = create_bootstrap_dataset(train_data, dataset_size, m=3)

    # Generate the predictions for each bootstrapped dataset. Predictions are for each model.
    bootstrapped_predictions = {
        "knn": [],
        "irt": [],
        "nn": []
    }

    for dataset in bootstrapped_datasets:
        knn_pred = knn_predictions(dataset, test_data, k=11)
        bootstrapped_predictions["knn"].append(knn_pred)

        irt_pred = irt_predictions(dataset, test_data, lr=0.01, iterations=50)
        bootstrapped_predictions["irt"].append(irt_pred)

        nn_pred = nn_predictions(dataset, test_data, 
                              max_questions=train_sparse.shape[1], 
                              num_epoch=10, 
                              k=100, 
                              lr=0.05, 
                              lamb=0.001)
        bootstrapped_predictions["nn"].append(nn_pred)
    
    # Weights for each model determined by their performance on validation accuracy.
    weights = [0.3, 0.5, 0.2] # knn: 0.3, irt: 0.5, nn: 0.2

    # Evaluate the ensemble of bootstrapped predictions.
    ensemble_accuracy = bagging_evaluation(bootstrapped_predictions, test_data, weights, threshold=0.5)
    print(f"Ensemble Accuracy: {ensemble_accuracy}")


    # Please Note: These two functions below are solely used to determine if the ensemble accuracy compares
    # to the individual model accuracies and how their weights contribute to that accuracy. You are welcome to
    # uncomment them to see the results.

    # Check individual model accuracies
    # for model, predictions in bootstrapped_predictions.items():
    #     model_avg_predictions = np.mean(predictions, axis=0)
    #     model_accuracy = evaluate(test_data, model_avg_predictions, threshold=0.5)
    #     print(f"{model.upper()} Model Accuracy: {model_accuracy}")

    # Check weighted contributions
    # for model, weight in zip(bootstrapped_predictions.keys(), weights):
    #     model_avg_predictions = np.mean(bootstrapped_predictions[model], axis=0)
    #     weighted_contribution = np.mean(weight * model_avg_predictions)
    #     print(f"{model.upper()} Weighted Contribution to Ensemble: {weighted_contribution}")

if __name__ == "__main__":
    main()