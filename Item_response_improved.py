from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

def load_question_meta(root_dir="./data"):
    # A helper function to load the question metadata file.
    path = os.path.join(root_dir, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subjects = eval(row[1])
                for s in subjects:
                    if s == 0:
                        subjects.remove(s)
                data[int(row[0])] = subjects
            except NameError:
                # Pass first row.
                pass
    return data


def load_subject_meta(root_dir="./data"):
    # A helper function to load the subject metadata.
    path = os.path.join(root_dir, "subject_meta.csv")

    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data[int(row[0])] = row[1]
            except ValueError:
                # Pass first row.
                pass
    return data

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, beta_subject, question_meta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param beta_subject: Vector
    :param question_meta: A dictionary
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for index, i in enumerate(data["user_id"]):
        j = data["question_id"][index]
        c = data["is_correct"][index]

        # Ccmpute the effective question difficulty with subject adjustments
        subjects = question_meta[j]
        pct_per_subject = 1 / len(subjects)
        beta_j = beta[j] + sum(beta_subject[s] * pct_per_subject for s in subjects)

        diff = theta[i] - beta_j
        log_lklihood += (c * diff - (np.log(1 + np.exp(diff))))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, beta_subject, question_meta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param beta_subject: Vector
    :param question_meta: A dictionary
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for index, i in enumerate(data["user_id"]):
        j = data["question_id"][index]
        c = data["is_correct"][index]

        # Compute the effective question difficulty with subject adjustments
        subjects = question_meta[j]
        pct_per_subject = 1 / len(subjects)
        beta_j = beta[j] + sum(beta_subject[s] * pct_per_subject for s in subjects)

        # Compute Gradient Terms
        p = sigmoid(theta[i] - beta_j)

        theta[i] += (c - p) * lr
        beta[j] += (p - c) * lr

        # Update subject difficulties
        for s in subjects:
            beta_subject[s] += lr * (p - c) * pct_per_subject
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, beta_subject


def irt(data, val_data, lr, iterations, question_meta, num_subjects):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param question_meta: A dictionary
    :param num_subjects: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1

    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    beta_subject = np.zeros(num_subjects)

    val_acc_lst = []
    train_log_lld = []
    val_log_lld = []

    for i in range(iterations):
        # Train Negative Log Likelihood
        train_neg_lld = neg_log_likelihood(data, theta, beta, beta_subject, question_meta)
        train_log_lld.append(train_neg_lld)

        # Validation Negative Log Likelihood
        val_neg_lld = neg_log_likelihood(val_data, theta, beta, beta_subject, question_meta)
        val_log_lld.append(val_neg_lld)

        # Validation Accuracy
        score = evaluate(val_data, theta, beta, beta_subject, question_meta)
        val_acc_lst.append(score)
        print("Iteration {}: NLLK Train: {} \t NLLK Val: {} \t Score: {}".format(
            i + 1, train_neg_lld, val_neg_lld, score))
        
        # Update Parameters
        theta, beta, beta_subject = update_theta_beta(data, lr, theta, beta, beta_subject, question_meta)

    return theta, beta, beta_subject, val_acc_lst, train_log_lld, val_log_lld


def evaluate(data, theta, beta, beta_subject, question_meta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param beta_subject: Vector
    :param question_meta: Dictionary
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subjects = question_meta[q]
        pct_per_subject = 1 / len(subjects)
        beta_j = beta[q] + sum(beta_subject[s] * pct_per_subject for s in subjects)
        x = (theta[u] - beta_j).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Load Metadata
    question_meta = load_question_meta("./data")
    subject_meta = load_subject_meta("./data")
    num_subjects = max(subject_meta.keys()) + 1

    # iterations = [25, 50, 100, 200]
    # learning_rates = [0.001, 0.01, 0.05, 0.1]
    # best_acc = 0
    # best_hyper = []
    # for iteration in iterations:
    #     for lr in learning_rates:
    #         theta, beta, beta_subject, val_acc_lst, train_log_lld, val_log_lld = irt(train_data, val_data, lr, iteration, question_meta, num_subjects)
    #         val_acc = val_acc_lst[-1]
    #         if val_acc > best_acc:
    #             best_acc = val_acc
    #             best_hyper = [iteration, lr]
    # print("Best hyperparameters: ", best_hyper)
    # Best hyperparameters:  [100, 0.05]

    learning_rate = 0.05
    num_iterations = 100
    theta, beta, beta_subject, val_acc_lst, train_log_lld, val_log_lld = irt(train_data, val_data, learning_rate, num_iterations, question_meta, num_subjects)
    
    print(f"Final Validation Accuracy: {val_acc_lst[-1]:.4f}")

    test_acc = evaluate(test_data, theta, beta, beta_subject, question_meta)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(train_log_lld) + 1), train_log_lld, label="Training NLL")
    plt.plot(range(1, len(val_log_lld) + 1), val_log_lld, label="Validation NLL")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.legend()
    plt.title("Training and Validation NLL")
    plt.show()
    
    #####################################################################

    j1, j2, j3 = 2, 60, 150
    
    print(f"Question {j1} Difficulty: {beta[j1 - 1]:.4f}")
    print(f"Question {j2} Difficulty: {beta[j2 - 1]:.4f}")
    print(f"Question {j3} Difficulty: {beta[j3 - 1]:.4f}")
    
    prob_j1 = sigmoid(theta - beta[j1])
    prob_j2 = sigmoid(theta - beta[j2])
    prob_j3 = sigmoid(theta - beta[j3])

    # Sort theta for creating graphs
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]
    prob_j1 = prob_j1[sorted_indices]
    prob_j2 = prob_j2[sorted_indices]
    prob_j3 = prob_j3[sorted_indices]

    # Plot the probability curves
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_theta, prob_j1, label=f"Question {j1}")
    plt.plot(sorted_theta, prob_j2, label=f"Question {j2}")
    plt.plot(sorted_theta, prob_j3, label=f"Question {j3}")
    plt.xlabel("Theta (User Ability)")
    plt.ylabel("P(Correct Response)")
    plt.title("Probability of Correct Response as a Function of Theta")
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
