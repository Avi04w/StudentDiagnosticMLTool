from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
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
        diff = theta[i] - beta[j]
        log_lklihood += (c * diff - (np.log(1 + np.exp(diff))))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for index, i in enumerate(data["user_id"]):
        j = data["question_id"][index]
        c = data["is_correct"][index]
        
        p = sigmoid(theta[i] - beta[j])

        theta[i] += (c - p) * lr
        beta[j] += (p - c) * lr
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_log_lld = []
    val_log_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_lld.append(neg_lld)

        # Validation Negative Log Likelihood
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_lld.append(neg_lld_val)

        # Validation Accuracy
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("Iteration {}: NLLK Train: {} \t NLLK Val: {} \t Score: {}".format(
            i + 1, neg_lld, neg_lld_val, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_lld, val_log_lld


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.01
    num_iterations = 50
    theta, beta, val_acc_lst, train_log_lld, val_log_lld = irt(train_data, val_data, lr=learning_rate, iterations=num_iterations)
    
    print(f"Final Validation Accuracy: {val_acc_lst[-1]:.4f}")

    test_acc = evaluate(test_data, theta, beta)
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
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
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
