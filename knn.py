import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat_transposed = nbrs.fit_transform(matrix.T) # transpose for question similarity
    mat = mat_transposed.T #transpose back to original format

    acc = sparse_matrix_evaluate(valid_data, mat)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k = [1, 6, 11, 16, 21, 26]

    # user based KNN
    user_accuracies = []
    print("User-based collaborative filtering:")
    for i in k:
        acc = knn_impute_by_user(sparse_matrix, val_data, i)
        user_accuracies.append(acc)
    plt.plot(k, user_accuracies)
    plt.xlabel("k-value")
    plt.ylabel("Validation Accuracy")
    plt.title("User-based collaborative filtering")
    plt.savefig("user-based_knn")
    plt.show()

    for i in range(len(k)):
        print(f"Impute By User: k-value: {k[i]}, Validation Accuracy: {user_accuracies[i]}")

    # select the best k for user-based filtering
    best_k_index = np.argmax(user_accuracies)
    best_k_user = k[best_k_index]
    # evaluate the test set using best k
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print(f"Test Accuracy (user-based, k={best_k_user}): {test_acc_user}")

    # item based KNN
    item_accuracies = []
    print("\nItem-based collaborative filtering: ")
    for i in k:
        acc = knn_impute_by_item(sparse_matrix, val_data, i)
        item_accuracies.append(acc)
    plt.plot(k, item_accuracies)
    plt.xlabel("k-value")
    plt.ylabel("Validation Accuracy")
    plt.title("Item-based collaborative filtering")
    plt.savefig("item-based_knn")
    plt.show()

    for i in range(len(k)):
        print(f"Impute By Item: k-value: {k[i]}, Validation Accuracy: {item_accuracies[i]}")
    
    # select best k for item based filtering
    best_k_item_index = np.argmax(item_accuracies)
    best_k_item = k[best_k_item_index]
    # evaluate on test set using best k
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print(f"Test Accuracy (item-based, k={best_k_item}): {test_acc_item}")

    # comparison of user based vs item based performance
    print("\nComparison:")
    print(f"User-based: Best k={best_k_user}, Test Accuracy={test_acc_user}")
    print(f"Item-based: Best k={best_k_item}, Test Accuracy={test_acc_item}")
    if test_acc_user > test_acc_item:
        print("User-based collaborative filtering performs better.")
    else:
        print("Item-based collaborative filtering performs better.")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
