import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = F.sigmoid(self.g(out))
        out = F.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            # Add the regularization term to the loss.
            reg_term = lamb / 2 * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.0) + reg_term

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accs.append(valid_acc)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
    return valid_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Q3 c)
    # k_list = [10, 50, 100, 200, 500]
    # lr_list = [0.001, 0.01, 0.05, 0.1]
    # epoch_list = [1, 3, 5, 10, 15, 20]
    # lamb = 0

    # max_acc = 0
    # best_hyper = []
    # for k in k_list:
    #     for lr in lr_list:
    #         for num_epoch in epoch_list:
    #             model = AutoEncoder(train_matrix.shape[1], k)
    #             train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #             valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
    #             if valid_accuracy > max_acc:
    #                 max_acc = valid_accuracy
    #                 best_hyper = [k, lr, num_epoch]

    # print("Best hyperparameters: ", best_hyper)
    # Best: k = 50, lr = 0.05, num_epoch = 15

    #####################################################################
    # Q3 d)
    # Set hyperparameters.
    k = 100
    model = AutoEncoder(train_matrix.shape[1], k)
    lr = 0.05
    num_epoch = 10
    lamb = 0

    # valid_accs = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # epochs = list(range(1, num_epoch + 1))

    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, valid_accs, marker='o', linestyle='-', label='Validation Accuracy')
    # plt.title('Validation Accuracy Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # model = AutoEncoder(train_matrix.shape[1], k)
    # train(model, lr, lamb, train_matrix, zero_train_matrix, test_data, num_epoch)
    # test_accuracy = evaluate(model, zero_train_matrix, test_data)
    # print("Final Test Accuracy: ", test_accuracy)

    # Q3 e)
    # max_acc = 0
    # best_lambda = 0
    # lambdas = [0, 0.001, 0.01, 0.1]
    # for lamb in lambdas:
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #     valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
    #     if valid_accuracy > max_acc:
    #         max_acc = valid_accuracy
    #         best_lambda = lamb
    # print("Best lambda: ", best_lambda)
    # # Best lambda = 0.001

    lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    valid_accuracy = evaluate(model, zero_train_matrix, test_data)

    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, test_data, num_epoch)
    test_accuracy = evaluate(model, zero_train_matrix, test_data)

    print("Final Validation Accuracy: ", valid_accuracy)
    print("Final Test Accuracy: ", test_accuracy)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
