from Item_response_improved import irt_modified, evaluate, load_question_meta, load_subject_meta
import pandas as pd
from utils import load_train_csv, load_valid_csv, load_public_test_csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Load question metadata for the new IRT model
    question_meta = load_question_meta("./data")
    subject_meta = load_subject_meta("./data")
    num_subjects = max(subject_meta.keys()) + 1

    # Hyperparameters
    learning_rate = 0.05
    iterations = 100

    # Train the new IRT model with subject-aware difficulty
    print("Training IRT Model...")
    theta_new, beta_new, beta_subject, val_acc_new, train_log_lld_new, val_log_lld_new = irt_modified(
        train_data, val_data, lr=learning_rate, iterations=iterations,
        question_meta=question_meta, num_subjects=num_subjects
    )
    val_accuracy_new = val_acc_new[-1]
    test_accuracy_new = evaluate(test_data, theta_new, beta_new, beta_subject, question_meta)
    print(f"Test Accuracy = {test_accuracy_new:.4f}\n")

    subject_noise = np.random.normal(0, 0.5, num_subjects)
    beta_subject_noisy = beta_subject + subject_noise
    test_accuracy_new = evaluate(test_data, theta_new, beta_new, beta_subject_noisy, question_meta)
    print(f"Test Accuracy = {test_accuracy_new:.4f}\n")

if __name__ == "__main__":
    main()