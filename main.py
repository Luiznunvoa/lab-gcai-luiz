import numpy as np
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("Training data shape:", X_train.shape, "Labels shape:",
y_train.shape)
print("Test data shape:", X_test.shape, "Labels shape:", y_test.shape)
# Print the first 5 training labels
print("First 5 training labels:", y_train[:5].ravel())