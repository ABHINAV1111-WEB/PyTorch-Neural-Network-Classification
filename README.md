# PyTorch-Neural-Network-Classification
A simple yet powerful demonstration of building and training a neural network in PyTorch to classify points in a synthetic dataset (two concentric circles). This project highlights the importance of non-linear activation functions (like ReLU) in learning complex decision boundaries.

# Project Overview:
Implemented a custom PyTorch model (CircleModelV2) with multiple layers and ReLU activations.

Compared performance of linear models vs non-linear models on circular data.

Visualized decision boundaries to show how activations enable curved separations.

Trained and evaluated the model using BCEWithLogitsLoss and accuracy metrics.

# Features
Data Generation: Synthetic dataset using sklearn.datasets.make_circles.

Model Architecture: Multi-layer perceptron with ReLU activations.

Training Loop: Forward pass, loss calculation, backpropagation, optimizer step.

Evaluation Mode: Demonstrates the use of model.eval() and torch.inference_mode().

Visualization: Decision boundary plots for train/test sets.

# Respository Structure:
├── helper_functions.py   # Utility functions for plotting predictions & decision boundaries
├── model.py              # CircleModelV2 definition
├── train.py              # Training loop
├── README.md             # Project documentation
└── notebooks/            # Jupyter notebooks with experiments

# Results
Without ReLU → decision boundary is a straight line, poor accuracy(approx 51%).

With ReLU → decision boundary curves around the circles, much higher accuracy(approx 99%).

# Key Learnings
Why ReLU introduces non-linearity and enables complex decision boundaries.

Difference between train() and eval() modes in PyTorch.

Role of Dropout and BatchNorm statistics in training stability and generalization.

How to interpret Softmax vs Sigmoid for binary vs multiclass classification.


# Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.
