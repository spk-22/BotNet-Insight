---
# IoT Botnet Detection with GraphSAGE

This repository contains code for detecting IoT botnet attacks (specifically Mirai) using a **GraphSAGE** model. The approach involves representing IoT device data as a graph and leveraging the power of Graph Neural Networks (GNNs) for classification.

## Overview

The project follows these key steps:

1.  **Data Loading and Preprocessing**: Benign and Mirai attack traffic data (in CSV format) are loaded, concatenated, and preprocessed. This includes handling missing values, scaling numerical features, and separating timestamps and labels.
2.  **Graph Construction**: The preprocessed time-series data is transformed into a graph structure where each row (representing a snapshot of device activity) becomes a node. Edges are created between consecutive nodes to capture temporal relationships.
3.  **GraphSAGE Model Implementation**: A two-layer GraphSAGE model is implemented using `torch_geometric` for node classification. The model learns embeddings for each node by aggregating information from its neighbors.
4.  **Training and Evaluation**: The GraphSAGE model is trained on the constructed graph using a supervised learning approach. Training involves optimizing cross-entropy loss. The model's performance is evaluated using metrics like accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.
5.  **Early Stopping**: To prevent overfitting and ensure optimal model performance, early stopping is implemented based on validation accuracy.

## Dataset

The model utilizes two types of IoT network traffic data:

* **Benign**: Normal IoT device network behavior.
* **Mirai**: Network traffic generated during a Mirai botnet attack.

These datasets are expected to be in CSV format within `benign/` and `mirai/` directories, respectively. The data is preprocessed to handle `Timestamp` columns, replace infinite values, drop columns with `NaN`s, and scale numerical features using `StandardScaler`.

## Setup

To run the code, you'll need to install the following libraries:

```bash
!pip install torch torch-geometric scikit-learn pandas networkx
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html -q
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html -q
```

**Note**: The `torch-scatter` and `torch-sparse` installations are specific to `torch==2.6.0+cu124`. Adjust the URL if you are using a different PyTorch version.

## Code Structure

The main script performs the following actions:

* **Imports**: Necessary libraries like `torch`, `torch_geometric`, `sklearn`, and `pandas` are imported.
* **Google Drive Mounting**: If running in Google Colab, it mounts Google Drive to access datasets.
* **Data Loading and Preprocessing**:
    * Loads CSV files from specified `benign_dir` and `mirai_dir`.
    * Combines datasets and adds a `label` column (0 for benign, 1 for Mirai).
    * Handles `inf` values, drops columns with `NaN`s, and scales numerical features.
* **Graph Creation**:
    * Converts scaled features and labels into PyTorch tensors.
    * Constructs a simple sequential graph where each node is connected to its immediate successor and predecessor, forming `edge_index`.
    * Creates a `torch_geometric.data.Data` object and saves it as `iot_data_graph.pth`.
* **Model Definition (`GraphSAGE`)**:
    * Defines a two-layer GraphSAGE model with `SAGEConv` layers, `BatchNorm1d`, `ReLU` activations, and dropout.
    * Includes a final linear layer for classification.
* **Training and Evaluation Functions**:
    * `train(model, data, optimizer, criterion)`: Performs one training step.
    * `evaluate(model, data, verbose=False)`: Evaluates the model on the test set, printing accuracy and a detailed classification report.
    * `evaluate_model(model, data)`: Provides comprehensive evaluation including accuracy, precision, recall, F1-score, confusion matrix, and ROC curve plotting.
    * `plot_roc_curve(model, data)`: Specifically plots the ROC curve for binary classification.
* **Training Loop**:
    * Initializes the `GraphSAGE` model, `Adam` optimizer, and `CrossEntropyLoss` (with class weights for imbalance).
    * Splits data into training, validation, and test sets using `train_test_split`.
    * Trains the model for a specified number of epochs with early stopping based on validation accuracy.
    * Saves the best performing model.
* **Results Visualization**:
    * Plots training loss and test accuracy over epochs.
    * Displays the confusion matrix and ROC curve after final evaluation.

## Usage

1.  **Prepare your data**: Ensure your benign and Mirai attack CSV files are organized in `/content/drive/My Drive/IOT/benign/` and `/content/drive/My Drive/IOT/mirai/` respectively, or update the `benign_dir` and `mirai_dir` variables in the script.
2.  **Run the notebook/script**: Execute the code cells sequentially. The script will:
    * Load and preprocess the data.
    * Construct the graph.
    * Train the GraphSAGE model.
    * Evaluate its performance with various metrics and visualizations.
    * Save the best model to `/content/drive/My Drive/IOT/iot_model.pth`.

## Results

The script will output:

* Shape and label counts of the combined dataset.
* Details about feature columns used.
* Dimensions of feature (`x`) and label (`y`) tensors.
* Total number of nodes and edges in the constructed graph.
* Training loss and test accuracy for each epoch.
* A detailed classification report on the test set (accuracy, precision, recall, F1-score).
* Plots of training loss and test accuracy over epochs.
* A confusion matrix visualizing classification performance.
* An ROC curve with AUC score.

## Future Improvements

* Explore more sophisticated graph construction methods that capture richer relationships (e.g., based on IP addresses, port numbers, or specific flow features).
* Implement more advanced Graph Neural Network architectures (e.g., GAT, GCN with attention mechanisms).
* Investigate the impact of different hyperparameter settings on model performance.
* Consider causal sampling or sliding window approaches for creating graph snapshots to better capture temporal dynamics in larger datasets.
* Evaluate the model on a wider range of IoT botnet attacks and different IoT device types.
