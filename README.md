## Platform Operation

### 1. Individual Prediction
- Allows users to input biometric data (pregnancy, glucose, blood pressure, etc.) to get a prediction on the likelihood of diabetes.
- The deep learning model processes the data and returns a diagnosis along with medical recommendations.

### 2. Massive Analysis
- Users can upload a CSV file with multiple patient records to obtain batch predictions.
- The platform processes the data and generates a detailed report with key statistics, distributions and correlations.

### 3. Virtual Assistant
- An AI-based assistant answers diabetes-related questions, provides nutritional plans, medication recommendations and emotional support.
- It uses the `deepseek-r1:7b` model to generate accurate and contextualized answers.

---

# Advanced Deep Learning Optimization Guide

This repository provides a sophisticated implementation of diverse deep learning models and advanced optimization methodologies, aiming to enhance accuracy and reduce training time. The project examines innovative architectures and algorithms, facilitating cutting-edge experimentation and deployment across a wide array of applications.

## Key Features

- **Advanced Deep Learning Architectures**:
  - **Convolutional Neural Networks (CNNs)**: State-of-the-art image recognition and classification models.
  - **Recurrent Neural Networks (RNNs)**: Tailored for processing sequential datasets such as text and time-series.
  - **Long Short-Term Memory Networks (LSTMs)**: Designed for capturing long-range dependencies in sequence data.
  - **Autoencoders**: Facilitating dimensionality reduction and unsupervised data reconstruction.
  - **Generative Adversarial Networks (GANs)**: Enabling the generation of high-quality synthetic datasets, including images and other complex data structures.

- **Optimization Algorithms**:
  - Cutting-edge algorithms: **Adam**, **Stochastic Gradient Descent (SGD)**, **RMSProp**.
  - Regularization methods to mitigate overfitting: **Dropout**, **Batch Normalization**, and **L2 Regularization**.

- **Data Handling and Preprocessing**:
  - Efficient dataset loading (e.g., MNIST, CIFAR-10).
  - Comprehensive preprocessing capabilities: normalization, augmentation, and batching.

- **Evaluation Metrics and Visualization Tools**:
  - Sophisticated metrics, including accuracy, precision, recall, and F1-score.
  - High-resolution visualizations of training dynamics (loss, accuracy, and validation performance).

## Overview of the Neural Network Optimization Project

This guide provides a comprehensive roadmap for implementing, executing, and testing a neural network optimization pipeline. It elaborates on the project’s purpose, preparation steps, and deliverables.

---

## **Project Scope**

The project utilizes **Artificial Neural Networks (ANNs)** to address supervised classification challenges. Key steps include:

1. **Data Acquisition and Loading**:
   - Input data is read from a structured CSV file containing feature vectors and corresponding labels.

2. **Data Preprocessing**:
   - Partitioning datasets into training and testing subsets.
   - Conversion into tensor formats compatible with PyTorch frameworks.

3. **Model Design**:
   - Constructs a multi-layer neural network comprising two hidden layers for predictive analytics.

4. **Model Training**:
   - Employs optimization algorithms to refine model parameters and minimize error metrics.

5. **Performance Evaluation**:
   - Evaluates generalization using unseen test data.

6. **Output Results**:
   - Quantifies performance metrics such as accuracy and confusion matrices.
   - Saves trained model checkpoints and produces insightful loss evolution visualizations.

---

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ReapeRAlan/deep-learning-optimization.git
   cd deep-learning-optimization
   ```

### 1. Ollama and the `deepseek-r1:7b` Model
- Ollama** is a tool that allows you to run language models locally. DiabeDoc Pro uses the `deepseek-r1:7b` model for the virtual assistant.
- To use Ollama and the `deepseek-r1:7b` model, follow these steps:

#### Ollama installation
1. Download Ollama from its [official site](https://ollama.ai/).
2. Install Ollama on your operating system.
3. Download the `deepseek-r1:7b` model:
   ````bash
   ollama pull deepseek-r1:7b

## Installation Tools

### Create a Virtual Environment
- **Windows**:
  ````bash
  python -m venv DiabeApp
  DiabeApp “scripts” activate
  ```
- **Mac**: 
   ````bash
   python3 -m venv DiabeApp
   source DiabeApp/bin/activate
   ```

- Deactivate virtual environment:
   ````bash
   deactivate
   ```

## Install Dependencies
- Install the necessary tools from requirements.txt
   ````bash
   pip install -r requirements.txt
   ```

## Deactivate virtual environment:
   ````bash
   deactivate
   ```
## Project Workflow

### **1. Configuration**

Adjust hyperparameters in the `config.py` file under the `utils/` directory to suit your requirements (e.g., learning rate, batch size, number of epochs).

### **2. Execution**

Run the main script to initiate training and evaluation:

```bash
python main.py
```

### **3. Results Visualization**

Leverage the utilities in `plot_utils.py` to generate plots for loss, accuracy, and validation performance.

---

## Prerequisites

- **Python Version**: 3.8 or higher
- Required libraries:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

---

## Data File Format

Prepare a CSV file named `sample_data.csv` in the `./data/datasets/` directory with the following structure:
- Columns for numerical features.
- A final column for numerical labels (e.g., class indices).

**Example**:

```csv
feature1,feature2,feature3,feature4,label
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
```

---

## Running the Main Script

### **1. Initiate Training**

Execute the main script:

```bash
python main.py
```

### **2. Expected Outputs**

- **Console Output**:
  Displays loss progression and accuracy statistics:

```bash
Using device: cpu
Epoch 1/50 - Loss: 1.2345
Epoch 2/50 - Loss: 0.9876
...
Test set accuracy: 0.85
Model saved at: ./models/saved_model.pth
```

- **Generated Files**:
  - `results/loss_accuracy_plot.png`: Graph showing loss trends.
  - `models/saved_model.pth`: Trained model checkpoint.

---

## Testing the Trained Model

Create a script `test_model.py` to evaluate the trained model:

```python
import torch
from models.nn_model import initialize_nn
from utils.config import CONFIG

# Load the saved model
model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"])
model.load_state_dict(torch.load(CONFIG["save_model_path"]))
model.eval()

# Test on new data
new_data = torch.tensor([[5.9, 3.0, 5.1, 1.8]], dtype=torch.float32)  # Replace with actual data
output = model(new_data)
_, predicted_class = torch.max(output, 1)
print(f"Prediction for input {new_data.numpy()}: Class {predicted_class.item()}")
```

Run the script:

```bash
python test_model.py
```

**Sample Output**:

```bash
Prediction for input [[5.9 3.  5.1 1.8]]: Class 2
```

---

## Project Directory Structure

The repository is organized as follows:

```
.
|-- data/
|   |-- datasets/
|       |-- sample_data.csv  # Dataset file
|
|-- models/
|   |-- nn_model.py          # Neural network architecture
|   |-- saved_model.pth      # Trained model checkpoint
|
|-- results/
|   |-- loss_accuracy_plot.png  # Training progress visualization
|
|-- utils/
|   |-- config.py            # Global configuration file
|   |-- metrics.py           # Metric calculation utilities
|   |-- plot_utils.py        # Plotting utilities
|
|-- main.py                  # Main script
```

---

## Conclusion

This project implements an advanced pipeline for training, evaluating, and saving neural network models for classification tasks. It is adaptable to diverse datasets by simply modifying the CSV input file and hyperparameters in `config.py`.

Should you have inquiries or require further assistance, feel free to reach out. Happy experimenting! 😊

---

## Contributions

Contributions are highly encouraged! Please submit an issue or pull request for improvements.

## Licensing

This project is distributed under the MIT License. Refer to the LICENSE file for detailed terms.

## Acknowledgments

This initiative draws inspiration from significant advancements in deep learning optimization and emphasizes practical implementations of contemporary methodologies.

