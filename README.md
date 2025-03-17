


# 🧠 Advanced Deep Learning Optimization Guide

This repository provides a sophisticated implementation of diverse deep learning models and advanced optimization methodologies, aiming to enhance accuracy and reduce training time. The project examines innovative architectures and algorithms, facilitating cutting-edge experimentation and deployment across a wide array of applications.

---

## 🚀 Platform Operation

### 🔹 1. Individual Prediction
- 🏥 Allows users to input **biometric data** (pregnancy, glucose, blood pressure, etc.) to get a **prediction** on the likelihood of diabetes.
- 🧠 The deep learning model **processes the data** and returns a **diagnosis** along with medical recommendations.

### 🔹 2. Massive Analysis
- 📂 Users can upload a **CSV file** with multiple patient records to obtain **batch predictions**.
- 📊 The platform processes the data and generates a **detailed report** with key statistics, distributions, and correlations.

### 🔹 3. Virtual Assistant 🤖
- 💬 An **AI-based assistant** answers **diabetes-related questions**, provides **nutritional plans**, medication recommendations, and emotional support.
- 🔍 Uses the **`deepseek-r1:7b`** model to generate **accurate and contextualized answers**.

---

## 🛠️ Key Features

### 🔹 Advanced Deep Learning Architectures
- 🖼️ **CNNs (Convolutional Neural Networks)** – State-of-the-art image recognition models.
- ⏳ **RNNs (Recurrent Neural Networks)** – Designed for sequential data like text and time-series.
- 🔁 **LSTMs (Long Short-Term Memory Networks)** – Capture long-range dependencies in sequences.
- 🏗 **Autoencoders** – Used for **dimensionality reduction** and **data reconstruction**.
- 🎭 **GANs (Generative Adversarial Networks)** – Generate **high-quality synthetic datasets**.

### 🔹 Optimization Algorithms
- 🚀 Cutting-edge algorithms: **Adam**, **Stochastic Gradient Descent (SGD)**, **RMSProp**.
- 🛡️ Regularization techniques: **Dropout**, **Batch Normalization**, **L2 Regularization**.

### 🔹 Data Handling and Preprocessing
- 📦 Efficient dataset loading (e.g., **MNIST, CIFAR-10**).
- 🔧 Comprehensive preprocessing: **normalization, augmentation, batching**.

### 🔹 Evaluation and Visualization
- 📊 Metrics: **Accuracy, Precision, Recall, F1-score**.
- 📈 Training visualization: **Loss, Accuracy, and Validation Performance**.

---

## 🏗 Overview of the Neural Network Optimization Project

This guide provides a comprehensive roadmap for **implementing, executing, and testing** a **neural network optimization pipeline**.

### 🔹 Project Scope:
1. 📥 **Data Acquisition and Loading**  
   - Reads structured **CSV files** with feature vectors and corresponding labels.
   
2. 🔍 **Data Preprocessing**  
   - Splits data into **training/testing subsets**.
   - Converts data into **PyTorch-compatible tensors**.

3. 🏗 **Model Design**  
   - Multi-layer neural network with **two hidden layers** for predictive analytics.

4. 🎯 **Model Training**  
   - Uses optimization algorithms to **refine parameters and minimize errors**.

5. 📊 **Performance Evaluation**  
   - Evaluates model generalization using **unseen test data**.

6. 📈 **Output Results**  
   - Generates metrics like **accuracy & confusion matrices**.
   - Saves **trained model checkpoints** and loss evolution **visualizations**.

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ReapeRAlan/deep-learning-optimization.git
cd deep-learning-optimization
```

### 2️⃣ Install Ollama and `deepseek-r1:7b` Model
Ollama allows you to run language models locally. **DiabeDoc Pro** uses the `deepseek-r1:7b` model.

#### Ollama Installation:
1. Download **Ollama** from its [official site](https://ollama.ai/).
2. Install **Ollama** on your system.
3. Download the `deepseek-r1:7b` model:
   ```bash
   ollama pull deepseek-r1:7b
   ```

---

## 🛠️ Installation Tools

### 🔹 Create a Virtual Environment
#### 💻 Windows:
```bash
python -m venv DiabeApp
DiabeApp/scripts/activate
```
#### 🍏 Mac/Linux:
```bash
python3 -m venv DiabeApp
source DiabeApp/bin/activate
```
#### 🔹 Deactivate Virtual Environment:
```bash
deactivate
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Project Workflow

### 🔹 Train the Model:
Run the training script:
```bash
python mainALTERNATIVE.py
```

### 🔹 Run API:
Once the model is trained, run the API:
```bash
python DiabeDoc.py
```

### 🔹 Deploy the Application:
Launch the **Streamlit** application:
```bash
streamlit run DiabeApp.py
```

---

## 📋 Prerequisites

- 🐍 **Python Version**: `3.8` or higher

---

## 📂 Data File Format

Prepare a CSV file **`sample_data.csv`** in `./data/datasets/` with the following structure:

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

## ▶️ Running the Main Script

### **1️⃣ Initiate Training**
```bash
python main.py
```

### **2️⃣ Expected Outputs**
- **Console Output**:
  ```bash
  Using device: cpu
  Epoch 1/50 - Loss: 1.2345
  Epoch 2/50 - Loss: 0.9876
  ...
  Test set accuracy: 0.85
  Model saved at: ./models/saved_model.pth
  ```
- **Generated Files**:
  - 📈 `results/loss_accuracy_plot.png`: **Loss trends**
  - 💾 `models/saved_model.pth`: **Trained model checkpoint**

---

## 🛠 Testing the Trained Model

Create a script **`test_model.py`** to evaluate the trained model:
```python
import torch
from models.nn_model import initialize_nn
from utils.config import CONFIG

# Load the saved model
model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"])
model.load_state_dict(torch.load(CONFIG["save_model_path"]))
model.eval()

# Test on new data
new_data = torch.tensor([[5.9, 3.0, 5.1, 1.8]], dtype=torch.float32)
output = model(new_data)
_, predicted_class = torch.max(output, 1)
print(f"Prediction for input {new_data.numpy()}: Class {predicted_class.item()}")
```
Run the script:
```bash
python test_model.py
```

Sample output:
```bash
Prediction for input [[5.9 3.  5.1 1.8]]: Class 2
```

---

## 📁 Project Directory Structure
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

## 🎯 Conclusion

This project implements an advanced pipeline for **training, evaluating, and saving neural network models** for classification tasks.

---

## 🎭 Contributions

Contributions are highly encouraged! Submit an **issue** or **pull request**.

## 📝 Licensing

This project is distributed under the **Personal License**. See LICENSE.md for details.


---

