# AG News Classification Using LSTM

## 📖 Project Overview
This project implements a deep learning-based text classification system that categorizes news articles into four distinct categories: **World**, **Sports**, **Business**, and **Science/Technology**. Using Long Short-Term Memory (LSTM) networks and Natural Language Processing (NLP) techniques, the system automatically classifies news headlines and descriptions from the AG News dataset.

## 🎯 Objectives
- Apply deep learning concepts to Natural Language Processing (NLP)
- Develop an LSTM-based model for multi-class text classification
- Preprocess and normalize raw text data for optimal performance
- Evaluate model accuracy and analyze performance metrics
- Create a user-friendly GUI for real-time predictions
- Implement model persistence for deployment

## 🛠️ System Requirements

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Programming Language**: Python 3.8+
- **IDE/Platform**: Jupyter Notebook or Google Colab
- **Key Libraries**:
  - TensorFlow/Keras
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
  - Tkinter/Streamlit (for GUI)

### Hardware Requirements
- **Processor**: Intel i5 / AMD Ryzen 5 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 5 GB free space
- **GPU**: Optional (NVIDIA GPU with CUDA support recommended for faster training)

## 📊 Dataset
The project uses the **AG News Dataset** containing news headlines and short descriptions categorized into:
- **0**: World
- **1**: Sports
- **2**: Business
- **3**: Science/Technology

## 🔧 Methodology

### Data Pipeline
1. **Data Loading & Inspection**: Load and explore dataset distribution
2. **Text Normalization**: Convert to lowercase, remove punctuation/stopwords, clean text
3. **Tokenization**: Convert text to integer sequences using Keras Tokenizer
4. **Sequence Padding**: Standardize sequence length for LSTM input
5. **Train-Test Split**: 80-20 split for training and evaluation
6. **Model Building**: LSTM architecture with embedding, dropout, and dense layers
7. **Model Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
8. **Artifact Saving**: Save model and tokenizer for deployment

### Model Architecture
- **Embedding Layer**: 128-dimensional word embeddings
- **LSTM Layer**: 128 units with sequential dependency learning
- **Dropout Layer**: Prevents overfitting
- **Dense Layer**: Fully connected with ReLU activation
- **Output Layer**: Softmax activation for 4-class classification

**Total Parameters**: 1,412,100 (5.39 MB)

## 📈 Results

### Performance Metrics
- **Test Accuracy**: 59.47%
- **Test Loss**: 1.2281

### Training Progress
The model was trained for 10 epochs, showing consistent learning with final validation accuracy of approximately 70%.

### GUI Interface
A simple graphical interface allows users to:
- Input news headlines or descriptions
- Get instant category predictions
- View confidence scores and probability distributions

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit
