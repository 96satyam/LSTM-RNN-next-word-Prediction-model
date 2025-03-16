# LSTM-RNN-next-word-Prediction-model

###  A deep learning project that predicts the next word in a sequence using an LSTM-based Recurrent Neural Network (RNN).

#### 📌 Project Overview
This project builds an LSTM-based RNN model to predict the next word in a given sentence. It is trained on Shakespeare’s Hamlet dataset from the NLTK Gutenberg corpus and can be extended for various NLP applications like autocomplete, AI writing assistants, and chatbots.

#### 🛠️ Tech Stack
Programming Language: Python 🐍
Libraries: TensorFlow, Keras, NLTK, NumPy, Pandas, Sklearn
Deep Learning Model: LSTM (Long Short-Term Memory) RNN
Deployment: Colab + Streamlit
#### 📂 Dataset
The dataset used is Shakespeare's Hamlet from NLTK's Gutenberg Corpus. The text is tokenized, converted into sequences, and padded before training the LSTM model.

#### ⚙️ Model Architecture
The model consists of:
✅ Embedding Layer for word representation
✅ Two LSTM Layers for learning context dependencies
✅ Dropout Layer for regularization
✅ Dense Layer (Softmax Activation) for word prediction

#### 🏋️ Training & Performance
The model is trained with categorical cross-entropy loss and the Adam optimizer.
It predicts the next word in a sentence with improving accuracy over epochs.
### 🎯 Example Prediction
Input: "To be or not to be"
Predicted Next Word: "buried" (Can be fine-tuned for better results)
