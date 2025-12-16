# Customer Churn Prediction Using ANN

## Description
This project focuses on predicting customer churn using an Artificial Neural Network (ANN). Customer churn refers to when a customer stops using a company’s service. By predicting churn in advance, businesses can take steps to retain customers and reduce losses.

This project was developed to understand data preprocessing, neural networks, and classification problems.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
- Streamlit
- Tensorboard
---

## Features
- Data preprocessing and cleaning
- Feature encoding and scaling
- Artificial Neural Network model training
- Customer churn prediction
- Model evaluation using accuracy

---

## Dataset
The dataset contains customer demographic and banking information such as age, gender, geography, balance, and account activity. It is used to analyze customer behavior and identify patterns related to churn.

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/iskushpatel/Customer_Churn_prediction_using_ANN.git
2. Navigate to the project directory:
   ```bash
   cd Customer_Churn_prediction_using_ANN
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
4. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook
5. Run the notebook cells to train and test the model.
---

## Model Overview
This project uses an Artificial Neural Network (ANN) for binary classification to predict customer churn. After preprocessing, the input layer receives numerical features obtained through encoding and feature scaling.

The ANN consists of one input layer, two hidden layers with ReLU activation, and one output layer with sigmoid activation to predict the probability of churn. The model is trained using the Adam optimizer and binary cross-entropy loss function, making it suitable for churn prediction tasks.

-----------
## Model Deployment
This model is deployed on Streamlit cloud:

[https://customerchurnpredictionusingann-kwiivbnmhuhzwcucbtopro.streamlit.app/](https://customerchurnpredictionusingann-kwiivbnmhuhzwcucbtopro.streamlit.app/)
