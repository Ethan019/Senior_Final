import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


# Load the data
data = pd.read_csv('spam.csv')

#spam.csv code
data['Label']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
# Split the data into training and testing sets


#Uncomment below for datasets over than spam.csv
#X_train, X_test, y_train, y_test = train_test_split(data.Body, data.Label, test_size=0.2, random_state=30)

#spam.csv code
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Label, test_size=0.2, random_state=44)

# Vectorize the email body using the CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_train_vec = MinMaxScaler().fit_transform(X_train_vec)
X_train_vec = np.reshape(X_train_vec, (X_train_vec.shape[0], -1))
X_test_vec = vectorizer.transform(X_test).toarray()
X_test_vec = MinMaxScaler().fit_transform(X_test_vec)
X_test_vec = np.reshape(X_test_vec, (X_test_vec.shape[0], -1))





start_time_ELM = time.time()
#### START OF ELM MODEL DO NOT TOUCH ####
# Set up the neural network


input_size = X_train_vec.shape[1]
hidden_size = 1000
input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])



# Define the activation function
def sigmoid(x):
    return 1/(1+ np.exp(-x))

# Define the function to compute the hidden nodes
def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = sigmoid(G)
    return H



output_weights = np.dot(np.linalg.pinv(hidden_nodes(X_train_vec)), y_train)


# Define the prediction function
def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out



#### END OF ELM MODEL



# Make predictions on the test data
prediction = predict(X_test_vec)


end_time=time.time()
elapsed_time=end_time-start_time_ELM
# Get the predicted label for each sample


train_labels = np.unique(y_train)
test_labels = np.unique(y_test)

print("Unique labels in training set:", train_labels)
print("Unique labels in test set:", test_labels)

unique_labels = np.unique(y_test)
valid_labels = [label for label in unique_labels if np.sum(y_test == label) > 0]

accuracy = accuracy_score(y_test, np.round(prediction).astype(int))
precision = precision_score(y_test, np.round(prediction).astype(int), labels=valid_labels, average=None, zero_division=1)
recall = recall_score(y_test, np.round(prediction).astype(int), labels=valid_labels, average=None, zero_division=1)
f1 = f1_score(y_test, np.round(prediction).astype(int), labels=valid_labels, average=None, zero_division=1)

print("ELM Model Results: ")
print("*********************")
print(f"Time running: {elapsed_time} seconds")
print("Number of nodes used:", hidden_size)
print("*********************")
num_spam = (data['Label'] == 1).sum()
num_not_spam = (data['Label'] == 0).sum()
print("Number of spam emails: ", num_spam)
print("Number of non-spam emails: ", num_not_spam)
print("*********************")
print("Metrics")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print accuracy and confusion matrix
print("Accuracy:", accuracy*100)

cm = confusion_matrix(y_test, np.round(prediction).astype(int), labels=valid_labels)
sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.figure(figsize=(2, 2))

plt.show()


print(classification_report(y_test, np.round(prediction).astype(int),labels=valid_labels, zero_division=1))

sys.exit(0)

