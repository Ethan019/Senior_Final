import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
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
#### START OF NB MODEL DO NOT TOUCH ####
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# Make predictions on the testing data
nb_predictions = nb.predict(X_test_vec)



# Make predictions on the test data



end_time=time.time()
elapsed_time=end_time-start_time_ELM
# Get the predicted label for each sample
### END OF NB MODEL

# Evaluate the model using various metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
nb_cm = confusion_matrix(y_test, nb_predictions)
nb_classification_report = classification_report(y_test, nb_predictions)

# Print the evaluation metrics
print("Naive Bayes Model Results: ")
print("*********************")
print(f"Time running: {elapsed_time} seconds")
print("*********************")
num_spam = (data['Label'] == 1).sum()
num_not_spam = (data['Label'] == 0).sum()
print("Number of spam emails: ", num_spam)
print("Number of non-spam emails: ", num_not_spam)
print("*********************")
print("Metrics")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(nb_cm)
print("Classification Report:")
print(nb_classification_report)
sys.exit(0)

