import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data_dict = pickle.load(open('data.pickle', 'rb'))

max_len = max(len(entry) for entry in data_dict['data'])

#preprocessing data due to inhomogenous shape of handmark data points
data_padded = [entry + [0] * (max_len - len(entry)) for entry in data_dict['data']]

data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

classifiers = {
    'Naive Bayes': GaussianNB(),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    'Random Forests': RandomForestClassifier(random_state=42),
    'Support Vector Machines': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

#train and eval best model
results = {}
for name, model in classifiers.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    results[name] = accuracy
    print(f'{name} Accuracy: {accuracy:.4f}')

best_model = max(results, key=results.get)
print(f'\nBest Model: {best_model} with Accuracy: {results[best_model]:.4f}')