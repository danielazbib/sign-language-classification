import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


data_dict = pickle.load(open('data.pickle', 'rb'))

max_len = max(len(entry) for entry in data_dict['data'])

#preprocessing data due to inhomogenous shape of hand data points
data_padded = [entry + [0] * (max_len - len(entry)) for entry in data_dict['data']]

data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

classifiers = {
    'Naive Bayes': GaussianNB(),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    'Random Forests': RandomForestClassifier(),
    'Support Vector Machines': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results = {}
model_names = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


for name, model in classifiers.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    results[name] = accuracy


    report = classification_report(y_test, y_predict, output_dict=True)
    
    model_names.append(name)
    accuracy_list.append(report['accuracy'])
    precision_list.append(report['weighted avg']['precision'])
    recall_list.append(report['weighted avg']['recall'])
    f1_list.append(report['weighted avg']['f1-score'])

    print(f"Classification Report - {name}:\n{classification_report(y_test, y_predict)}\n")

best_model = max(results, key=results.get)
print(f'\nBest Model: {best_model} with Accuracy: {results[best_model]:.4f}')

best = classifiers[best_model]

#pickles best trained model to be used in main.py
f = open('model.p', 'wb')
pickle.dump({'model': best}, f)
f.close()

# model evauluation metric line graph
plt.figure(figsize=(23, 6))
plt.plot(model_names, accuracy_list, marker='o', label='Accuracy')
plt.plot(model_names, precision_list, marker='o', label='Weighted Avg Precision')
plt.plot(model_names, recall_list, marker='o', label='Weighted Avg Recall')
plt.plot(model_names, f1_list, marker='o', label='Weighted Avg F1 Score')

plt.title('Model Evaluation Metrics')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.legend()
plt.show()

#model evauluation metric bargraph
barWidth = 0.2

plt.figure(figsize=(20, 3))

r1 = np.arange(len(model_names))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, accuracy_list, width=barWidth, label='Accuracy')
plt.bar(r2, precision_list, width=barWidth, label='Weighted Avg Precision')
plt.bar(r3, recall_list, width=barWidth, label='Weighted Avg Recall')
plt.bar(r4, f1_list, width=barWidth, label='Weighted Avg F1 Score')

plt.xlabel('Models', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(model_names))], model_names)

plt.title('Model Evaluation Metrics')
plt.legend(loc='upper right')
plt.show()

#confusion matrix heatmap graph
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()