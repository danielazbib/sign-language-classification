import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow import pad_sequences

data_dict = pickle.load(open('data.pickle', 'rb'))

print(type(data_dict['data']))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# padded_data = pad_sequences(data_dict['data'], dtype='float32', padding='post')

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)
model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))