import numpy as np
import pandas as pd
from numpy import genfromtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_and_write():
    df = pd.read_csv("data.csv",delimiter=",",usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
    print(df.head())
#read_and_write()

# Load the dataset
features = genfromtxt("data.csv", delimiter=',', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31), dtype=float, skip_header=1)
class_value = genfromtxt("data.csv", delimiter=',', usecols=(1), dtype=str, skip_header=1)

print(class_value)

# Encode labels and standardize features
y = LabelEncoder().fit_transform(class_value)
X = StandardScaler().fit_transform(features)

# Reshape labels
y = np.reshape(y, (-1, 1))

# Build the model
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=15, batch_size=15)

# Evaluate the model
accuracy = model.evaluate(X, y)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))
