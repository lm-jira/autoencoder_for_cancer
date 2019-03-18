from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
import keras_metrics as km
from constant import *

import csv
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)
train_x, train_y = [], []
test_x, test_y = [], []

tumor_data = []
normal_data = []

with open("/home/jira/hackdays/data/filtered_training_normal_data_woNA.txt", "r") as f:
    data = csv.reader(f, delimiter="\t")
    normal_data = [row for row in data]

with open("/home/jira/hackdays/data/filtered_test_tumor_data_woNA.txt", "r") as f:
    data = csv.reader(f, delimiter="\t")
    tumor_data = [row for row in data]
            
for row in normal_data:
    data = [float(temp) for temp in row]
    test_x.append(data)
    test_y.append([1,0])
    
for row in tumor_data:
    data = [float(temp) for temp in row]
    test_x.append(data)
    test_y.append([0,1])
    
test_index = np.arange(len(test_y))
test_x = np.array(test_x)
test_y = np.array(test_y)
#np.random.shuffle(test_index)

print("number of normal = {}".format(len(normal_data)))
print("number of tumor = {}".format(len(tumor_data)))
print("number of test = {}".format(len(test_y)))

encoder = load_model('model_encoder.h5')
encoded_test_x = encoder.predict(test_x[test_index])

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(ENCODE_DIM, )))
#model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights('new_model_classifier_nodropout.h5')

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[km.precision(), km.recall(), km.f1_score()])

log_filepath = "/home/jira/hackdays/test_log"
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1)
cbks = [tb_cb]

#model.fit(encoded_train_x, train_y[train_index], validation_split=0.1, epochs=50, callbacks=cbks)
score = model.evaluate(encoded_test_x, test_y)

print("eval result")
print("[binary_crossentropy, precision, recall, f1_score] = {}".format(score))
#print(model.predict(encoded_test_x))
#print(test_y)
