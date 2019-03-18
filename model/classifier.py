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

with open("/home/jira/hackdays/data/filtered_training_data_woNA.txt", "r") as f:
    data = csv.reader(f, delimiter="\t")    
    for row in data:
        if row[-1] in ["Metastatic", "Primary Tumor", "Recurrent Tumor"]:
            tumor_data.append(row[:-1])
        else:
            normal_data.append(row[:-1])
            
index=0
num_train=0
num_test=0

for row in normal_data:
    data = [float(temp) for temp in row]

    if index%10==0:
        test_x.append(data)
        test_y.append([1,0])
        num_test+=1
    else:
        train_x.append(data)
        train_y.append([1,0])
        num_train+=1
    index+=1
    
t_index = 0
for row in tumor_data:
    data = [float(temp) for temp in row]

    if t_index%10==0:
        test_x.append(data)
        test_y.append([0,1])
        num_test+=1
    else:
        train_x.append(data)
        train_y.append([0,1])
        num_train+=1
    t_index+=1
    if t_index > index*4:
        break

train_index = np.arange(num_train)
test_index = np.arange(num_test)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

np.random.shuffle(train_index)
np.random.shuffle(test_index)

encoder = load_model('model_encoder.h5')
print(train_x.shape)
encoded_train_x = encoder.predict(train_x[train_index])
encoded_test_x = encoder.predict(test_x[test_index])

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(ENCODE_DIM, )))
#model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[km.precision(), km.recall(), km.f1_score()])

early_stopping_monitor = EarlyStopping(patience=15)

log_filepath = "/home/jira/hackdays/train_log"
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1)
cbks = []#[tb_cb]

model.fit(encoded_train_x, train_y[train_index], validation_split=0.1, epochs=50, callbacks=cbks)
model.save('new_model_classifier_nodropout.h5')

test_result = model.evaluate(encoded_test_x, test_y[test_index])
print("eval result")
print(test_result)

#print(model.predict(encoded_test_x))
#print(test_y)
