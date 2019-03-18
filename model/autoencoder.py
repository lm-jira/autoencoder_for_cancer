from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import csv
import numpy as np
from constant import *


# random index lists of normal/tumor test data
normal_test_index = np.random.permutation(NUM_NORMAL)[:NUM_NORMAL//10]
tumor_test_index = np.random.permutation(NUM_TUMOR)[:NUM_TUMOR//10]

input = Input(shape=(INPUT_DIM, ))

x = Dense(1000, activation='relu')(input)
#x = Dense(2000, activation='relu')(x)
#x = Dense(1000, activation='relu')(x)

encoded = Dense(ENCODE_DIM, activation='relu')(x)

y = Dense(1000, activation='relu')(encoded)
#y = Dense(2000, activation='relu')(y)
#y = Dense(5000, activation='relu')(y)
decoded = Dense(INPUT_DIM, activation='sigmoid')(y)

autoencoder = Model(input, decoded)
encoder = Model(input, encoded)

#sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
#autoencoder.compile(optimizer=sgd, loss="binary_crossentropy")
#autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")

adagrad = optimizers.Adagrad(lr=LEARNING_RATE)
autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNC)

x_train = []
x_test = []

with open("/home/jira/hackdays/data/filtered_training_tumor_data_woNA.txt", "r") as file:
    data = csv.reader(file, delimiter="\t")

    for index, row in enumerate(data):
        if index in tumor_test_index:
            x_test.append([float(temp) for temp in row])
        else:
            x_train.append([float(temp) for temp in row])

with open("/home/jira/hackdays/data/filtered_training_normal_data_woNA.txt", "r") as file:
    data = csv.reader(file, delimiter="\t")

    for index, row in enumerate(data):
        if index in normal_test_index:
            x_test.append([float(temp) for temp in row])
        else:
            x_train.append([float(temp) for temp in row])

print("training samples = {}".format(len(x_train)))
print("test samples = {}".format(len(x_test)))

x_train = np.array(x_train)
x_train = np.random.shuffle(x_train)

x_test = np.array(x_test)
x_test = np.random.shuffle(x_test)

autoencoder.fit(x_train, x_train, epochs=50, shuffle=True, validation_data=(x_test,x_test), steps_per_epoch=21, validation_steps=2)

autoencoder.save('model_autoencoder.h5')
encoder.save('model_encoder.h5')


