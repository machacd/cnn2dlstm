import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv1D, GlobalAveragePooling1D, Flatten, MaxPooling1D, AveragePooling3D, BatchNormalization
from keras.layers import Activation, Input, LSTM, TimeDistributed, MaxPooling2D, MaxPooling3D, ConvLSTM2D, Reshape, GlobalAveragePooling3D, GlobalMaxPooling3D


def create_data(l):
    # some training data
    np.random.seed(l)
    x = np.arange(l)/200 + np.random.uniform(10)
    y1 = np.sin(1.8*x)*np.random.uniform(0.5, 1.1, len(x))
    y2 = np.cos(10.3*x-2)
    y = y1 + y2
    return(x, y1, y2, y)


def create_target(y, shift, threshold):
    up = (y[shift:] > (1+threshold)*y[:-shift])*1  # possibly fix
    down = ((y[shift:] < (1-threshold)*y[:-shift]))*(-1)
    return(1+up+down)


def create_features(y1, y2, target, shift):
    return(np.concatenate([y2[:-shift], target],
                          axis=0).reshape((3, len(y1)-shift))).transpose()
    # target variable (up or down movement)


def reshape_features(feat, lsplit, csplit):
    all = []
    for i in range(lsplit):
        all.append(feat[i:len(feat)-lsplit+1+i])
    return(np.concatenate(all)).reshape((-1,
                                         lsplit,
                                         feat.shape[1]),
                                        order='F').reshape((-1,
                                                            int(lsplit /
                                                                csplit),
                                                            csplit, feat.shape[1], 1))


shift = 10

[x, y1, y2, y] = create_data(1000)
target = create_target(y, shift, 0.1)

f = plt.figure()
plt.plot(target)
plt.plot(y)
# plt.show()


lsplit = 4
csplit = 2

# combine all into a single array
a = create_features(y1, y2, target, shift)
feat = reshape_features(a, lsplit, csplit)
design_x = feat[:, :, :, :-1, :]
design_y = keras.utils.to_categorical(feat[:, -1, -1, -1, 0], num_classes=3)


# create shifted arrays


# reshape into the correct form for cnn2d (second reshape) with lstm (first reshape)


model = Sequential()
model.add(TimeDistributed(
    Conv2D(filters=int(4), kernel_size=(
        csplit, feat.shape[1]-1)),
    input_shape=(None, csplit, feat.shape[-2]-1, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(8,  return_sequences=False,
               input_shape=(lsplit, feat.shape[-2]-1)))
model.add(Dense(8, activation='linear'))
model.add(Dropout(0.9))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(design_x, design_y,
                    epochs=200,
                    batch_size=64,
                    validation_split=0.1,
                    shuffle=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()


# vdate
[x_v, y1_v, y2_v, y_v] = create_data(500)
target_v = create_target(y_v, shift, 0.1)

feat_v = reshape_features(create_features(
    y1_v, y2_v, target_v, shift), lsplit, csplit)
val_x = feat_v[:, :, :, :-1, :]

# create shifted arrays
pp = model.predict_proba(val_x)

plt.plot(pp[:, -1])
plt.plot(y_v)
# plt.show()
f.savefig('a.pdf')
