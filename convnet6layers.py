import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D, BatchNormalization
import pickle
from sklearn import preprocessing
from keras import backend as K
import os
import cv2
from keras.callbacks import TensorBoard
from tempfile import TemporaryFile
import time

class convnet_64:
    def __init__(self, X, Y,cates,epoc = 3,bat_sz = 32 ):
        self.X = X
        self.Y = Y
        self.epoc = epoc
        self.bat_sz = bat_sz
        self.cates = cates

    def convnet(self):
        init_op=tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        chanDim = -1

        model = Sequential()

        model.add(Conv2D(32, (3,3), input_shape = self.X.shape[1:], padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.15))

        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.15))

        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.15))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(self.cates))
        model.add(Activation('softmax'))


        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        model.fit(self.X,self.Y,batch_size=self.bat_sz, validation_split=0.1, epochs=self.epoc)
        self.model = model

    def valid(self, test_X, test_Y, numtolet):
        self.test_X = test_X
        self.test_Y = test_Y
        self.numtolet = numtolet

        convpreds = []
        ntest_Y = []
        nconvpreds = []
        corr = 0
        alls = 0

        preds = model.predict_classes(self.test_X)
        print(preds)
        for pred in preds:
            tmp = int(pred)
            convpreds.append(numtolet.item().get(tmp))
        count = (len(test_Y) - 1)
        for t in range(len(self.test_Y)):
            ntest_Y.append("'|' ACT: {}".format(self.test_Y[t]))
            nconvpreds.append("PRED: {} '|'".format(convpreds[t]))
            if self.test_Y[t] == convpreds[t]:
                corr += 1
            alls += 1

        predtoact = dict(zip(ntest_Y,nconvpreds))
        acc = (corr/alls)
        print("Accuracy: {}".format(acc))
        print(predtoact)
    def save_model(self,name):
        self.model.save("convnet32-64-128-{}.model".format(name))
    def load_model(self,name):
        self.model = load("convnet32-64-128-{}.model".format(name))
        
