#  class FITone  - фомируем и запускаем модель на обучение
#  class FITs - в потоке запускаем заданное кол-во моделей на обучение
#
#
import numpy as np
import GenDense

from collections import OrderedDict
from operator import itemgetter
import collections

import threading
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense    #, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

def read_dan_mnist():
    ## В Keras встроены средства работы с популярными наборами данных
    (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()

    ## Преобразование размерности данных в наборе + Нормализация данных
    x_train = x_train_org.reshape(60000, 784).astype('float32') / 255
    x_test = x_test_org.reshape(10000, 784).astype('float32') / 255
    #Преобразуем метки в формат one hot encoding
    y_train = utils.to_categorical(y_train_org, 10)
    y_test = utils.to_categorical(y_test_org, 10)

    return x_train, y_train, x_test, y_test, y_train_org, y_test_org

#   callbacks -отражения информации о потоке обучения
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    
    def __init__(self, id, procent=1, wait=1000):
        super().__init__()
        self.id = id

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        print(' -> id = '+ str(self.id) +'  Epoch: '+str(epoch)+'   accuracy: '+ str(accuracy)+"  val_acc: "+str(val_acc))

#   обучаем модель
class FITone(threading.Thread):

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self)

        self.xtrain  = args[0]['xtrain']
        self.ytrain  = args[0]['ytrain'] 
        self.xtest  = args[0]['xtest'] 
        self.ytest =  args[0]['ytest'] 

        self.model_ = args[1]
        self._dense = GenDense.GenDense(0)
        self._inputs_basa = Input(shape=(784,), name="inputs_basa" )

        self.id = args[2] # id модели

#   создаем модель по маске
    def SetModel(self):
        x = self._dense.GenModel(self._inputs_basa, ds=self.model_)
        out_0 = Dense(self.ytest.shape[1], activation='softmax')(x)
        self.model =  Model(inputs=self._inputs_basa, outputs=out_0)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(self.model.summary())

#   запуск на обучение
    def run(self):
        self.batchSize = 150
        self.epohs = 10
        self.verbose = 2
        callbacks = [CustomEarlyStopping(id = self.id)]
        self.history = self.model.fit(self.xtrain, 
                                      self.ytrain, 
                                      batch_size = self.batchSize, 
                                      epochs = self.epohs, 
                                      verbose = self.verbose, 
                                      validation_split = 0.2, 
                                      callbacks=callbacks
                                    )

#   запускаем N-моделей
class FITs():
    def __init__(self, *args, **kwargs):
        xtrain, ytrain, xtest, ytest, ytrain_org, ytest_org = read_dan_mnist()
        self.Dan = {'xtrain': xtrain, 'ytrain': ytrain, 'xtest':xtest, 'ytest':ytest}
        self.DFit=dict()

#   загружаем маску и записваем модели
#   планировалоси создавать два разный потока моделей
    def SetModel(self, models:dict):
        self.models = models
        self.DFit.clear()

        for key, val in self.models.items():
            self.DFit[key]=FITone(self.Dan, val, key)
            self.DFit[key].SetModel()

#  запуск на обучение всех моделей
    def run(self):
        for key, val in self.DFit.items():
            self.DFit[key].start()

        for key, val in self.DFit.items():
            self.DFit[key].join()

#   считываем модели с лучшим результатом
    def ReadHistory(self):
        if len(self.DFit)==0:  return
        d = dict()

#   формируем критерий
        for key, val in self.DFit.items():
            d[key]=self.DFit[key].history.history['val_accuracy'][-1]

#   сортируем по значению
        ii=0        
        d1=OrderedDict(sorted(d.items(), key = itemgetter(1), reverse = True))
        masHis, models_, modelsprn, = dict(),  dict(),  dict()

        for  key, val  in d1.items():
            print(" id = {key}  loss = {val}".format(key=key, val=val))
            masHis[key] = self.DFit[key].history.history        #   история обучения
            models_[key] = self.models[key]                     #   конфигурация модели
            #   формируем сеть для печати и добавляем лучшее значение
            z = OrderedDict(sorted(self.DFit[key]._dense.PrintModel.items()))
            z['val_accuracy'] = str(round(self.DFit[key].history.history['val_accuracy'][-1],6))
            modelsprn[key] =  z
            ii+=1
            if ii >= 3:
                break
        return masHis, models_, modelsprn

