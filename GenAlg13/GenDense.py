# class генерации блоков dense по маске данных из внешпей вункции
#  создан от базового class ModelBasa.ModelBasa

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
import ModelBasa

class GenDense(ModelBasa.ModelBasa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#  генерация одного блока
    def __generat_block_layer(self, layers_, **kwargs): # d ={0:(100, 1)} 100- слой dense; 1 - тип активации
        dlayer=kwargs.get('d', {}) 

        if len(dlayer)==0:
            return layers_

        self.config_ai += [dlayer]
        printmodel_=[]
        for key, val in dlayer.items():
            if key == 0:
                layers_= Dense(val[0])(layers_)
                layers_= Activation(self.tActivation[val[1]])(layers_)  # сделать  тип активации
                printmodel_ += ["Dense(" +str(val[0])+",  Activation= "+self.tActivation[val[1]]+")"]
            if key == 1:
                layers_ = Dropout(val)(layers_ )
                printmodel_ += ["Dropout( "+str(val)+" )"]
            if key == 2:
                layers_ = BatchNormalization()(layers_ )
        return layers_, printmodel_

 #  генерация всей модели из блоков
    def GenModel(self, *args, **kwargs):
        if args.__len__()==0: return None

        x = args[0]
        dlayers = kwargs.get('ds', {}) 

        if len(dlayers)==0: return x
        self.PrintModel.clear()
        for key, val in dlayers.items():
            x, self.PrintModel[key] = GenDense.__generat_block_layer(self, x, d= val)
        return x



