import numpy as np
import DanConst
import ModelBasa
import copy

#   базовый class GenDan  - основа для генерации кода используемые в денс слоях и в сверточных
class GenDan(ModelBasa.ModelBasa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_block = 10
        self.ds=dict()
        self.nActivation = len( DanConst.typeActivation)
        self.nDenseMin = 1
        self.nDenseMax = 200
        self.nDenseStep = 10
        self.nDropoutMax = 0.5
        self.nBlock = 12
#        self.bots=dict()
        dblock = kwargs.get('d', {}) 

        if len(dblock)!=0:
            self.nDenseMin = dblock.get('nDenseMin', {self.nDenseMin})
            self.nDenseMax = dblock.get('nDenseMin', {self.nDenseMax})
            self.nDenseStep = dblock.get('nDenseStep', {self.nDenseStep})
            self.nDropoutMax = dblock.get('nDropoutMax', {self.nDropoutMax})
            self.nBlock = dblock.get('nBlock', {self.nBlock})

# класс для генерации кода с денс слоями, по которым в дальнейшем будем строить нейронную сеть 
# и в которой будем менять  коэффициенты
class GenDanDense(GenDan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __form_dan(self):
        blocks=dict()
# строка денс и строка активации
        blocks[0] = [np.random.randint(self.nDenseMin, self.nDenseMax) * self.nDenseStep, 
                     np.random.randint(0, self.nActivation)]
# строка Dropout
        if np.random.randint(0, 2) == 1: # для Dropout
            blocks[1] = round(np.random.uniform(0.05, self.nDropoutMax), 2)
# строка BatchNormalization
        if np.random.randint(0, 2) == 1: # для BatchNormalization
            blocks[2] =""
        return blocks

# генерируем модель random_- произвольное кол-во блоков,  
    def CreateModel(self, id, random_,  nblock = 0, blockrange=()):    
        self.id = id                                    # nblock max кол-во блоков
        nblock = self.count_block if nblock == 0 else nblock
# формируем кол-во бллоков с какого по какой пример (1,10,1) с 1..10 с шагом 1 или  (3,11,3)с 3 по 11 с шагом 3 
        if random_:  
            n = len(blockrange)
            fun1 = lambda x: 1  if x <= 0 else x
            if n ==1:   block_ = np.random.randint(1, fun1(blockrange[0])+1) 
            elif n==2:  block_ = np.random.randint(fun1(blockrange[0]), fun1(blockrange[1])+1) 
            else:       block_ = 1 
        else:           
                        block_ = nblock
# генерим словарь блоков
        self.config_bot.clear()

        for it_block in range(block_):
            self.config_bot[it_block] = GenDanDense.__form_dan(self)
        return self.config_bot

# заготовка для клонирования удачных моделей с последующим изменением
    def _ModificatAdd(self, *args):
        model_basa = args[0]
        add_layers = args[1]
        modificat = args[2]

# клонирование удачных моделей с последующим изменением их коэфиициентов (мутация)
    def _ModificatMulti(self, basa, k=0.3):
        k= round(np.random.uniform(0.05, k), 3)
        xbasa = copy.deepcopy(basa)
        for key, val in basa.items():
            xbasa[key][0][0]=(val[0])[0]*(1+k*(1 if np.random.randint(0, 2)==1 else -1))
        return xbasa

class GenerationModels():
    def __init__(self, *args, **kwargs):
        self.Models = dict()
        self.nModels = args[0]
        self._genDanDense = GenDanDense(0)

    def CreateModels(self, random_, nblock = (1, 15, 1)):
# формируем кол-во блоков       
        self.Models.clear()
        n = len(nblock)
        fun1 = lambda x: 1  if x <= 0 else x
        _start = fun1(nblock[0] if n > 1 else 1)
        _end   = fun1(nblock[1] if n >=2 else 1)+1
        _step  = fun1(nblock[2] if n ==3 else 1)
        nlayers = _start
        naprav = 1

# генерируем блоки по произвольно или по ленейно закону
        if random_:     # произвольному закону
            for id_model in range(self.nModels+1):
                self.Models[id_model] = copy.deepcopy(self._genDanDense.CreateModel(id_model, 
                                                                       True, _end, (_start, _end)))
        else:           # генерируем  по линейному закону
            for id_model in range(self.nModels+1):
                self.Models[id_model]=copy.deepcopy(self._genDanDense.CreateModel(id_model, random_,  nlayers))
                nlayers += naprav*_step

                if nlayers > _end:
                    nlayers = _end
                    naprav = -1

                if nlayers < _start:
                    nlayers = _start
                    naprav = 1

    def __copy_basa(self, basa):    # clear basa b find min max lay
        self.Models.clear()
        i=0
        ndense_min = 100
        ndense_max = 0
# вычисляем количество блоков с лучшими результатами. Для мутации 
# к примеру получились три лучших результата с 1 блоком с 4 блоками и с 6 блоками
# значит в следующей итерации мы должны создать сетки в которых будут от 1 блока до 6 
# в последующей итерации кол-во блоков будет стремиться к одному значению
# это видно в отчетах в файлах json
        for key, val in basa.items():
            n = len(val)
            ndense_min = min(n, ndense_min)
            ndense_max = max(n, ndense_max)
            self.Models[i] = copy.deepcopy(val)
            i+=1
        return ndense_min, ndense_max

# создаем копию модулей
    def CreateCopyModels(self, basa, k):
        nmin, nmax = GenerationModels.__copy_basa(self, basa)
        for it in range(0, 3):
            self.Models[len(self.Models)] = self._genDanDense._ModificatMulti(self.Models[it], k)
        
        for id_model in range(len(self.Models)-1, self.nModels):
            self.Models[id_model] = copy.deepcopy(self._genDanDense.CreateModel(id_model, True, nmax, (nmin, nmax)))
    
    def Get(self):
        return self.Models