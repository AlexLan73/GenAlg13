# Олег, Доброе время суток 
#  привожу фрагмент лога Log0.txt и log1.txt в котором видно, что на тестовых данных val_accuracy = 0.98 
#  конфигурация сети находится в файлах с расширением json 
#  id-номер робота
#
# -> id = 0  Epoch: 9   accuracy: 0.9868542  val_acc: 0.98083335
#48000/48000 - 12s - loss: 0.0399 - accuracy: 0.9869 - val_loss: 0.0720 - val_accuracy: 0.9808
# -> id = 4  Epoch: 9   accuracy: 0.98864585  val_acc: 0.97975
#48000/48000 - 9s - loss: 0.0341 - accuracy: 0.9886 - val_loss: 0.0772 - val_accuracy: 0.9797
# -> id = 17  Epoch: 8   accuracy: 0.9959583  val_acc: 0.9759167
#48000/48000 - 10s - loss: 0.0160 - accuracy: 0.9960 - val_loss: 0.0978 - val_accuracy: 0.9759
#Epoch 10/10
# -> id = 17  Epoch: 9   accuracy: 0.99714583  val_acc: 0.97566664
#48000/48000 - 5s - loss: 0.0124 - accuracy: 0.9971 - val_loss: 0.0954 - val_accuracy: 0.9757
# id = 0  val_accuracy = 0.9808333516120911
# id = 1  val_accuracy = 0.9798333048820496
# id = 4  val_accuracy = 0.9797499775886536

import CFitGen
import GenDanDense
import WriteRead

#   в основу системы "лежит" блок 
# Блок состоит из:
#  Dense Activation - обязательно 
#  Dropout, BatchNormalization- рандомно
#
# 0 этап -> 20 ботов с генерацией блоков от 1 до 15 и используем все виды активации на 10 эпох
#   лучший вариант записан в T0_b20_l15_e10  => 
#   T0 номер теста;  b20 - кол-во бот; l15 кол-во блоков;  e10-кол-во эпох
#
def stage_x(CountBot, Countlayer, _max_mutation,_repeat, name_test):
    _writeread = WriteRead.WriteReadDan()                       # класс

    _generationModels = GenDanDense.GenerationModels(CountBot)
    _generationModels.CreateModels(True, (1, Countlayer, 1))
    _models =_generationModels.Models

    for i in range(_repeat):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!  "+str(i)+ "   !!!!!!!!!!!!!!!!!!!!!!!!!")
        _fits = CFitGen.FITs()
        _fits.SetModel(_models)
        _fits.run()
        history, models_max, modelsprn = _fits.ReadHistory()

        namefile = name_test + str(i)
        _writeread.Set(namefile)

        _writeread.write_file_json(modelsprn)
        _writeread.write_file_pic(history, models_max)

        _generationModels.CreateCopyModels( models_max, _max_mutation)
        _models = _generationModels.Get()

if __name__ == "__main__":


    # запускаем 0 тест для поиско кол-во слоев
#    stage_x(20, 15, 0.5, 1, "T0_b20_l15_e10_")

    # запускаем 1 тест, весь перечень активации
#    stage_x(20, 8, 0.5, 5, "T1_b20_l8_e10_") 

    # запускаем 2 тест, весь перечень активации
    stage_x(20, 3, 0.5, 5, "T2_b20_l3_e10_") 
       


