# глобальные переменные
# https://docs.python.org/3/library/enum.html
from enum import Enum
class TypeNet(Enum):
    Dense = 0
    Conv2 = 1

class TypeModificator(Enum):
    Add = 0
    Multi = 1
#isinstance(Color.GREEN, Color)
#True

# типы активаторов которые есть в tf и в керасе
#  для первого прогона использую все
typeActivationBasa = ["exponential", "hard_sigmoid", "linear", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh"]

#  для уточненного прогона
typeActivation = ["hard_sigmoid", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh"]

