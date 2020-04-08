# Базовый class для создание всех нейронных сетей

import DanConst

class ModelBasa(object):
    def __init__(self, *args, **kwargs):
        self.id = 0 if len(args) == 0 else args[0]
        self.tActivation = DanConst.typeActivation
        self.config_ai=[]
        self.config_bot=dict()
        self.PrintModel=dict()

    def Model(self, *args, **kwargs):
        pass

    def get_id_config(self):
        return (self.id, self.config_ai)

    def get_config(self):
        return (self.config_ai)

    def get_id(self):
        return (self.id)

    def get_id_bot(self):
        return (self.id, self.config_bot)

    def get_bot(self):
        return (self.config_bot)

