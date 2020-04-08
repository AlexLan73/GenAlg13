#  класс для записи, чтения данных на диск
import json
import pickle

class WriteReadDan(object):
    def __init__(self, name_ = "testdan"):
        self.filename = name_
#  установка имени файла
    def Set(self, name_ = "testdan"):
        self.filename = name_

#  запись данных: история обучения по эпохом и конфигурация сетки в коде
    def write_file_pic(self, history, models, name_ = ""):
        name_ = name_ if name_ != "" else self.filename

        data={"history":history, "models":models}
        with open(name_ +'.pickle', 'wb') as f:
            pickle.dump(data, f)

#  чтение данных: история обучения по эпохом и конфигурация сетки в коде
    def read_file_pic(self, name_  = ""):
        name_ = name_ if name_ != "" else self.filename
        with open(name_ +'.pickle', 'rb') as f:
            data = pickle.load(f)
        history, models = data["history"], data["models"]
        return history, models

#  запись данных: конфигурация сетки в текстовом виде расширение json
    def write_file_json(self, d, name_  = ""):
        name_ = name_ if name_ != "" else self.filename
        with open(name_+".json", "w") as write_file:
            json.dump(d, write_file)
    

