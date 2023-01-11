import os
from flask import Flask
from flask import request
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from category_encoders import TargetEncoder, OneHotEncoder
import warnings
import pandas as pd

app = Flask(__name__)
port = int(os.environ.get('PORT', 3000))

[()]
EFT = "Engine_Fuel_Type"
EHP = "Engine_HP"
EC = "Engine_Cylinders"
NOD = "Number_of_Doors"
MC = "Market_Category"
MSRP = "MSRP"
YEAR = "Year"
MODEL = "Model"
MAKE = "Make"
TT = "Transmission_Type"
DW = "Driven_Wheels"
VSI = "Vehicle_Size"
VST = "Vehicle_Style"
DROP = [EFT, TT, DW, VSI, VST]
COLTARGETENC = [YEAR, MODEL, MAKE]


class DataReader:
    def __init__(self):
        self.__path = "data/CarPricesOriginal.csv"

    def read(self):
        return pd.read_csv(self.__path)


class Model:

    def __init__(self, instantCreate=False):
        self.__TargetEncode = {}
        self.__OneHotEncoder = OneHotEncoder()
        self.__MinMaxScaler = MinMaxScaler()
        self.__TrainingsData = self.__readTrainingData()
        self.__TrainedModel = False
        if instantCreate:
            self.create()

    def __readTrainingData(self):
        rawData = DataReader().read()
        rawData[EFT].fillna(f"data[{EFT}].mode()", inplace=True)
        rawData[EHP].fillna(
            rawData[EHP].median(), inplace=True)
        rawData[EC].fillna(
            rawData[EC].median(), inplace=True)
        rawData[NOD].fillna(
            rawData[NOD].median(), inplace=True)
        rawData.drop([MC], inplace=True, axis=1)
        return rawData

    def __addTargetEncoder(self, colname, encoder):
        self.__TargetEncode[colname] = encoder

    def __getTargetEncoder(self, colname):
        return self.__TargetEncode[colname]

    def __getOHEncoder(self):
        return self.__OneHotEncoder

    def __getTrainingsData(self):
        return self.__TrainingsData

    def __getMinMaxScaler(self):
        return self.__MinMaxScaler

    def __addTrainedModel(self, model):
        self.__TrainedModel = model

    def __getTrainedModel(self):
        return self.__TrainedModel

    def create(self):
        shuffled_data = shuffle(self.__getTrainingsData(), random_state=100)
        X_train = shuffled_data.drop([MSRP], axis=1)
        y_train = shuffled_data[MSRP]

        for col in COLTARGETENC:
            encoder = TargetEncoder(cols=col)
            encoder.fit(X_train[col], y_train.to_frame()[MSRP])
            X_train[col] = encoder.transform(X_train[col])
            self.__addTargetEncoder(col, encoder)

        self.__getOHEncoder().fit(X_train[DROP])
        one_hot_encoded_output_train = self.__getOHEncoder().transform(
            X_train[DROP])

        X_train = pd.concat([X_train, one_hot_encoded_output_train], axis=1)

        X_train.drop(DROP, axis=1,
                     inplace=True)

        self.__getMinMaxScaler().fit(X_train)
        X_train_new = self.__getMinMaxScaler().transform(X_train)

        model = DecisionTreeRegressor()
        model.fit(X_train_new, y_train)
        self.__addTrainedModel(model)

    def predict(self, request):
        testData = pd.DataFrame.from_dict(request)
        testData.drop([MC], inplace=True, axis=1)
        for col in COLTARGETENC:
            testData[col] = self.__getTargetEncoder(
                col).transform(testData[col])
        one_hot_encoded_output_test = self.__getOHEncoder().transform(
            testData[DROP])
        X_test = pd.concat([testData, one_hot_encoded_output_test], axis=1)
        X_test.drop(DROP, axis=1,
                    inplace=True)
        X_test_new = self.__getMinMaxScaler().transform(X_test)
        return str(self.__getTrainedModel().predict(X_test_new)[0])


currentModel = Model(True)


#@app.route('/create')
#def createModel():
#    try:
#        currentModel.create()
#        return "success"
#    except Exception as e:
#        return str(e)


#@app.route('/', methods=['POST'])
#def predict():
#    try:
#        return currentModel.predict(request.get_json(force=True))
#    except Exception as e:
#        return str(e)

@app.route('/')
def predict():
    return "sucess"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
