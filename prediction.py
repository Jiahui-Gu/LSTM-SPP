import numpy as np
import pandas as pd
import datetime as dt
from keras.models import load_model
import matplotlib.pyplot as plt


# Date Parser
def parser(date):
    return dt.datetime.strptime(date, '%Y-%m-%d')


def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


model = load_model('//lstm_model')
data_raw = pd.read_csv('//data.csv', header=0, parse_dates=[0], index_col=0,
                       date_parser=parser)
data = data_raw.drop('5. volume', axis=1)
data = data.head(7)
data = data.values
n_input = 7
train_size = int(len(data))
train = data[0:train_size, :]
train = train.reshape((train.shape[0], 1, train.shape[1]))
history = [x for x in train]
predictions = list()
yhat_sequence = forecast(model, history, n_input)
predictions.append(yhat_sequence)
predictions = np.array(predictions)
print(predictions)
