#LSTM-SPP (LSTM Stock Price Prediction Model)

LSTM-SPP is a basic multi-variate, multi-step LSTM Recurrent Neural Network (RNN). It is configured specifically for stock price predictions, with the default settings allowing predictions of up to 7 trading days ahead. The project is readily modifiable by providing the core essentials in an algorithmic trading model based on Artificial Neural Networks. I strongly recommend to use of additional modules to increase the value of using LSTM-SPP.

#Possible Modules

Some possible modules that can be added into the project are as follows:

* Technical Analysis (TA) Modeling
    * Such a module can be provided to the Artifical Neural Network (ANN) for more accurate predictions of stock prices and trends.
    * The TA predictions can also help analysts in their trading decisions by providing crucial information nuances not available to traditional TA methods.

* Volatility/Volume Sensitivity
    * The market is unpredictable, but volatility sensitivities can help the ANN to make adjustments accordingly to provide results with less error and deviation. A sudden surge in volume can also indicate increased volatility.

* Convoluted-LSTM Hybrid Framework
    * In a paper titled 'Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach', the interesting concept of using CNNs for stock price prediction is explored with heavy reliance on technical analysis. A combination of the modules mentioned may provide better trading signals for analysts.

* Graph Visualisation
    * A simple implementation of the matplotlib or plotly library can help you visualise the predictions and existing data. This may be useful for technical analysts, especially after fine-tuning necessary settings and modifying the multi-step range.

#Usage
LSTM-SPP is not intended for use in real stock market conditions. While backtesting has returned low RMSE values (<1%) for a 100 epoch model, such a method of confidence is flawed due to the nature of predicting past data. Since LSTM-SPP only contains the core essentials, an array of modules must be added to ensure its effectiveness outside of a testing environment.

* With reference to a paper titled 'Application of Machine Learning: Automated Trading Informed by Event Driven Data', the examination of out-of-sample test sets and our empirical results showed that, despite poor prediction accuracy, the resulting trading strategies out-performed the market over a period. However, the results of the simple trading strategy based on the next day’s predicted return are limited. These trading models do not take into account market dynamics that could influence the cumulative return of the strategy. While liquidity may not be a big problem when trading a highly liquid instrument, otherfactors like transaction costs, bid-ask spreads and market impact of our trades could play a big role in reducing strategy returns. 

* Furthermore for believers of the Efficient Market Hypothesis (EMH), algorithmic trading and ANNs may not provide any further value than traditional methods of investment. 

If you are not deterred by the risks, the model can be used in the following manner:
1. Download historical OHLC data of your chosen equity in csv format, refer to the sample in the project folder for format requirements. 

2. Use main.py for the building of a LSTM-SPP model and prediction.py to produce predictions for the next 7 trading days. The main.py needs to run again every time a new OHLC data is used. 

3. You may choose to modify simple settings such as the epochs, number of neurons used and various other layers in the LSTM. Other simple modifications can include the time-steps and features for a personalised prediction model. Be mindful that prediction models generally suffer more as the time-step increases. Some features included may also mislead the ANN into providing a model that does not fit well with existing data. 

4. After running prediction.py, an array of 7 trading day stock price predictions is produced (on default settings). The values are in chronological order. 

#License
Copyright ©️ 2020 jaydxn1

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


