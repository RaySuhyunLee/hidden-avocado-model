"""
HAM(Hidden Avocado Model)
Avocado price prediction using HMM

Original dataset by Justin Kiggins
https://www.kaggle.com/neuromusic/avocado-prices

This source code is mainly inspired by rubikscode's web tutorial
https://rubikscode.net/2018/10/29/stock-price-prediction-using-hidden-markov-model/
"""

from datetime import datetime
import time
import itertools
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# prediction params
HIDDEN_STATES = 2
latency = 10

# generate all possible descrete outcomes
frac_price_range = np.linspace(-5e-7, 5e-7, 40)
frac_volume_range = np.linspace(-40, 40, 40)

# plot settings
plt.style.use('ggplot')

possible_outcomes = np.array(list(itertools.product(
    frac_price_range, frac_volume_range)))

def prepare_data():
    df = pd.read_csv('avocado.csv')
    data = df[(df.region == 'TotalUS') & (df.type == 'conventional')]\
            [['Date', 'AveragePrice', 'Total Volume']]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Date'])
    train_data, test_data = train_test_split(
        data, test_size=0.25, shuffle=False)

    return (train_data, test_data)

def date_to_timestamp(dt):
    #return time.mktime(datetime.strptime(datestr, '%Y-%m-%d').timetuple())
    return (dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's') 

def extract_features(data):
    average_price = np.array(data['AveragePrice'])
    date = np.array(data['Date'])
    volume = np.array(data['Total Volume'])

    frac_price = []
    frac_volume = []
    average_price_past = average_price[0]
    date_past = date[0]
    volume_past = volume[0]

    for (ap, d, v) in zip(average_price[1:], date[1:], volume[1:]):
        time_elapsed = date_to_timestamp(d) - date_to_timestamp(date_past)
        frac_price.append((ap - average_price_past) / time_elapsed)
        frac_volume.append((v - volume_past) / time_elapsed)

        average_price_past = ap
        date_past = d
        volume_past = v

    # for debuging
    """print(max(frac_price))
    print(min(frac_price))
    print(max(frac_volume))
    print(min(frac_volume))"""
    return np.column_stack((frac_price, frac_volume)) 

def predict_frac(model, prev_features):
    # measure the score for each possible outcome
    outcome_score = []
    for po in possible_outcomes:
        data = np.row_stack((prev_features, po))
        outcome_score.append(model.score(data))

    predicted_frac = possible_outcomes[np.argmax(outcome_score)]
    return predicted_frac

def predict_avocado():
    (train_data, test_data) = prepare_data()
    train_features = extract_features(train_data)
    test_features = extract_features(test_data)
    test_prices = np.array(test_data['AveragePrice'])
    test_volumes = np.array(test_data['Total Volume'])
    test_dates = np.array(test_data['Date'])

    model = hmm.GaussianHMM(n_components=HIDDEN_STATES)

    # fit
    model.fit(train_features)

    # start price prediction
    predicted_prices = []
    predicted_volumes = []
    prediction_len = len(test_features) - latency
    for i in range(0, prediction_len):
        prev_features = test_features[i:i+latency]
        prev_price = test_prices[i+latency]
        prev_volume = test_volumes[i+latency]

        # get time delta between two dates
        prev_date = test_dates[i+latency]
        curr_date = test_dates[i+latency+1]
        time_elapsed = date_to_timestamp(curr_date) - date_to_timestamp(prev_date) 

        print('[Prediction] {} of {}'.format(i, prediction_len))
        (predicted_frac_price, predicted_frac_volume) = \
            predict_frac(model, prev_features)
        predicted_price = prev_price + predicted_frac_price * time_elapsed
        predicted_volume = prev_volume + predicted_frac_volume * time_elapsed
        predicted_prices.append(predicted_price)
        predicted_volumes.append(predicted_volume)

    # save prediction result
    real_prices = test_prices[latency:prediction_len + latency]
    real_volumes = test_volumes[latency:prediction_len + latency]
    dates = test_dates[latency:prediction_len + latency]
    prediction_result = {
            'date': dates,
            'predicted price': predicted_prices, 
            'real price': real_prices,
            'predicted volume': predicted_volumes,
            'real volume': real_volumes}
    pd.DataFrame(prediction_result).to_csv('prediction.csv')

    # plot
    fig1 = plt.figure()
    axes1 = fig1.add_subplot(111)
    axes1.set_title('Avocado Price Prediction')
    axes1.plot(dates, real_prices, "bo-", label="real")
    axes1.plot(dates, predicted_prices, "go-", label="predicted")
    axes1.legend()

    fig2 = plt.figure()
    axes2 = fig2.add_subplot(111)
    axes2.set_title('Avocado Volume Prediction')
    axes2.plot(dates, real_volumes, "bo-", label="real")
    axes2.plot(dates, predicted_volumes, "go-", label="predicted")
    axes2.legend()

    plt.show()

    # calculate sse
    residual = real_prices - predicted_prices
    sse = sum(residual**2)
    print('Sum of Squared Error: {}'.format(sse))

predict_avocado()
