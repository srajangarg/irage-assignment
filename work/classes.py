import glob
from datetime import datetime, timedelta
from utils import create_market_state
from utils import get_all_greeks_and_prices
import pandas as pd
import numpy as np
import py_vollib_vectorized

class Ticker():
    def __init__(self, ticker, expiry_date):
        expiry_dt   = datetime.strptime(expiry_date, '%Y%m%d')
        expiry_str  = expiry_dt.strftime("%y%b").upper()

        dfs      = []
        glob_str =  f"./data/*/*_{ticker}{expiry_str}*.csv"

        print("globbing", glob_str)
        for f in glob.glob(glob_str):
            print(f"loading data from {f}")
            expiry_time = expiry_dt + timedelta(hours=20)
            dfs.append(create_market_state(f, expiry_time))

        self.all_data = pd.concat(dfs).sort_index()

    def get_option_chain(self, timestamp):
        # get option chain at timestamp e.g. "20210623 15:15:00"
        return OptionChain(self.all_data[self.all_data.level_0 == timestamp])


class OptionChain():
    def __init__(self, df):
        self.df       = df.copy()
        # self.df contains all the "known" information including askPrice_1, ask_iv,
        # bidPrice_1, bid_iv, spread, spread_iv, futPrice, moneyness, time_to_expiry, instrument

    def fit_iv_model(self, iv_model):
        self.iv_model = iv_model
        self.iv_model.fit(self)
        self.df['fit_iv'] = self.iv_model.get_fit_ivs(self.df['moneyness'])

    def plot(self, ax):
        if self.iv_model is None:
            print("no IV model fit")
            return
        # ax.plot(self.df['moneyness'], self.df['bid_iv'])
        # ax.plot(self.df['moneyness'], self.df['ask_iv'])
        ax.plot(self.df['moneyness'], self.df['fit_iv'])


    def time_to_expiry(self):
        return self.df.time_to_expiry.mode().values[0]

    def future_price(self):
        return self.df.futPrice.mode().values[0]

    def iv_parameters(self):
        return self.iv_model.get_params()

    def ivs_prices_greeks(self, strikes_arr):
        df = pd.DataFrame(strikes_arr, columns=['strike'])

        df['futPrice']       = self.future_price()
        df['moneyness']      = np.log(df.strike/df.futPrice)
        df['instrument']     = np.where(df['moneyness'] > 0, 'c', 'p')
        df['time_to_expiry'] = self.time_to_expiry()
        df['fit_iv']         = self.iv_model.get_fit_ivs(df['moneyness'])

        regular_greeks_and_price = get_all_greeks_and_prices(df)
        vol_greeks               = self.iv_model.get_greeks(df['moneyness'])

        df = pd.concat([df, regular_greeks_and_price, vol_greeks.reset_index()], axis=1)
        return df




class Portfolio:
    def __init__(self, ticker, positions={}):
        self.ticker = ticker
        self.positions = positions


def result_df(oc1, oc2):
    moneys  = np.linspace(-0.15, 0.15, num=31)
    strikes = oc1.future_price() * np.exp(moneys)
    res1    = oc1.ivs_prices_greeks(strikes)
    res2    = oc2.ivs_prices_greeks(strikes)

    df = pd.DataFrame(moneys, columns=['moneyness'])

    df['delta_price'] = res1['delta'] * (oc2.future_price() - oc1.future_price())
    df['gamma_price'] = (res1['gamma'] / 2.0) * ((oc2.future_price() - oc1.future_price())**2)
    df['theta_price'] = res1['theta'] * (oc1.time_to_expiry() - oc2.time_to_expiry())

    for i in range(len(oc1.iv_parameters())):
        df[f'vol_{i}_price'] = res1['vega'] * res1[f'greek_{i}'] * (oc2.iv_parameters()[i] - oc1.iv_parameters()[i])

    df['estimated_price_diff'] = df['delta_price'] + df['gamma_price'] + df['theta_price']
    for i in range(len(oc1.iv_parameters())):
        df['estimated_price_diff'] += df[f'vol_{i}_price']

    df['actual_price_diff'] = res2['fit_price'] - res1['fit_price']
    df['abs_pricing_error'] = df['actual_price_diff'] - df['estimated_price_diff']
    df['delta_exposure'] = oc1.future_price() * res1['delta']
    df['%_pe_delta_exposure'] = 100 * df['abs_pricing_error'] / df['delta_exposure'].abs()
    df['%_pe_option_price'] = 100 * df['abs_pricing_error'] / res1['fit_price'].abs()

    return df.set_index('moneyness')




















