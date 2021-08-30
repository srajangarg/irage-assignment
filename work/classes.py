import glob
from datetime import datetime, timedelta
from utils import create_market_state
import pandas as pd
import numpy as np
import py_vollib_vectorized
import scipy

class Ticker():
    def __init__(self, ticker, expiry_date):
        expiry_dt   = datetime.strptime(expiry_date, '%Y%m%d')
        expiry_str  = expiry_dt.strftime("%y%b").upper()

        dfs = []
        for f in glob.glob(f"./data/*/*_{ticker}{expiry_str}*.csv"):
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


    def personal_prices_greeks(self, strikes_arr):
        df = pd.DataFrame(strikes_arr, columns=['strike'])

        df['futPrice']       = self.future_price()
        df['moneyness']      = np.log(df.strike/df.futPrice)
        df['instrument']     = np.where(df['moneyness'] > 0, 'c', 'p')
        df['time_to_expiry'] = self.time_to_expiry()
        df['fit_iv']         = self.iv_model.get_fit_ivs(df['moneyness'])


        # s/k vector
        s_vec = df.futPrice.values
        k_vec = df.strike.values
        d_sk = np.log(np.divide(s_vec,k_vec))
        nlen = df.futPrice.values.shape[0]
        # interest rate
        r = np.zeros(nlen)
        # dividends
        q = np.zeros(nlen)
        sigma = df.fit_iv.values
        t1 = df.time_to_expiry.values
        # Calculate d1
        d1 = (d_sk+((r-q+np.power(sigma,2)/2)*(t1)))/(sigma*np.power(t1,0.5))
        # Calculate cdf of d1
        cdf_d1 = scipy.stats.norm.cdf(d1)
        df['cdf_d1'] = d1
        # Delta in Put is CDF(-d1)
        df['cdf_d1_signed']     = np.where(df['instrument'] == 'c', cdf_d1, (1-cdf_d1))
        e_qt = np.exp(-1*q*t1)
        e_rt = np.exp(-1*r*t1)
        df['e_qt'] = e_qt
        # Delta in Put has negative sign
        df['e_qt_with_delta_sign']     = np.where(df['instrument'] == 'c', e_qt, -1*e_qt)
        df['delta'] = df['e_qt_with_delta_sign']*df['cdf_d1_signed'] 


        # Calculate d2 using d1
        d2 = d1-sigma*np.power(t1,0.5)
        cdf_d2 = scipy.stats.norm.cdf(d2)
        df['cdf_d2'] =  cdf_d2
        # use vega from wikipedia
        vega1 = s_vec*e_qt*cdf_d1*np.power(t1,0.5)
        vega2 = k_vec*e_rt*cdf_d2*np.power(t1,0.5)
        df['vega1'] = vega1
        df['vega2'] = vega2


        # rho has negative d2 in put
        df['cdf_d2_signed'] =  np.where(df['instrument'] == 'c', cdf_d2, (1-cdf_d2))
        rho_unsigned = df.cdf_d2_signed.values * t1*k_vec*e_rt
        df['rho_unsigned'] = rho_unsigned
        # rho has a negative sign outside in put formula
        df['rho'] =  np.where(df['instrument'] == 'c', rho_unsigned, -1*rho_unsigned)


        # gamma - formula 1
        df['gamma1'] = np.divide(e_qt*cdf_d1,s_vec*sigma*np.power(t1,0.5))
        df['gamma2'] = np.divide(k_vec*e_rt*cdf_d2,sigma*np.power(t1,0.5)*np.power(s_vec,2))

        theta_t1 = -1 * np.divide((e_qt*s_vec*cdf_d1*sigma),(2*np.power(t1,0.5)))

        cdf_d2_signed = df['cdf_d2_signed'].values
        cdf_d1_signed = df['cdf_d1_signed'].values
        theta_t2 = -1*r*k_vec*e_rt*cdf_d2_signed
        theta_t3 = q*s_vec*e_qt*cdf_d1_signed
        theta_t23 = theta_t2+theta_t3

        df['theta_t1'] = theta_t1
        df['theta_t23'] = theta_t23
        df['theta_t23_signed'] =  np.where(df['instrument'] == 'c', theta_t23, -1*theta_t23)

        df['theta'] = df['theta_t23_signed'].values+theta_t1

        price_unsigned = s_vec*np.exp((r-q)*t1)*cdf_d1_signed-k_vec*cdf_d2_signed
        df['price_unsigned'] = price_unsigned
        df['price_undiscounted'] =  np.where(df['instrument'] == 'c', price_unsigned, -1*price_unsigned)

        df['price'] =  df.price_undiscounted.values*e_rt

        df1 = df[['strike', 'futPrice', 'moneyness',    'instrument',   'time_to_expiry',   'fit_iv']]

        df2 = df[['delta',  'gamma1','gamma2',  'theta',    'rho',  'vega1','vega2']]
        df3 = df[['price']]
        # df3

        return df1, df2, df3

    def ivs_prices_greeks(self, strikes_arr):
        df = pd.DataFrame(strikes_arr, columns=['strike'])

        df['futPrice']       = self.future_price()
        df['moneyness']      = np.log(df.strike/df.futPrice)
        df['instrument']     = np.where(df['moneyness'] > 0, 'c', 'p')
        df['time_to_expiry'] = self.time_to_expiry()
        df['fit_iv']         = self.iv_model.get_fit_ivs(df['moneyness'])


        price   = py_vollib_vectorized.models.vectorized_black_scholes(df.instrument, df.futPrice, df.strike,
                                                                                df.time_to_expiry, 0, df.fit_iv)
        greeks  = py_vollib_vectorized.get_all_greeks(df.instrument, df.futPrice, df.strike,
                                                      df.time_to_expiry, 0, df.fit_iv)
        # df = pd.concat([df.reset_index(), greeks], axis=1)

        return df, greeks, price

#        return df

class Portfolio:
    def __init__(self, ticker, positions={}):
        self.ticker = ticker
        self.positions = positions


























