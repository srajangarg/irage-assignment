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
        return (self.df['mid_iv'] - self.df['fit_iv']).values

    def plot(self, ax):
        if self.iv_model is None:
            print("no IV model fit")
            return
        # ax.plot(self.df['moneyness'], self.df['bid_iv'])
        ax.plot(self.df['moneyness'], self.df['mid_iv'])
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
        df['moneyness']      = np.log(df.futPrice/df.strike)
        df['instrument']     = np.where(df['moneyness'] < 0, 'c', 'p')
        df['time_to_expiry'] = self.time_to_expiry()
        df['fit_iv']         = self.iv_model.get_fit_ivs(df['moneyness'])

        regular_greeks_and_price = get_all_greeks_and_prices(df)
        vol_greeks               = self.iv_model.get_greeks(df['moneyness'])

        df = pd.concat([df, regular_greeks_and_price, vol_greeks.reset_index()], axis=1)
        return df


    def estimate_price_diff_df(self, other, max_moneyness=0.15):
        moneys  = np.linspace(-max_moneyness, max_moneyness, num=int(200*max_moneyness+1))
        strikes = self.future_price() * np.exp(moneys)
        res1    = self.ivs_prices_greeks(strikes)
        res2    = other.ivs_prices_greeks(strikes)

        df = pd.DataFrame(moneys, columns=['moneyness'])

        df['delta_price'] = res1['delta'] * (other.future_price() - self.future_price())
        df['gamma_price'] = (res1['gamma'] / 2.0) * ((other.future_price() - self.future_price())**2)
        df['theta_price'] = res1['theta'] * (self.time_to_expiry() - other.time_to_expiry())

        assert len(self.iv_parameters()) == len(other.iv_parameters())
        for i in range(len(self.iv_parameters())):
            df[f'vol_{i}_price'] = res1['vega'] * res1[f'greek_{i}'] * (other.iv_parameters()[i] - self.iv_parameters()[i])

        df['estimated_price_diff'] = df['delta_price'] + df['gamma_price'] + df['theta_price']
        for i in range(len(self.iv_parameters())):
            df['estimated_price_diff'] += df[f'vol_{i}_price']

        df['actual_price_diff'] = res2['fit_price'] - res1['fit_price']
        df['abs_pricing_error'] = df['actual_price_diff'] - df['estimated_price_diff']
        df['delta_exposure'] = self.future_price() * res1['delta']
        df['%_pe_delta_exposure'] = 100 * df['abs_pricing_error'] / df['delta_exposure'].abs()
        df['%_pe_option_price'] = 100 * df['abs_pricing_error'] / res1['fit_price'].abs()

        return df.set_index('moneyness')

class Portfolio:
    def __init__(self, ticker, positions={}):
        self.ticker = ticker
        self.positions = positions


class TickerSet:
    def __init__(self, timestamp_1, timestamp_2, tickers=None, expiry_date="20210826"):
        if tickers is None:
            tickers = ['AARTIIND', 'ABFRL', 'ACC', 'ADANIENT', 'ADANIPORTS', 'ALKEM',
           'AMARAJABAT', 'AMBUJACEM', 'APLLTD', 'APOLLOHOSP', 'APOLLOTYRE',
           'ASHOKLEY', 'ASIANPAINT', 'ASTRAL', 'AUBANK', 'AUROPHARMA',
           'AXISBANK', 'BAJAJFINSV', 'BAJAJ_AUTO', 'BAJFINANCE', 'BALKRISIND',
           'BANDHANBNK', 'BANKBARODA', 'BATAINDIA', 'BEL', 'BERGEPAINT',
           'BHARATFORG', 'BHARTIARTL', 'BHEL', 'BIOCON', 'BOSCHLTD',
           'BPCL', 'BRITANNIA', 'CADILAHC', 'CANBK', 'CHOLAFIN', 'CIPLA',
           'COALINDIA', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL', 'CUB',
           'CUMMINSIND', 'DABUR', 'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY',
           'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL', 'GLENMARK',
           'GMRINFRA', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GUJGASLTD',
           'HAVELLS', 'HCLTECH', 'HDFC', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE',
           'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'IBULHSGFIN',
           'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDEA', 'IDFCFIRSTB', 'IGL',
           'INDHOTEL', 'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 'INFY', 'IOC',
           'IRCTC', 'ITC', 'JINDALSTEL', 'JSWSTEEL', 'JUBLFOOD', 'KOTAKBANK',
           'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTI', 'LTTS', 'LUPIN', 'L_TFH',
           'MANAPPURAM', 'MARICO', 'MARUTI', 'MCDOWELL_N', 'METROPOLIS',
           'MFSL', 'MGL', 'MINDTREE', 'MOTHERSUMI', 'MPHASIS', 'MRF',
           'MUTHOOTFIN', 'M_M', 'M_MFIN', 'NAM_INDIA', 'NATIONALUM', 'NAUKRI',
           'NAVINFLUOR', 'NESTLEIND', 'NMDC', 'NTPC', 'ONGC', 'PAGEIND', 'PEL',
           'PETRONET', 'PFC', 'PFIZER', 'PIDILITIND', 'PIIND', 'PNB', 'POWERGRID',
           'PVR', 'RAMCOCEM', 'RBLBANK', 'RECLTD', 'RELIANCE', 'SAIL', 'SBILIFE',
           'SBIN', 'SHREECEM', 'SIEMENS', 'SRF', 'SRTRANSFIN', 'STAR', 'SUNPHARMA',
           'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL',
           'TCS', 'TECHM', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TVSMOTOR',
           'UBL', 'ULTRACEMCO', 'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'ZEEL']


        self.oc_pairs = {}
        for ticker in tickers:
            t   = Ticker(ticker, expiry_date=expiry_date)
            oc1 = t.get_option_chain(timestamp_1)
            oc2 = t.get_option_chain(timestamp_2)
            if len(oc1.df) < 8 or len(oc2.df) < 8:
                print("skipped", ticker)
                continue

            self.oc_pairs[ticker] = (oc1, oc2)


    def get_option_chain_pairs(self):
        return self.oc_pairs
















