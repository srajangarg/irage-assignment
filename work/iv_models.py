import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseIVModel():
    def __init__(self):
        self.params = None
        pass

    def get_params(self):
        return self.params


class MidIV(BaseIVModel):
    def __init__(self):
        pass

    def fit(self, option_chain):
        self.params = option_chain.df['mid_iv'].values
        self.iv_map = {}
        self.ix_map = {}

        for i, (ix, row) in enumerate(option_chain.df.iterrows()):
            self.iv_map[row['moneyness'] ] = row['mid_iv']
            self.ix_map[row['moneyness'] ] = i


    def get_fit_ivs(self, moneyness_arr):
        return [self.iv_map[moneyness] for moneyness in moneyness_arr]

    def get_greeks(self, moneyness_arr):

        df = pd.DataFrame(index=moneyness_arr)
        for i in range(len(self.params)):
            df[f'greek_{i}'] = (i == moneyness_arr.map(self.ix_map)).astype(float).values

        return df

class ModifiedMidIVPolynomial(BaseIVModel):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, option_chain):
        self.params = np.polyfit(option_chain.df['moneyness'], option_chain.modified_mid_iv(), deg=self.degree)

    def get_fit_ivs(self, moneyness_arr):
        return np.poly1d(self.params)(moneyness_arr)

    def get_greeks(self, moneyness_arr):
        df = pd.DataFrame(index=moneyness_arr)
        for i in range(len(self.params)):
            df[f'greek_{i}'] = np.power(moneyness_arr.values, len(self.params) - 1 - i)

        return df

class BasicMidIVPolynomial(BaseIVModel):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, option_chain):
        self.params = np.polyfit(option_chain.df['moneyness'], option_chain.df['mid_iv'], deg=self.degree)

    def get_fit_ivs(self, moneyness_arr):
        return np.poly1d(self.params)(moneyness_arr)

    def get_greeks(self, moneyness_arr):
        df = pd.DataFrame(index=moneyness_arr)
        for i in range(len(self.params)):
            df[f'greek_{i}'] = np.power(moneyness_arr.values, len(self.params) - 1 - i)

        return df
