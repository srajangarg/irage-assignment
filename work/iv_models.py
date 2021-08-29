import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseIVModel():
    def __init__(self):
        self.params = None
        pass

    def get_params(self):
        return self.params



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