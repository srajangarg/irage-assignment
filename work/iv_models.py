import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseIVModel():
    def __init__(self):
        self.params = None
        pass

    def get_params(self):
        return self.params

    def fit(self):
        raise NotImplementedError()

    def get_fit_ivs(self):
        raise NotImplementedError()

    def get_greeks(self, moneyness_arr):
        raise NotImplementedError()

def kernel_smoothening(option_chain, k_const=0.00000075):
    def k_func(self, x, h_m):
        return np.exp(-1*(x**2)/(2*h_m))/(2*3.14)**(0.5)

    strike_vals    = option_chain.df['strike'].values
    moneyness_vals = option_chain.df['moneyness'].values
    mid_iv_vals    = option_chain.df['mid_iv'].values
    N_star         = len(strike_vals)
    h_m            = k_const*(max(strike_vals)-min(strike_vals))/(N_star-1)

    mid_iv_cap = []
    for a in range(N_star):

        m_j = moneyness_vals[a]

        denom = 0.0
        for b in range(N_star):
            denom += self.k_func(m_j-moneyness_vals[b], h_m)

        actual_sigma = 0
        for c in range(N_star):
            numer = self.k_func(m_j-moneyness_vals[c], h_m)
            actual_sigma += (numer/denom)*(mid_iv_vals[c])

        mid_iv_cap.append(actual_sigma)

    return(np.array(mid_iv_cap))


class BasicMidIVPolynomial(BaseIVModel):
    def __init__(self, degree=2, smoothening=None):
        self.degree = degree

        if smoothening is None:
            self.to_fit_vals = option_chain.df['mid_iv']
        elif smoothening == "kernel_smoothening":
            self.to_fit_vals = kernel_smoothening(option_chain)
        else:
            assert False

    def fit(self, option_chain):
        self.params = np.polyfit(option_chain.df['moneyness'], option_chain.df['mid_iv'], deg=self.degree)

    def get_fit_ivs(self, moneyness_arr):
        return np.poly1d(self.params)(moneyness_arr)

    def get_greeks(self, moneyness_arr):
        df = pd.DataFrame(index=moneyness_arr)
        for i in range(len(self.params)):
            df[f'greek_{i}'] = np.power(moneyness_arr.values, len(self.params) - 1 - i)

        return df
