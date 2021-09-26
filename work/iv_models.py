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

def kernel_smoothening(option_chain, k_const=0.00000075, fit_column='mid_iv'):
    def k_func(x, h_m):
        return np.exp(-1*(x**2)/(2*h_m))/(2*3.14)**(0.5)

    strike_vals    = option_chain.df['strike'].values
    moneyness_vals = option_chain.df['moneyness'].values
    mid_iv_vals    = option_chain.df[fit_column].values
    N_star         = len(strike_vals)
    h_m            = k_const*(max(strike_vals)-min(strike_vals))/(N_star-1)

    mid_iv_cap = []
    for a in range(N_star):

        m_j = moneyness_vals[a]

        denom = 0.0
        for b in range(N_star):
            denom += k_func(m_j-moneyness_vals[b], h_m)

        actual_sigma = 0
        for c in range(N_star):
            numer = k_func(m_j-moneyness_vals[c], h_m)
            actual_sigma += (numer/denom)*(mid_iv_vals[c])

        mid_iv_cap.append(actual_sigma)

    return(np.array(mid_iv_cap))


class IVPolynomial(BaseIVModel):
    def __init__(self, degree=2, smoothening=None, weighting=None):
        self.degree      = degree
        self.smoothening = smoothening
        self.weighting   = weighting

    def fit(self, option_chain):
        if self.smoothening is None:
            to_fit_vals = option_chain.df['mid_iv']
        elif self.smoothening == "kernel_smoothening":
            to_fit_vals = kernel_smoothening(option_chain)
        else:
            raise NotImplementedError()

        if self.weighting is None:
            weights = None
        elif self.weighting == "inverse_spread":
            weights = np.divide(1.0, option_chain.df['spread'])
        else:
            raise NotImplementedError()

        self.params = np.polyfit(option_chain.df['moneyness'], to_fit_vals, deg=self.degree, w=weights)

    def get_fit_ivs(self, moneyness_arr):
        return np.poly1d(self.params)(moneyness_arr)

    def get_greeks(self, moneyness_arr):
        df = pd.DataFrame(index=moneyness_arr)
        for i in range(len(self.params)):
            df[f'greek_{i}'] = np.power(moneyness_arr.values, len(self.params) - 1 - i)

        return df
