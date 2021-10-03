import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from svi_class import SVIClass

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

class SVIModel(BaseIVModel):
    def __init__(self, tolerance=1e-7):
        self.init_msigma = np.array([0.1,0.1])
        self.init_adc = np.array([0.03,0.33, 0.07])
        self.tolerance = tolerance
        self.svi_model_object = None
        self.model_fitted_vol = None
        self.time_to_expiry = None
        # calibrated_params = [_a_star, _d_star,_c_star, m_star, sigma_star]
        self.calibrated_params = []
        # params = [ a, b, rho, m, sigma]
        self.params = []


    def fit(self, option_chain):
        self.svi_model_object = SVIClass(option_chain.df, self.init_adc, self.init_msigma, tolerance=self.tolerance)
        self.model_fitted_vol= self.svi_model_object.svi_vol()
        self.calibrated_params = self.svi_model_object.calibrated_params
        self.time_to_expiry = option_chain.df.iloc[0]['time_to_expiry']

        a = self.calibrated_params[0]
        b = self.calibrated_params[2]/self.calibrated_params[4]
        rho = self.calibrated_params[1]/self.calibrated_params[2]

        self.params = [a, b, rho, self.calibrated_params[3], self.calibrated_params[4]]


    def get_fit_ivs(self, moneyness_arr):
        xi = -moneyness_arr
        y = (xi-self.calibrated_params[3])/self.calibrated_params[4]
        z = np.sqrt(y**2+1)
        omega = np.array(self.calibrated_params[0] + self.calibrated_params[1] * y + self.calibrated_params[2] * z)
        sigma = np.sqrt(omega/self.time_to_expiry)

        return sigma

    def get_greeks(self, moneyness_arr):
        df = pd.DataFrame(index=moneyness_arr)
        moneyness_arr = moneyness_arr.values
        xi = -moneyness_arr

        one_array = np.ones(moneyness_arr.shape[0])
        df['greek_0'] = np.ones(moneyness_arr.shape[0])
        df['greek_1'] = self.params[2]*(xi-self.params[3])+np.power(np.power(xi-self.params[3],2)+self.params[4]**2,0.5)
        df['greek_2'] = self.params[1]*(xi-self.params[3])
        df['greek_3'] = (np.divide((self.params[3]-xi),np.power(np.power(xi-self.params[3],2)+self.params[4]**2,0.5)) - self.params[2])*self.params[1]
        df['greek_4'] = np.divide(one_array*self.params[4],np.power(np.power(xi-self.params[3],2)+self.params[4]**2,0.5))*self.params[1]

        df = df.divide(2 * self.get_fit_ivs(moneyness_arr) * self.time_to_expiry, axis=0)
        return df
