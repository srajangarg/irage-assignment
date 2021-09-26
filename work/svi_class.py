#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:57:40 2018

@author: caolifeng
"""
from scipy.optimize import minimize
import numpy as np


class svi_model:
    def __init__(self, data, init_adc, init_msigma, tol):
        
        self.init_msigma = init_msigma
        self.init_adc = init_adc
        self.tol = tol
        self.data = data

    def forward_price(self):
        
        """
        s0 is the price of 50etf on the day
        Converted to the forward price of each option expiration
        """
        
        f = self.data.iloc[0]['futPrice']
        return f

    def outter_function(self, params):
       
        """
        outside
        """
       
        m, sigma = params
        sigma = max(0, sigma)
        adc_0 = self.init_adc
        f = self.forward_price()

        def inner_fun(params):
           
            """
            Inner function Use residual minimum fitting to estimate parameters a d c, slsqp
            Note that implied vol is converted to omega=vol**2*t
            """
            
            a, d, c = params
            error_sum = 0.0
            xi = np.power(self.data['moneyness'],-1)
            y = (xi-m)/sigma
            z = np.sqrt(y**2+1)
            error_sum = np.sum(np.array(a + d * y + c * z -
                                        np.array(self.data['mid_iv'])**2*self.data.iloc[0]['time_to_expiry']) ** 2)
            return error_sum
        bnds = (
            (1e-10, max(np.array(self.data['mid_iv']))), (-4*sigma, 4*sigma), (0, 4*sigma))
        b = np.array(bnds, float)
        cons = (
            {'type': 'ineq', 'fun': lambda x: x[2]-abs(x[1])},
            {'type': 'ineq', 'fun': lambda x: 4*sigma-x[2]-abs(x[1])}
        )
        inner_res = minimize(inner_fun, adc_0, method='SLSQP', tol=1e-6)
        
        a_star, d_star, c_star = inner_res.x
        self._a_star, self._d_star, self._c_star = inner_res.x

        sum = 0.0
        xi = np.power(self.data['moneyness'],-1)
        y = (xi-m)/sigma
        z = np.sqrt(y**2+1)
        sum = np.sum(np.array(a_star + d_star * y + c_star *
                              z - np.array(self.data['mid_iv'])**2*self.data.iloc[0]['time_to_expiry']) ** 2)
        return sum

    def optimization(self):
        
        """
        
        """

        outter_res = minimize(
            self.outter_function, self.init_msigma, method='Nelder-Mead', tol=self.tol)

        m_star, sigma_star = outter_res.x
        self._m_star, self._sigma_star = outter_res.x
        #obj = outter_res.fun
        calibrated_params = [self._a_star, self._d_star,
                             self._c_star, m_star, sigma_star]
        return calibrated_params

    def svi_vol(self):
       
        """
        
        """

        #f = self.forward_price()
        self.optimization()
        xi = np.power(self.data['moneyness'],-1)
        y = (xi-self._m_star)/self._sigma_star
        z = np.sqrt(y**2+1)
       
        omega = np.array(self._a_star + self._d_star * y + self._c_star * z)
        sigma = np.sqrt(omega/self.data.iloc[0]['time_to_expiry'])

        return sigma


