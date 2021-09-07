import pandas as pd
import numpy as np
import py_vollib_vectorized
from datetime import datetime, timedelta
import scipy

def create_market_state(BCAST_FILE, expiry_time, get_greeks=True, mf_lower = -0.2, mf_upper = 0.2, RESAMPLING = '30s'):
    '''
    Creates and return Market State in a multiIndex Format
    '''
    bcast_file = BCAST_FILE
    bcast_data = pd.read_csv(bcast_file, index_col=0)
    bcast_data = bcast_data[~(bcast_data.bidPrice_1.isna() & bcast_data.askPrice_1.isna())]
    bcast_data = bcast_data[['bidPrice_1', 'askPrice_1', 'strike', 'instrument_type', 'sec']].sort_values('sec')
    bcast_data['midPrice'] = (bcast_data.bidPrice_1 + bcast_data.askPrice_1)/2
    bcast_data.instrument_type = bcast_data.instrument_type.str.strip('"').astype(str)
    bcast_data.strike = bcast_data.strike.str.strip('"').astype(int)
    bcast_data['futPrice'] = np.where(bcast_data.instrument_type == 'XX', bcast_data.midPrice, np.nan)
    bcast_data.futPrice = bcast_data.futPrice.ffill()
    bcast_data.index = pd.to_datetime(bcast_data.index).tz_convert(None)
    print(expiry_time)
    bcast_data['time_to_expiry'] = (expiry_time - bcast_data.index).total_seconds()/(86400*365)
    bcast_data = bcast_data[~(bcast_data.instrument_type == 'XX')]
    bcast_data = bcast_data[((bcast_data.instrument_type == 'PE') & (bcast_data.strike < bcast_data.futPrice)) | ((bcast_data.instrument_type == 'CE') & (bcast_data.strike > bcast_data.futPrice))]
    bcast_data['spread'] = bcast_data.askPrice_1 - bcast_data.bidPrice_1
    value_columns =  ['spread', 'futPrice', 'time_to_expiry', 'bidPrice_1', 'askPrice_1']
    curve_state = bcast_data.pivot_table(values = value_columns, index=bcast_data.index, columns=['strike']).ffill().resample(RESAMPLING).last().ffill()
    curve_state = curve_state.stack()
    curve_state['futPrice'] = curve_state.index.get_level_values(0).map(bcast_data.futPrice.resample(RESAMPLING).last().ffill().reset_index().set_index('index').futPrice.to_dict())

    curve_state = curve_state.reset_index()
    curve_state['moneyness'] = np.log(curve_state.strike/curve_state.futPrice)
    curve_state = curve_state[(curve_state['moneyness'] > mf_lower) & (curve_state['moneyness'] < mf_upper) ]

    curve_state['instrument'] = np.where(curve_state['moneyness'] > 0, 'c', 'p')
    curve_state['bid_iv'] = py_vollib_vectorized.vectorized_implied_volatility(curve_state.bidPrice_1, curve_state.futPrice, curve_state.strike, curve_state.time_to_expiry,  0, curve_state.instrument, return_as='numpy')
    curve_state['ask_iv'] = py_vollib_vectorized.vectorized_implied_volatility(curve_state.askPrice_1, curve_state.futPrice, curve_state.strike, curve_state.time_to_expiry,  0, curve_state.instrument, return_as='numpy')

    curve_state['mid_iv'] = (curve_state.bid_iv + curve_state.ask_iv)/2
    curve_state['spread_iv'] = curve_state.ask_iv - curve_state.bid_iv

    return curve_state

def get_all_greeks_and_prices(sdf):
    df = sdf.copy()
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
    df['cdf_d1_signed']  = np.where(df['instrument'] == 'c', cdf_d1, (1-cdf_d1))
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
    # vega2 = k_vec*e_rt*cdf_d2*np.power(t1,0.5)
    df['vega'] = vega1
    # df['vega2'] = vega2

    df['cdf_d2_signed'] =  np.where(df['instrument'] == 'c', cdf_d2, (1-cdf_d2))

    # gamma - formula 1
    df['gamma'] = np.divide(e_qt*cdf_d1,s_vec*sigma*np.power(t1,0.5))
    # df['gamma2'] = np.divide(k_vec*e_rt*cdf_d2,sigma*np.power(t1,0.5)*np.power(s_vec,2))

    theta_t1 = -1 * np.divide((e_qt*s_vec*cdf_d1*sigma),(2*np.power(t1,0.5)))

    cdf_d2_signed = df['cdf_d2_signed'].values
    cdf_d1_signed = df['cdf_d1_signed'].values
    theta_t2 = -1*r*k_vec*e_rt*cdf_d2_signed
    theta_t3 = q*s_vec*e_qt*cdf_d1_signed
    theta_t23 = theta_t2+theta_t3


    df['theta_t23_signed'] =  np.where(df['instrument'] == 'c', theta_t23, -1*theta_t23)

    df['theta'] = df['theta_t23_signed'].values+theta_t1

    price_unsigned = s_vec*np.exp((r-q)*t1)*cdf_d1_signed-k_vec*cdf_d2_signed
    df['price_unsigned'] = price_unsigned
    df['price_undiscounted'] =  np.where(df['instrument'] == 'c', price_unsigned, -1*price_unsigned)

    df['fit_price'] =  df.price_undiscounted.values*e_rt
    return df[['delta', 'gamma', 'theta',  'vega', 'fit_price']]


