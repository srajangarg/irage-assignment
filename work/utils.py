import pandas as pd
import numpy as np
import py_vollib_vectorized
from datetime import datetime, timedelta

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