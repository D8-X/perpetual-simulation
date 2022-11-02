# emulate like blockchain
import requests
import json
import time

from sqlalchemy import null
import emulator_parameters
from amm import AMM
from trader import CollateralCurrency
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 

def fetch_index_price():
    #https://developers.coinbase.com/api/v2
    res = requests.get("https://api.coinbase.com/v2/prices/btc-usd/spot")
    res._content
    a_json = json.loads(res._content)
    return float(a_json["data"]["amount"])

def construct_price_vec(perp):
    global v_amounts
    v_prices = [perp.get_price(x) for x in v_amounts]
    return [v_prices, v_amounts]

def traders_trade(traders, amm, time_idx):
    # traders trade in random (but np.random.seed-predictable) order
    idx_arr = np.random.choice(len(traders), len(traders), replace=False)
    perp = amm.get_perpetual(perp_idx)
    trades = np.empty((0,3), float)
    for k in idx_arr:
        [dPos, is_close] = traders[k].query_trade_amount()
        if np.abs(dPos) >= perp.params['fLotSizeBC']:
            px = traders[k].trade(dPos, is_close)
            if px is not None:
                trades = np.vstack([ trades, np.array([time_idx, px, dPos]) ])
    return trades

# initialization ---------
T = 30
block_wait_sec = 3
num_noise_traders = 50
[perp_params, amm_params] = emulator_parameters.set_parameters()
amm = AMM(amm_params, amm_params['initial_staker_cash_cc'], amm_params['initial_default_fund_cash_cc'])
idx_s2 = fetch_index_price()
perp_idx = amm.add_perpetual(
                    perp_params['BTC']['initial_amm_cash_cc'], 
                    perp_params['BTC']['initial_margin_cash_cc'],
                    np.array([idx_s2]), # no index data feed, just 1 value
                    None, # we don't include any quanto data
                    perp_params['BTC'], 
                    CollateralCurrency.BASE,
                    #
                    min_spread = perp_params["BTC"]['fMinimalSpread'], 
                    incentive_rate = 0
                )

trader_list = emulator_parameters.initialize_traders(num_noise_traders, perp_idx, amm, idx_s2, cash_qc=10_000)

# to run GUI event loop
#plt.ion()
#figure = plt.figure()
#ax = plt.axes(projection='3d')
#x = np.linspace(0, 10, 100)
#y = np.sin(x)
#line1, = ax.plot(x, y)
#plt.grid()
#plt.axis([ -5, 5, 32_000, 36_000])

v_amounts = np.arange(-5, 5, 0.01)
v_amounts = np.setdiff1d(v_amounts, v_amounts[np.abs(v_amounts)<0.002])
N = v_amounts.shape[0]
Volume = np.zeros((T,N), float)
Price = np.empty((T,N), float)
Time = np.empty((T,N), float)
trades = np.empty((0,3), float)
# time-loop --------------
for t in range(T):
    print("t={:.0f}".format(t))
    # update oracle price
    idx_s2 = fetch_index_price()
    print("idx price = ", idx_s2)
    amm.perpetual_list[perp_idx].set_idx_s2(idx_s2, 0)
    # trade
    
    trades_now = traders_trade(trader_list, amm, t)
    if len(trades_now)>0:
        trades =  np.vstack([ trades, trades_now ])
    # construct price vector
    [price_vec, vol_vec] = construct_price_vec(amm.perpetual_list[perp_idx])
    Time[t,:] = t*np.ones((1, len(price_vec)))
    Price[t,:] = np.array(price_vec)
    Volume[t,:] = np.array(vol_vec)
    print(vol_vec[0:5])
    print(price_vec[0:5])
    
    #figure.canvas.draw()
    #figure.canvas.flush_events()

    # wait for block to finish
    print("sleep...")
    time.sleep(block_wait_sec)
    print("-")
    # update mark price before block starts
    amm.perpetual_list[perp_idx].update_mark_price()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z)
ax.plot_surface(Time, Price, np.abs(Volume), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
idx_buy  = trades[:,2]>0
idx_sell = trades[:,2]<0
ax.scatter(trades[idx_buy,0], trades[idx_buy,1], np.abs(trades[idx_buy,2]), marker = 'o', c='green')
ax.scatter(trades[idx_sell,0], trades[idx_sell,1], np.abs(trades[idx_sell,2]), marker = 's', c='red')
ax.set_zlabel('volume')
ax.set_xlabel('time')
ax.set_ylabel('price')
ax.set_title('surface')
#figure.canvas.draw()
plt.show()
print('done')
