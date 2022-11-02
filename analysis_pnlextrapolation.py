#/usr/bin/python3
# -*- coding: utf-8 -*-
#
# PnL extrapolation Analysis for arbitrage trader
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from pandas.io.parsers import read_csv
from scipy import stats
import pickle
import os
import re
import analysis_pnl
from scipy.optimize import minimize

def extract_volume(out_filename):
    files, num_traders  = analysis_pnl.get_files()
    vol_vec = np.zeros(len(files))
    pnl_vec = np.zeros((len(files), 2))
    print("#traders; BTMXvolInBTC")
    initial_staker_cash = 1
    initial_margin_cash = 1
    initial_amm_cash = 1
    initial_default_fund_cash_cc = 1
    initial_investment = initial_default_fund_cash_cc + initial_amm_cash + initial_margin_cash
    for k in range(len(files)):
        filename = 'results/'+files[k]
        df = pd.read_csv(filename)
        vol_vec[k] = df['total_bitmex_btc_vol'][df.shape[0]-1]
        px = df['idx_px'][df.shape[0]-1]
        pnl_vec[k, 0] = df['arb_pnl'][df.shape[0]-1] + df['liquidator_earnings_vault'][df.shape[0]-1]*px
        
        pnl_vec[k, 1] = (df['protocol_earnings_vault'][df.shape[0]-1] + \
             + df['staker_cash'][df.shape[0]-1]-initial_staker_cash-initial_investment)*px
        dT = pd.to_datetime(df.iloc[[0,-1]]['time'], format="%y-%m-%d %H:%M", utc=True)
        num_trading_days = (dT.iloc[1].timestamp() - dT.iloc[0].timestamp())/86400
        print("trading days = ", num_trading_days)
        print(num_traders[k], "; ", vol_vec[k], "; ", pnl_vec[k])
    
    plt.style.use('bmh')
    plt.plot(num_traders, vol_vec, 'r-x')
    plt.xticks(num_traders)
    plt.xlabel("Number of traders")
    plt.ylabel("BitMEX Volume (BTC) "+str(round(num_trading_days))+" days")
    plt.show()
    plt.figure()
    plt.plot(num_traders, pnl_vec[:,0], 'r-x', label='BitMEX Link')
    plt.plot(num_traders, pnl_vec[:,1], 'b-x', label='protocol Profit')
    plt.legend()
    plt.xticks(num_traders)
    plt.xlabel("Number of traders")
    plt.ylabel("BitMEX Link PnL (USD) "+str(round(num_trading_days))+" days")
    plt.show()
    print("scaling factor:", 365/num_trading_days)

    M = np.stack((num_traders, vol_vec, pnl_vec[:,0], pnl_vec[:,1])).transpose()
    df = pd.DataFrame(M, columns = ['Traders','VolumeBTC', 'PnLBitMEXLink', 'PnLprotocol'])
    df.to_csv(out_filename)

def sort_vols(num_traders, vol_vec, pnl_vec):
    order_idx = np.argsort(num_traders)
    vol_vec[:] = [vol_vec[i] for i in order_idx]
    pnl_vec[:] = [pnl_vec[i] for i in order_idx] 
    num_traders[:] = [num_traders[i] for i in order_idx] 
    return num_traders, vol_vec, pnl_vec

if __name__=="__main__":
    out_filename = 'results/BitMEXBTCVolume.csv'
    extract_volume(out_filename)
    df = pd.read_csv(out_filename)
    num_traders= df["Traders"].to_numpy()
    vol_vec = df["VolumeBTC"].to_numpy()
    pnl_vec = df["PnLBitMEXLink"].to_numpy()
    pnl_vec2 = df["PnLprotocol"].to_numpy()
    dPnl_vec= (df['PnLBitMEXLink']/df["VolumeBTC"])#df["PnLBitMEXLink"].diff().to_numpy()
    #dPnl_vec[0] = pnl_vec[0]
    dVol_vec= df["VolumeBTC"].diff().to_numpy()
    dVol_vec[0] = vol_vec[0]
    dNum_traders= df["Traders"].diff().to_numpy()
    dNum_traders[0] = num_traders[0]
    dVol_vec = dVol_vec/dNum_traders

    def model(x, params):
        #return params[0]+params[1]*(x**0.2) +params[2]* (x**0.5)
        #return params[0]+params[1]**2 * (x**params[2])#+params[2]**2*(x**0.25)
        #return params[0]+params[1]*x-params[2]*np.exp(-params[3]*x)
        #return params[0]+np.abs(params[1]) * x - np.abs(params[2]) * x**(-np.abs(params[3]))
        return params[0]**2 + params[1] * x**params[2]

    def err(params):
        #idx = np.arange(5, num_traders.shape[0])
        y_hat = model(num_traders, params)
        y = dVol_vec
        return np.sum(((y_hat-y)/y)**2)
    
    def errPnL(params):
        #idx = np.arange(5, num_traders.shape[0])
        y_hat = model(num_traders, params)
        y = dPnl_vec
        return np.sum(((y_hat-y)/y)**2)

    x0=([0, 12, -2])
    res = minimize(err, x0, method = 'Nelder-Mead')
    print(res)
    print("parameters=", res.x)
    x_toshow = np.concatenate((num_traders, [500, 1000]))#np.arange(10, 800, 50)
    y_hat = model(x_toshow, res.x)
    plt.style.use('bmh')
    plt.plot(num_traders, dVol_vec, 'r-x', label='simulation')
    plt.plot(x_toshow, y_hat, 'b--+', label='model')
    plt.xticks(x_toshow)
    plt.xlabel("Number of traders")
    plt.ylabel("delta Vol / #Traders added")
    plt.legend()
    plt.show()

    x0=([0, 12, -2])
    resPnL = minimize(errPnL, x0, method = 'Nelder-Mead')
    print(resPnL)
    print("parameters=", resPnL.x)
    x_toshow = np.concatenate((num_traders, [500, 1000]))#np.arange(10, 800, 50)
    y_hat = model(x_toshow, resPnL.x)
    plt.style.use('bmh')
    plt.plot(num_traders, dPnl_vec, 'r-x', label='simulation')
    plt.plot(x_toshow, y_hat, 'b--+', label='model')
    plt.xticks(x_toshow)
    plt.xlabel("Number of traders")
    plt.ylabel("PnL / Volume")
    plt.legend()
    plt.show()


    def extrapolate_vol(num_traders):
        dV = model(num_traders, res.x)
        dTraders = [T-180 for T in num_traders] 
        return vol_vec[vol_vec.shape[0]-1]+dV*dTraders
    
    def extrapolate_pnl(num_traders):
        volume = extrapolate_vol(num_traders)
        pnl_per_volume = model(num_traders, resPnL.x)
        return pnl_per_volume*volume

    fig, axs = plt.subplots(1, 1)
    plt.style.use('bmh')
    axs.plot(num_traders, pnl_vec/1000, 'r-x', label='simulation')
    axs.set_xlabel("Number of traders")
    axs.set_ylabel("Cumulative BitMEX Link P&L (1,000 $) over simulation horizon")
    axs.set_xticks(x_toshow)
    x_extra = np.arange(180, 1040, 40)
    extrapol_pnl = extrapolate_pnl(x_extra)
    print(extrapolate_pnl)
    axs.plot(x_extra, extrapol_pnl/1000, 'b-d', label='extrapol')
    axs.legend()
    plt.savefig("BitmexLinkIncome.png")
    plt.show()

    fig, axs = plt.subplots(2, 1)
    plt.style.use('bmh')
    axs[0].plot(num_traders, vol_vec, 'r-x', label='simulation')
    axs[0].set_xlabel("Number of traders")
    axs[0].set_ylabel("Cumulative BitMEX vol (BTC) over simulation horizon")
    axs[0].set_xticks(x_toshow)
    
    x_extra = np.arange(180, 1000, 40)
    extrapol_vol = extrapolate_vol(x_extra)
    print(extrapol_vol)
    axs[0].plot(x_extra, extrapol_vol, 'b-d', label='extrapol')
    axs[0].legend()

    # annualized income
    px = 50000
    fee = 4*1e-4
    dTime = 365/282
    axs[1].plot(num_traders, vol_vec*fee*px*dTime, 'r-x', label='simulation')
    axs[1].plot(x_extra, extrapol_vol*fee*px*dTime, 'b-x', label='extrapolation')
    axs[1].set_xlabel("Number of traders")
    axs[1].set_ylabel("BitMEX Income at 50k BTCUSD,$ p.a.")
    plt.savefig("BitmexIncome.png")
    plt.show()
