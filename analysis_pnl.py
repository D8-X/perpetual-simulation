#/usr/bin/python3
# -*- coding: utf-8 -*-
#
# PnL Analysis for arbitrage trader
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats
import pickle
import os
import re

def get_files():
    file_list = os.listdir('results')
    regxp = re.compile("^res.*csv$")
    files = [x for x in file_list if regxp.match(x)]
    num_traders = [extract_num_traders(j) for j in files]
    # sort
    order_idx = np.argsort(num_traders)
    print(order_idx)
    files[:] = [files[i] for i in order_idx] 
    num_traders[:] = [num_traders[i] for i in order_idx] 
    return files, num_traders

def extract_num_traders(filename):
    name = re.sub("^res", "", filename)
    name = re.split("-", name)[0]
    return int(name)

files, num_traders = get_files()
pnl_vec = np.zeros(len(files))
bmx_vol_vec = np.zeros(len(files))
print("files = ", files)
print("num_traders = ", num_traders)

if __name__=="__main__":
    plt.figure()
    for k in range(len(files)):
        filename = 'results/'+files[k]
        #df = pd.read_pickle(filename)
        df = pd.read_csv(filename)
        dT = pd.to_datetime(df.iloc[[0,-1]]['time'], format="%y-%m-%d %H:%M", utc=True)
        num_trading_days = (dT.iloc[1].timestamp() - dT.iloc[0].timestamp())/86400
        endowment = pickle.load(open('results/Endowment'+str(num_traders[k])+"-"+str(int(num_traders[k]))+".pkl", "rb"))
        tot_initial_endowment = endowment['momentum']+endowment['noise']
        #pnl_tot[k] = float(df.iloc[[-1]]['arb_pnl']/(tot_initial_endowment*dTDays))
        pnl_vec[k] = df.iloc[[-1]]['arb_pnl']
        #bmx_vol_vec[k] = df.iloc[[-1]]['total_bitmex_btc_vol']/tot_initial_endowment
        print("#traders=", num_traders[k])
        print("pnl=", pnl_vec[k])
        plt.style.use('bmh')
        date_vec=pd.to_datetime(df['time'], format="%y-%m-%d %H:%M", utc=True)
        ts=(date_vec).apply(datetime.timestamp)
        mask = (ts% 60*60*24)==0
        print("1")
        plt.plot(date_vec[mask], df['total_bitmex_btc_vol'][mask], label=str(num_traders[k])+"traders")
        print("2")

    plt.ylabel("Additional BitMEX BTC Volume, BTC")
    plt.legend()
    plt.savefig("BitmexVolumeV2.png")
    plt.show()

    pnl_norm_vec = pnl_vec*100

    plt.style.use('bmh')
    plt.plot(num_traders, pnl_vec/1e6, 'r-x')
    plt.xticks(num_traders)
    plt.xlabel("Number of traders")
    plt.ylabel("BitMEX Link Arbitrage Profit, million $")
    plt.grid(True, 'major')
    plt.savefig("arb_pnlV2.png")
    plt.show()
    print(pnl_vec)


