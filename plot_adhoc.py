#%% import and read
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.pyplot import cm
from pricing_benchmark import prob_def_quanto
from simulation import init_index_data, load_perp_params
import seaborn as sns
from matplotlib import font_manager

font_path = './inter/Inter Desktop/Inter-Regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
# plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


COLL = 'MATIC'
QUOTE = 'USD'

FILENAME = 'results/res103-20-AVAXUSDMATIC_2022918-5453461430419889281.csv'

perpsymbol = re.search("-[A-Z]+_", FILENAME).group(0)[1:-1]
INDEX = perpsymbol[:(len(perpsymbol) - len(QUOTE + COLL))]

def main():
    
    file = FILENAME if FILENAME[-3:] == 'csv' else FILENAME + '.csv'
    df = pd.read_csv(file)
    
    

    df['mid_price_rel'] = (df['mid_price'] - df['idx_px']) / df['idx_px']
    df['mark_price_rel'] = (df['mark_price'] - df['idx_px']) / df['idx_px']
    df['cex_price_rel'] = df['cex_px'] / df['idx_px'] - 1
    df['datetime'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M', utc=True)
    df['timestamp'] = df['datetime'].apply(datetime.datetime.timestamp)
    
    df = df[(df['datetime'] >= datetime.datetime(2022, 9, 18, 1, 0, 1, tzinfo=datetime.timezone.utc)) & (df['datetime'] < datetime.datetime(2022, 9, 18, 2, 15, 0,  tzinfo=datetime.timezone.utc))]
    
    price_columns = [x for x in df.columns if re.search('^perp_pi', x)]
    price_columns.reverse()
    if len(price_columns) > 0:
        # transform price into price impact
        for col in price_columns:
            df[col] = df[col] / df['mid_price'] - 1

    from_date = df['timestamp'].min()
    to_date = df['timestamp'].max()

    print(f"Date range: {datetime.datetime.fromtimestamp(from_date)} -- {datetime.datetime.fromtimestamp(to_date)}")
    plot_analysis(df, file)


def plot_perp_slippage(fig, df):
    slippage = [x for x in df.columns if re.search('^perp_price', x)]
    slippage = [slippage[0], slippage[5], slippage[10],  slippage[15],  slippage[-16], slippage[-11], slippage[-6], slippage[-1]]
    slippage.reverse()
    # print(perp_pnls)
    color = iter(cm.rainbow(np.linspace(0, 1, len(slippage))))
    ax = fig.add_subplot()
    for i, p in enumerate(slippage):
        if i % 4 != 0:
            next
        c = next(color)
        ax.plot(df['datetime'], 1e4 * df[p], 'o-', c=c, label=re.sub("perp_price_", "", p))

    plt.xlabel("Time")
    plt.ylabel("Slippage (from mid-price, bps)")
    
    plt.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
    fig.autofmt_xdate()

def plot_perp_price(fig, df):
    
    color = iter(cm.rainbow(np.linspace(0, 1, 2)))
    ax = fig.add_subplot()
    c = next(color)
    ax.plot(df['datetime'], df['mark_price'], 'o-', c=c, label="Mark-price")
    c = next(color)
    ax.plot(df['datetime'], df['idx_px'], 'o-', c=c, label="Oracle")
    # c = next(color)
    # ax.plot(df['datetime'], 1e4 * df['mid_price'], 'o-', c=c, label="Mid-price")
    
    
    
    plt.xlabel("Time")
    plt.ylabel(f"{INDEX}/{QUOTE}")
    
    plt.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
    fig.autofmt_xdate()

def plot_analysis(df, file=None):
    print(f"Index: {INDEX}, Quote: {QUOTE}, Collateral: {COLL}")
    fig = plt.figure(figsize=(10,6), tight_layout=True)
    plot_perp_slippage(fig, df)
    # plot_perp_price(fig, df)
    plt.show()
    
def plot_oracles():
    symbols = ["BTCUSD", "ETHUSD", "MATICUSD"]
    (idx_s2, idx_s3, cex_ts, time_df) = init_index_data(
        datetime.datetime(2022, 5, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp(), 
        datetime.datetime(2022, 8, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp(), 
        reload=False, 
        symbol=symbols, 
        collateral=COLL
    )
    print(time_df.head(5))
    print(time_df.tail(5))
    time_df = pd.to_datetime(time_df.values[::60], format='%y-%m-%d %H:%M', utc=True)
    for sym in symbols:
        print(f"\n{sym}:\n")
        print(idx_s2[sym][:5])
        print(idx_s2[sym][-5:])
        idx_s2[sym] = idx_s2[sym][::60]
        
    fig, axs = plt.subplots(len(symbols), 1, sharex=True)
    for i, sym in enumerate(symbols):
        print(f"Plotting {sym}...")
        axs[i].plot(time_df, idx_s2[sym], "-", color="#664ADF", alpha=1)
        axs[i].set_ylabel(f"{sym}")
        axs[i].set_xlabel("Time")
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H'))
    fig.autofmt_xdate()
    fig.set_size_inches(10, 15)
    plt.show()
    
def plot_insurance_noquanto():
    # state
    M1 = 0 #3*25_000
    M2 = 3
    K2 = 4
    L1 = 23_000 * K2
    
    # trades
    k = np.arange(-16, 11, 0.01)
    k = np.where( np.abs(k) > 0.1, k, np.nan)
    
    # params
    s20 = 25_000
    sig = 0.08
    r = 0.01
    
    # for plot
    pow = 3

    expTheta = np.minimum(1e10, (M1 + L1 + k * s20) / ((K2 + k - M2) * s20))
    Theta = np.where(expTheta > 1e-10, np.log(expTheta), np.nan)
    d1 = (Theta - r + 0.5 * sig**2) / sig
    d2 = (Theta - r - 0.5 * sig**2) / sig
    
    ins = np.where(
        (K2 + k <= M2) & (M1 + L1 + k * s20 >= 0),
        0,
        np.where(
            (K2 + k >= M2) & (M1 + L1 + k * s20 <= 0),
            (K2 + k - M2) * np.exp(r) * s20 - (M1 + L1 + k * s20),
            np.where(
                (K2 + k > M2) & (M1 + L1 + k * s20 > 0),
                (K2 + k - M2) * np.exp(r) * s20 * (1 - norm.cdf(d2)) - (M1 + L1 + k * s20) * (1 - norm.cdf(d1)),
                np.where(
                    (K2 + k < M2) & (M1 + L1 + k * s20 < 0),
                    (K2 + k - M2) * np.exp(r) * s20 * norm.cdf(d2) - (M1 + L1 + k * s20) * norm.cdf(d1),
                    np.nan
                )
            )
        )
    ) / (np.abs(k) * s20)
    ins = np.where(ins >= 0, ins, np.nan) # numerical noise
    
    PD = np.where(
        (K2 + k <= M2) & (M1 + L1 + k * s20 >= 0),
        0,
        np.where(
            (K2 + k >= M2) & (M1 + L1 + k * s20 <= 0),
            1,
            np.where(
                (K2 + k > M2) & (M1 + L1 + k * s20 > 0),
                1 - norm.cdf(d1),
                np.where(
                    (K2 + k < M2) & (M1 + L1 + k * s20 < 0),
                    norm.cdf(d1),
                    np.nan
                )
            )
        )
    )
    
    plt.plot(k, ins ** (1 / pow), ":", color="#664ADF", alpha=1, label="Insurance premium")
    plt.plot(k, PD ** (1 / pow), "-", color="#FFDA9F", alpha=1, label="PD")
    locs, labels = plt.yticks()
    plt.yticks(locs, [f"{1e4 * x ** pow:.1f}" for x in locs])
    plt.xlabel(f"Trade size (BTC)")
    plt.ylabel(f"Basis Points")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    plt.show()
    
def plot_insurance_quanto():
    
    # params
    s20 = 25_000
    s30 = 0.9
    sig2 = 0.08
    sig3 = 0.12
    rho = 0.25
    r = 0.01
    
    # state
    M1 = 0 #3*25_000
    M2 = 0
    M3 = 3 * s20 / s30
    K2 = 6
    L1 = 23_000 * K2
    
    # trades
    k = np.arange(-15, 5, 0.01)
    k = np.where( np.abs(k) > 0.01, k, np.nan)
    
    # for plot
    pow = 2

    def prob_theta(a, b, c, n=100_000):
        # returns Prob(a  + b * exp(r2) + c * exp(r3) <= 0)
        z = np.random.multivariate_normal(
            (r - sig2 ** 2 / 2, r - sig3 ** 2 / 2),
            ((sig2 ** 2, rho * sig2 * sig3), (rho * sig2 * sig3, sig3 ** 2)),
            size=n
        )
        r2, r3 = z[:,0,np.newaxis], z[:,1,np.newaxis] 
        balance = np.ones_like(r2) * a  + np.exp(r2) * b + np.exp(r3) * c
        ind = np.where(balance <= 0, 1, 0)
        return np.sum(ind, axis=0) / n
    
    n = 200_000
    z = np.random.multivariate_normal(
            (r - sig2 ** 2 / 2, r - sig3 ** 2 / 2),
            ((sig2 ** 2, rho * sig2 * sig3), (rho * sig2 * sig3, sig3 ** 2)),
            size=n
        )
    r2, r3 = z[:,0,np.newaxis], z[:,1,np.newaxis] 
    ins = (np.exp(r2) - 1) * k * s20 - (M1 + np.exp(r2) * s20 * M2 + np.exp(r3)* s30 * M3 - np.exp(r2) * s20 * K2 + L1)
    ins = np.where((ins > 0) & (r2 * k > 0), ins , 0)
    ins = np.mean(ins, axis=0) / (np.abs(k) * s20)
     
    # PD = prob_theta(M1 + L1 + k * s20, (M2 - K2 - k) * s20, M3 * s30, n=400_000)
    PD = np.array([prob_def_quanto(K2 + kk, L1+ kk * s20, s20, s30, sig2, sig3, rho, r, M1, M2, M3)[0] for kk in k])
    
    plt.plot(k, ins ** (1 / pow), ":", color="#664ADF", alpha=1, label="Insurance premium")
    plt.plot(k, PD ** (1 / pow), "-", color="#FFDA9F", alpha=1, label="PD")
    locs, labels = plt.yticks()
    plt.yticks(locs, [f"{1e4 * x ** pow:.1f}" for x in locs])
    plt.xlabel(f"Trade size (BTC)")
    plt.ylabel(f"Basis Points")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    plt.show()

def plot_revenue_stats(run_id, stat, nrows=None, ylab=None, legend=None):
    run_file = f"./results/sim_run_id_{run_id}.csv"
    df_run = pd.read_csv(run_file)
    sim_ids = df_run["Sim_ID"]
    res = None
    for i, sim_id in enumerate(sim_ids):
        sim_file = f"res[0-9]+-[0-9]+-[A-Z]+{QUOTE}{COLL}_[0-9]+-{sim_id}.csv"
        files = [f for f in os.listdir("./results/") if re.search(sim_file, f)]
        if len(files) == 0:
            print(f"Sim id {sim_id} not found for run id {run_id}.")
            continue
        for f in files:
            df_sim = pd.read_csv(f"./results/{f}", usecols=stat + ["time", "idx_s3"], nrows=nrows)
            if res is None:
                res = np.zeros((df_sim.shape[0], len(sim_ids), 3), dtype=np.float64, order="F")
            res[:,i,0] = (df_sim['df_cash'] + df_sim['amm_cash'] + df_sim['pool_margin'] + df_sim['protocol_earnings_vault']).values
            res[:,i,1] += df_sim['perp_volume_qc'].values
            res[:,i,2] = df_sim['idx_s3'].values
       
        print(f"Loaded sim {sim_id} ({i+1}/{len(sim_ids)})")
            
    res = res[::(60*12),:,:]
    res[:,:,0] -= res[0,:,0]
    time_df = pd.to_datetime(df_sim['time'], format='%y-%m-%d %H:%M', utc=True).values[::(60*12)]
    
    fig, axs = plt.subplots(3, 1, sharex=True)
    
    axs[0].plot(time_df, 1e-6 * np.median(res[:,:,1], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[0].set_ylabel("USD (mm)")
    axs[0].set_title("Traded Volume")
    
    axs[1].plot(time_df, 1e-6 * np.median(res[:,:,0] * res[:,:,2], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[1].set_ylabel("USD (mm)")
    axs[1].set_title("Total Earnings")
    
    axs[2].plot(time_df, 100 * np.median(res[:,:,0] * res[:,:,2] / res[:,:,1], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 * np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 *np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 * np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 *np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[2].set_ylabel("%")
    axs[2].set_title("Total Earnings per Volume")
    
    fig.autofmt_xdate()
    fig.set_size_inches(10, 15)
    plt.xlabel("Time")
    plt.show()
    
def plot_realized_revenue_stats(run_id, stat, nrows=None, ylab=None, legend=None):
    run_file = f"./results/sim_run_id_{run_id}.csv"
    df_run = pd.read_csv(run_file)
    sim_ids = df_run["Sim_ID"]
    res = None
    dtypes = {a: np.float64 for a in stat}
    dtypes['time'] = pd.StringDtype()
    usecols = stat + ["time"]
    for i, sim_id in enumerate(sim_ids):
        sim_file = f"res[0-9]+-[0-9]+-[A-Z]+{QUOTE}{COLL}_[0-9]+-{sim_id}.csv"
        files = [f for f in os.listdir("./results/") if re.search(sim_file, f)]
        if len(files) == 0:
            print(f"Sim id {sim_id} not found for run id {run_id}.")
            continue
        amm_balance = 0
        for f in files:
            df_sim = pd.read_csv(f"./results/{f}", usecols=usecols, dtype=dtypes, nrows=nrows, low_memory=False) # 
            df_sim = df_sim.iloc[::(60*24)]
            if res is None:
                res = np.zeros((df_sim.shape[0], len(sim_ids), 5), dtype=np.float64, order="F")
            # all the liquidity in the protocol that is not trader-owned
            # amm balance at spot
            amm_balance += (df_sim['perp_margin'] + (df_sim['perp_K2'] * df_sim['idx_px'] - df_sim['perp_L1']) / df_sim['idx_s3']).values
            # res[:,i,0] = (df_sim['df_cash'] + df_sim['amm_cash'] + df_sim['pool_margin'] + df_sim['protocol_earnings_vault'] + df_sim['staker_cash']).values
            res[:,i,0] = (df_sim['df_cash'] + df_sim['amm_cash'] + df_sim['protocol_earnings_vault'] + df_sim['staker_cash']).values
            res[:,i,1] += df_sim['perp_volume_qc'].values
            res[:,i,2] = df_sim['idx_s3'].values
            res[:,i,4] = (df_sim['df_cash'] + df_sim['amm_cash'] + df_sim['protocol_earnings_vault']).values
        res[:,i,3] = amm_balance
        res[:,i,0] += amm_balance
        res[:,i,4] += amm_balance
        print(f"Loaded sim {sim_id} ({i+1}/{len(sim_ids)})")
    
    # daily data
    # res = res[::(60*12),:,:] 
    # monthly volume and earnings
    res[30:,:,0] = res[30:,:,0] - res[:-30,:,0] # gross earnings, coll ccy
    res[30:,:,1] = res[30:,:,1] - res[:-30,:,1] # volume, quote ccy
    res[30:,:,4] = res[30:,:,4] - res[:-30,:,4] # protocol earnings, coll ccy
    # drop first n days
    drop = 30
    res = res[drop:,:,:]
    time_df = pd.to_datetime(df_sim['time'], format='%y-%m-%d %H:%M', utc=True).values #[::(60*12)]
    time_df = time_df[drop:]
    
    print("---Stats for last 30 days---\n\n")
    
    print(pd.DataFrame(
        {"Volume": res[:,:,1].flatten(), 
         "Gross Earnings": (res[:,:,0] * res[:,:,2]).flatten(), 
         "Protocol Earnings": (res[:,:,4] * res[:,:,2]).flatten(), 
         "Gross Earnings per Volume": ((res[:,:,0] * res[:,:,2]) / res[:,:,1]).flatten(), 
         "AMM Balance": (res[:,:,3] * res[:,:,2]).flatten()}
    ).describe())
    
    fig, axs = plt.subplots(5, 1, sharex=True)
    
    axs[0].plot(time_df, 1e-6 * np.median(res[:,:,1], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[0].plot(time_df, 1e-6 * np.quantile(res[:,:,1], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[0].set_ylabel("USD (mm)")
    axs[0].set_title("Monthly Traded Volume")
    
    axs[1].plot(time_df, 1e-6 * np.median(res[:,:,0] * res[:,:,2], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[1].plot(time_df, 1e-6 * np.quantile(res[:,:,0] * res[:,:,2], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[1].set_ylabel("USD (mm)")
    axs[1].set_title("Monthly Gross Earnings")
    
    axs[2].plot(time_df, 100 * np.median(res[:,:,0] * res[:,:,2] / res[:,:,1], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 * np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 *np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 * np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[2].plot(time_df, 100 *np.quantile(res[:,:,0] * res[:,:,2] / res[:,:,1], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[2].set_ylabel("%")
    axs[2].set_title("Monthly Gross Earnings per Volume")
    
    axs[3].plot(time_df, 1e-6 * np.median(res[:,:,4] * res[:,:,2], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[3].plot(time_df, 1e-6 * np.quantile(res[:,:,4] * res[:,:,2], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[3].plot(time_df, 1e-6 * np.quantile(res[:,:,4] * res[:,:,2], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[3].plot(time_df, 1e-6 * np.quantile(res[:,:,4] * res[:,:,2], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[3].plot(time_df, 1e-6 * np.quantile(res[:,:,4] * res[:,:,2], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[3].set_ylabel("USD (mm)")
    axs[3].set_title("Monthly Protocol Earnings")
    
    axs[4].plot(time_df, 100 * np.median(res[:,:,4] * res[:,:,2] / res[:,:,1], axis=-1), "-", color="#664ADF", linewidth=1.5, alpha=1)
    axs[4].plot(time_df, 100 * np.quantile(res[:,:,4] * res[:,:,2] / res[:,:,1], 0.25, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[4].plot(time_df, 100 *np.quantile(res[:,:,4] * res[:,:,2] / res[:,:,1], 0.75, axis=-1), "-", color="#FFDA9F", linewidth=1.5, alpha=1)
    axs[4].plot(time_df, 100 * np.quantile(res[:,:,4] * res[:,:,2] / res[:,:,1], 0.05, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[4].plot(time_df, 100 *np.quantile(res[:,:,4] * res[:,:,2] / res[:,:,1], 0.95, axis=-1), "-", color="#FFBDBD", linewidth=1.5, alpha=1)
    axs[4].set_ylabel("%")
    axs[4].set_title("Monthly Protocol Earnings per Volume")
    
    fig.autofmt_xdate()
    fig.set_size_inches(10, 15)
    plt.xlabel("Time")
    plt.show()
    
if __name__ == "__main__":
    # main()
    # plot_insurance_quanto()
    run_id = 8762580057492
    plot_realized_revenue_stats(run_id, 
                   ["perp_margin", "perp_K2", "perp_L1", "perp_volume_qc", "amm_cash", "pool_margin", "staker_cash", "protocol_earnings_vault", "idx_px", "idx_s3", "df_cash"], 
                   nrows=None, ylab=[f"{QUOTE} (mm)", f"{COLL} (mm)"], legend=["Traded Volume", "Realized Revenue"])
    
