#%% import and read
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.pyplot import cm
from simulation import load_perp_params
import seaborn as sns


sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


COLL = 'MATIC'
QUOTE = 'USD'

FILENAME = 'results/res119-20-AVAXUSDMATIC_2022919-9035130173671625568.csv'

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

    # print("\nPremium summary (in bps):")
    # print(1e4 * df[['mark_price_rel', 'avg_long_slip', 'avg_short_slip', '100klots_long_slip', '100klots_short_slip']].describe())
    q = [0.01, 0.05, 0.10, 0.20, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]

    print("\nFunding rate accrued over 8h summary:")
    x = df['funding_rate'].rolling(8 * 60).sum().abs().values

    x = np.nanquantile(df['funding_rate'].rolling(8 * 60).sum().abs().values, q)
    v = np.quantile(x, q)
    print("\n".join([f"{100*_q:.1f}%: {1e4 *_v: .4f} bps" for _q, _v in zip(q,v)]))
    print(f"mean +- stddev: {np.mean(x)} +- {np.std(x)}")


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
    # plot_perp_slippage(fig, df)
    plot_perp_price(fig, df)
    plt.show()

if __name__ == "__main__":
    main()
    