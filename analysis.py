#%% import and read
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.pyplot import cm
from simulation import load_perp_params

COLL = 'MATIC'
QUOTE = 'USD'

FILENAME = 'results/res107-0-ETHUSDMATIC_20221020--8985315591111932502.csv'

perpsymbol = re.search("-[A-Z]+_", FILENAME).group(0)[1:-1] 
INDEX = perpsymbol[:(len(perpsymbol) - len(QUOTE + COLL))]

def main():
    
    file = FILENAME if FILENAME[-3:] == 'csv' else FILENAME + '.csv'
    df = pd.read_csv(file)

    df['mid_price_rel'] = (df['mid_price'] - df['idx_px']) / df['idx_px']
    df['mark_price_rel'] = (df['mark_price'] - df['idx_px']) / df['idx_px']
    df['cex_price_rel'] = df['cex_px'] / df['idx_px'] - 1
    df['max_long_slip'] = (df['max_long_price'] - df['mid_price']) / df['mid_price']
    df['avg_long_slip'] = (df['avg_long_price'] - df['mid_price']) / df['mid_price']
    df['min_long_slip'] = (df['min_long_price'] - df['mid_price']) / df['mid_price']
    df['max_short_slip'] = (df['max_short_price'] - df['mid_price']) / df['mid_price']
    df['avg_short_slip'] = (df['avg_short_price'] - df['mid_price']) / df['mid_price']
    df['min_short_slip'] = (df['min_short_price'] - df['mid_price']) / df['mid_price']
    df['100klots_long_slip'] = (df['100klots_long_price'] - df['mid_price']) / df['mid_price']
    df['100klots_short_slip'] = (df['100klots_short_price'] - df['mid_price']) / df['mid_price']
    df['10k_long_slip'] = (df['10k_long_price'] - df['mid_price']) / df['mid_price']
    df['10k_short_slip'] = (df['10k_short_price'] - df['mid_price']) / df['mid_price']


    

    df['datetime'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M', utc=True)
    df['timestamp'] = df['datetime'].apply(datetime.timestamp)

    from_date = df['timestamp'].min()
    to_date = df['timestamp'].max()

    print(f"Date range: {datetime.fromtimestamp(from_date)} -- {datetime.fromtimestamp(to_date)}")

    print("\nPremium summary (in bps):")
    print(1e4 * df[['mark_price_rel', 'avg_long_slip', 'avg_short_slip', '100klots_long_slip', '100klots_short_slip']].describe())

    print("\nOpen+Close 100K lots summary:")
    x = (df['100klots_long_slip'].abs() + df['100klots_short_slip'].abs()).values
    q = [0.01, 0.05, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    v = np.quantile(x, q)
    print("\n".join([f"{100*_q:.1f}%: {1e4 *_v: .4f} bps" for _q, _v in zip(q,v)]))
    print(f"mean +- stddev: {np.mean(1e4 * x)} +- {np.std(1e4 * x)} (bps)")
    
    
    # fig, ax = plt.subplots(2, 1, sharex=True)    
    
    # n, bins, patches = ax[0].hist(1e4 * x, bins=200, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=True)
    # n = n.astype('int') # it MUST be integer
    # ax[0].title('Open and Close 100 ETH', fontsize=12)
    # ax[0].xlabel('Total slippage (bps)', fontsize=10)
    # ax[0].ylabel('Frequency', fontsize=10)
    # plt.show()



    print("\nFunding rate accrued over 8h summary:")
    x = df['funding_rate'].rolling(8 * 60).sum().abs().values
    # n, bins, patches = ax[1].hist(1e4 * x, bins=200, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=True)
    # n = n.astype('int') # it MUST be integer
    # ax[1].title('Open and Close 100 ETH', fontsize=12)
    # ax[1].xlabel('Total slippage (bps)', fontsize=10)
    # ax[1].ylabel('Frequency', fontsize=10)
    # plt.show()
    x = np.nanquantile(df['funding_rate'].rolling(8 * 60).sum().abs().values, q)
    v = np.quantile(x, q)
    print("\n".join([f"{100*_q:.1f}%: {1e4 *_v: .4f} bps" for _q, _v in zip(q,v)]))
    print(f"mean +- stddev: {np.mean(x)} +- {np.std(x)}")


    plot_analysis(df, file)


def plot_amm_funds(ax, df):
    ax.plot(df['datetime'], df['df_target'], 'r:', label='DF target', alpha=0.5)
    ax.plot(df['datetime'], df['df_cash'], 'r-', label='DF')

    ax.plot(df['datetime'], df['pool_margin'], '-', color="purple", label='AMM margin', alpha=0.5)
    ax.plot(df['datetime'], df['amm_cash'], 'b-', label='AMM fund')

    ax.plot(df['datetime'], df['pricing_staked_cash'], ':', color='orange', alpha=0.5)
    ax.plot(df['datetime'], df['cash_staked'], 'y-', label='External LP')
    
    ax.plot(df['datetime'], df['df_cash'] + df['amm_cash'] + df['pool_margin'], 'g-', label='Protocol')

    ax.set(xlabel="Time", ylabel=COLL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))

    ax.legend()
    ax.grid(linestyle='--', linewidth=1)

def plot_perp_funds(ax, df):
    # ax.plot(df['datetime'], df['perp_amm_target'], 'k:', label='AMM target', alpha=0.5)
    ax.plot(df['datetime'], df['perp_amm_target_baseline'], 'k:', alpha=0.5, label='AMM target')
    ax.plot(df['datetime'], df['perp_amm_target_stress'], 'k:', alpha=0.5)
    ax.plot(df['datetime'], df['perp_amm_cash'], 'b-', label='AMM fund')

    ax.plot(df['datetime'], df['perp_pricing_staked_cash'], 'y-', label='External LP')
    
    ax.plot(df['datetime'], df['perp_amm_cash'] + df['perp_pricing_staked_cash'], 'g-', label='Perp capital')

    ax.plot(df['datetime'], df['perp_margin'], '-', color="purple", label='Perp AMM margin')

    ax.set(xlabel="Time", ylabel=COLL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))

    ax.legend()
    ax.grid(linestyle='--', linewidth=1)

def plot_price_premia(ax, df):
    if df['cex_price_rel'].abs().max() > 0:
        ax.plot(df['datetime'], df['cex_price_rel']*1e4,  'k-', label='cex')
    ax.plot(df['datetime'], df['mid_price_rel']*1e4, 'r-', label='Mid')
    ax.plot(df['datetime'], df['mark_price_rel']*1e4,  'y:', label='Mark')

    ax.set(xlabel="Time", ylabel="premium over index (bps)")
    ax.grid(linestyle='--', linewidth=1)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))

def plot_perp_slippage(ax, df):
    perp_params = load_perp_params(f"{INDEX}{QUOTE}", COLL)
    # ax.plot(df['datetime'], df['max_long_slip']*1e4, '-', alpha=0.5, color='red', label=f"Max long/short")
    ax.plot(df['datetime'], df['100klots_long_slip']*1e4, '-', alpha=0.5, color='purple', label=f"{100_000 * perp_params['fLotSizeBC']} {INDEX} long/short")
    ax.plot(df['datetime'], df['10k_long_slip']*1e4, '-', alpha=0.5, color='blue', label=f"10,000 USD long/short")
    # ax.plot(df['datetime'], df['avg_long_slip']*1e4, '-', alpha=0.5, color='yellow', label=f"Avg long/short")
    ax.plot(df['datetime'], df['min_long_slip']*1e4, '-', alpha=0.5, color='green', label=f"Min long/short")

    # ax.plot(df['datetime'], df['max_short_slip']*1e4,  '-', alpha=0.5, color='red')
    ax.plot(df['datetime'], df['100klots_short_slip']*1e4, '-', alpha=0.5, color='purple')
    ax.plot(df['datetime'], df['10k_short_slip']*1e4, '-', alpha=0.5, color='blue')
    # ax.plot(df['datetime'], df['avg_short_slip']*1e4, '-', alpha=0.5, color='yellow')
    ax.plot(df['datetime'], df['min_short_slip']*1e4, '-', alpha=0.5, color='green')

    ax.set(xlabel="Time", ylabel="slippage from mid price (bps)")

    ax.legend()
    ax.grid(linestyle='--', linewidth=1)

def plot_pos_sizes(ax, df):
    ax.plot(df['datetime'], df['max_long_trade'], 'r', label="max long")
    ax.plot(df['datetime'], df['max_short_trade'], 'g', label="max short")
    ax.plot(df['datetime'], df['current_trader_exposure_EMA'], 'k:', label="EMA (long)")
    ax.plot(df['datetime'], -df['current_trader_exposure_EMA'], 'k:', label="EMA (short)")

    #axs[1,0].set_xticks(df['num_trades'][mask])
    ax.legend()
    ax.set(xlabel="Time", ylabel=f"position size ({INDEX})")

    ax.grid(linestyle='--', linewidth=1)
    #axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))

def plot_num_traders(ax, df):
    ax.plot(df['datetime'], df['num_noise_traders'], 'kd', label='Noise')
    ax.plot(df['datetime'], df['num_arb_traders'], 'g+', label='Arb')
    ax.plot(df['datetime'], df['num_momentum_traders'], 'bx', label='Momentum')
    ax.plot(df['datetime'], df['num_bankrupt_traders'], 'r:', label='Bankrupt')
    ax.plot(df['datetime'], df['num_noise_traders'] + df['num_arb_traders'] + df['num_momentum_traders'], 'y', label='Total Active')
    ax.set(xlabel="Time", ylabel="Number of traders")
    ax.legend()
    ax.grid(linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))


def plot_prices(ax, df):
    ax.plot(df['datetime'], df['idx_px'], 'k-', label="index price")
    ax.plot(df['datetime'], df['mark_price'], 'r:', label="mark price")
    ax.set(xlabel="Time", ylabel=f"{INDEX}/{QUOTE}")
    ax.grid(linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    
    
def plot_pnl(ax, df):
    # ax.plot(df['datetime'], df['trader_pnl_cc'], 'k-', label="")
    perp_pnls = [x for x in df.columns if re.search('_pnl_cc', x) and not re.search('^perp', x)]
    # print(perp_pnls)
    color = iter(cm.rainbow(np.linspace(0, 1, len(perp_pnls))))
    for p in perp_pnls:
        c = next(color)
        print(p)
        ax.plot(df['datetime'], df[p], c=c, label=p)
    ax.legend()
    ax.set(xlabel="Time", ylabel=f"{COLL}")
    ax.grid(linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    
def plot_earnings(ax, df):
    ax.plot(df['datetime'], (df['staker_cash']), 'b-', label='Lp')
    ax.plot(df['datetime'], df['protocol_earnings_vault'], 'k-', label="Protocol") 
    ax.plot(df['datetime'], (df['df_cash']- df['df_target']), 'k:', label='DF excess')
    ax.plot(df['datetime'], df['liquidator_earnings_vault'], 'g-', label='Liquidator')
    ax.set(xlabel="Time", ylabel=f"{COLL}")
    ax.grid(linestyle='--', linewidth=1)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    
def plot_analysis(df, file=None):
    print(f"Index: {INDEX}, Quote: {QUOTE}, Collateral: {COLL}")
    
    ## First plot
    
    fig, axs = plt.subplots(3, 2, sharex=True)
    mask = df["timestamp"] % (60 * 60) == 0
   
    plot_price_premia(axs[0,0], df[mask])
    
    plot_pos_sizes(axs[1,0], df[mask])

    plot_prices(axs[0,1], df[mask])

    # plot_amm_funds(axs[2,1], df[mask])
    plot_num_traders(axs[2,1], df[mask])

    plot_perp_slippage(axs[2,0], df[mask])

    plot_perp_funds(axs[1,1], df[mask])
   
    fig = plt.gcf()
    fig.set_size_inches((12, 11), forward=False)
    fig.autofmt_xdate()
    if file is not None:
        fig.savefig(file[:-4] + '-1.png', dpi=500)


    #### Second plot ####
    fig, axs = plt.subplots(2, 2, sharex=True)
    # quote ccy
    # axs[0,0].plot(df['datetime'][mask], df['protocol_earnings_vault'][mask] * df['idx_s3'][mask], 'k-', label="Protocol") 
    # axs[0,0].plot(df['datetime'][mask], (df['df_cash'][mask]- df['df_target'][mask]) * df['idx_s3'][mask], 'k:', label='DF excess')
    # axs[0,0].plot(df['datetime'][mask], (df['staker_cash'][mask]) * df['idx_s3'][mask], 'b-', label='Liq provider')
    # axs[0,0].plot(df['datetime'][mask], df['liquidator_earnings_vault'][mask] * df['idx_s3'][mask], 'g-', label='Liquidator')
    # axs[0,0].set(xlabel="Time", ylabel=f"{QUOTE}")
    # axs[0,0].grid(linestyle='--', linewidth=1)
    # axs[0,0].legend()
    # axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    axs[0,0].plot(df['datetime'][mask], 100 * df['lp_apy'][mask], 'k-', label='LP APY')
    axs[0,0].set(xlabel="Time", ylabel=f"%")
    axs[0,0].grid(linestyle='--', linewidth=1)
    axs[0,0].legend()
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    
  
    plot_earnings(axs[1,0], df[mask])
    
    plot_pnl(axs[0,1], df[mask])
    
    plot_amm_funds(axs[1,1], df[mask])

    fig = plt.gcf()
    fig.set_size_inches((12, 11), forward=False)
    if file is not None:
        fig.savefig(file[:-4] + '-2.png', dpi=500)

    plt.show()
# %%

    # %%

if __name__ == "__main__":
    main()
    