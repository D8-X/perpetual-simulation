#%% import and read
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.pyplot import cm
from simulation import load_perp_params

COLL = 'MATIC'
QUOTE = 'USD'

FILENAME = 'results/res135-20-AVAXUSDMATIC_2022919-7453644673771615700.csv'

perpsymbol = re.search("-[A-Z]+_", FILENAME).group(0)[1:-1]
INDEX = perpsymbol[:(len(perpsymbol) - len(QUOTE + COLL))]

def main():
    
    file = FILENAME if FILENAME[-3:] == 'csv' else FILENAME + '.csv'
    df = pd.read_csv(file)
    
    df = df.iloc[int(df.shape[0] * 2/3):]

    df['mid_price_rel'] = (df['mid_price'] - df['idx_px']) / df['idx_px']
    df['mark_price_rel'] = (df['mark_price'] - df['idx_px']) / df['idx_px']
    df['cex_price_rel'] = df['cex_px'] / df['idx_px'] - 1
    # df['max_long_slip'] = (df['max_long_price'] - df['mid_price']) / df['mid_price']
    # df['avg_long_slip'] = (df['avg_long_price'] - df['mid_price']) / df['mid_price']
    # df['min_long_slip'] = (df['min_long_price'] - df['mid_price']) / df['mid_price']
    # df['max_short_slip'] = (df['max_short_price'] - df['mid_price']) / df['mid_price']
    # df['avg_short_slip'] = (df['avg_short_price'] - df['mid_price']) / df['mid_price']
    # df['min_short_slip'] = (df['min_short_price'] - df['mid_price']) / df['mid_price']
    # some sims already have this info
    # if not '100klots_long_slip' in df.columns:
    #     df['100klots_long_slip'] = (df['100klots_long_price'] - df['mid_price']) / df['mid_price']
    #     df['100klots_short_slip'] = (df['100klots_short_price'] - df['mid_price']) / df['mid_price']
    #     df['10k_long_slip'] = (df['10k_long_price'] - df['mid_price']) / df['mid_price']
    #     df['10k_short_slip'] = (df['10k_short_price'] - df['mid_price']) / df['mid_price']


    price_columns = [x for x in df.columns if re.search('^perp_pi', x)]
    if len(price_columns) > 0:
        # transform price into price impact
        for col in price_columns:
            df[col] = df[col] / df['mid_price'] - 1
    

    df['datetime'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M', utc=True)
    df['timestamp'] = df['datetime'].apply(datetime.datetime.timestamp)

    from_date = df['timestamp'].min()
    to_date = df['timestamp'].max()

    print(f"Date range: {datetime.datetime.fromtimestamp(from_date)} -- {datetime.datetime.fromtimestamp(to_date)}")

    # print("\nPremium summary (in bps):")
    # print(1e4 * df[['mark_price_rel', 'avg_long_slip', 'avg_short_slip', '100klots_long_slip', '100klots_short_slip']].describe())
    q = [0.01, 0.05, 0.10, 0.20, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    
    
    # pi_col = [c for c in df.columns if re.search(f"^{INDEX}{QUOTE}_price_impact", c)]
    pi_col = [x for x in df.columns if re.search('^perp_price', x)]
    dg = pd.DataFrame(columns=["exchange", "daterange", "long/short", f"trade size({INDEX}", "mean(bps)", "min", "max", "variance", "nobs"])
    exchange = "D8X"
    daterange = str([datetime.date.fromtimestamp(from_date), datetime.date.fromtimestamp(to_date)])
    if len(pi_col) > 0:
        for col in pi_col: 
        # for col in [pi_col[0], pi_col[-1]]: 
            x = 1e4 * np.abs(df[col].values)
            v = np.quantile(x, q)
            pos = float(re.sub("^perp_price_", "", col))
            print(f"\nPrice impact summary: {col} {INDEX}")
            print(f"min/max: {np.nanmin(x)} / {np.nanmax(x)} (bps)")
            print("\n".join([f"{100*_q:.1f}%: {_v: .4f} bps" for _q, _v in zip(q,v)]))
            print(f"mean +- stddev: {np.nanmean(x)} +- {np.nanstd(x)} (bps)")
            dg.loc[dg.shape[0]] = {
                "exchange": exchange,
                "daterange": daterange,
                "long/short": "long" if pos > 0 else "short",
                f"trade size({INDEX}": pos,
                "mean(bps)": np.abs(np.nanmean(x)),
                "min": np.abs(np.nanmin(x)),
                "max": np.abs(np.nanmax(x)),
                "variance": np.abs(np.nanvar(x)),
                "nobs": x.shape[0],
            }
        # dg["exchange"] = "D8X"
        # dg["daterange"] = str([datetime.date.fromtimestamp(from_date), datetime.date.fromtimestamp(to_date)])
        print(dg.head())
        print(dg.tail())
        slip_file = re.sub(".csv", "-slippage_stats.csv", file)
        dg.to_csv(slip_file, index=False)
    # x = (df['100klots_long_slip'].abs() + df['100klots_short_slip'].abs()).values
    # q = [0.01, 0.05, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # v = np.quantile(x, q)
    # print("\n".join([f"{100*_q:.1f}%: {1e4 *_v: .4f} bps" for _q, _v in zip(q,v)]))
    # print(f"mean +- stddev: {np.nanmean(1e4 * x)} +- {np.nanstd(1e4 * x)} (bps)")
    
    # print("\nOpen+Close 100K lots, 24 hour rolling average:")
    # x = (df['100klots_long_slip'].rolling(24 * 60).mean().abs().fillna(method='bfill') + df['100klots_short_slip'].rolling(8 * 60).mean().abs().fillna(method='bfill')).values
    
    # v = np.quantile(x, q)
    # print("\n".join([f"{100*_q:.1f}%: {1e4 *_v: .4f} bps" for _q, _v in zip(q,v)]))
    # print(f"mean +- stddev: {np.nanmean(1e4 * x)} +- {np.nanstd(1e4 * x)} (bps)")
    
    
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

    ax.grid(linestyle='--', linewidth=1)
    ax.set(xlabel="Time", ylabel=COLL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_perp_funds(ax, df):
    # ax.plot(df['datetime'], df['perp_amm_target'], 'k:', label='AMM target', alpha=0.5)
    # ax.plot(df['datetime'], df['perp_amm_target_baseline'], 'k:', alpha=0.5, label='AMM target')
    ax.plot(df['datetime'], df['perp_amm_target_baseline'] + df['perp_pricing_staked_cash'] + df['perp_margin'], 'k:', alpha=0.5, label='Target capital')
    ax.plot(df['datetime'], df['perp_amm_target_stress'], 'k:', alpha=0.5)
    ax.plot(df['datetime'], df['perp_amm_cash'], 'b-', label='AMM pool')

    ax.plot(df['datetime'], df['perp_pricing_staked_cash'], 'y-', label='External LP')
    
    ax.plot(df['datetime'], df['perp_amm_cash'] + df['perp_pricing_staked_cash'] + df['perp_margin'], 'g-', label='Total capital')

    ax.plot(df['datetime'], df['perp_margin'], '-', color="purple", label='AMM margin')

    ax.set(xlabel="Time", ylabel=COLL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))

    # ax.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(linestyle='--', linewidth=1)

def plot_price_premia(ax, df):
    # if df['cex_price_rel'].abs().max() > 0:
    #     ax.plot(df['datetime'], df['cex_price_rel']*1e4,  'k-', label='CEX mark-premium')
    # ax.plot(df['datetime'], df['mid_price_rel']*1e4, 'r:', alpha=0.5,  label='DEX mid-premium')
    # ax.plot(df['datetime'], df['mark_price_rel']*1e4,  'r-', label='DEX mark-premim')
    ax.plot(df['datetime'], df['funding_rate']*1e4 * 8 * 60, 'g-', label='DEX funding rate') # * 8 * 60 so it's an 8h rate
    ax.plot(df['datetime'], df['funding_rate'].rolling(8 * 60).sum()*1e4, 'g:', alpha=0.5)

    ax.set(xlabel="Time", ylabel="Basis Points")
    ax.grid(linestyle='--', linewidth=1)
    ax.legend()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))

def plot_perp_slippage(ax, df):
    # perp_params = load_perp_params(f"{INDEX}{QUOTE}", COLL)
    # lot_size = perp_params['fLotSizeBC']
    
    # # min position
    # ax.plot(df['datetime'], df['min_short_slip']*1e4, '-', alpha=0.5, color='green')
    # ax.plot(df['datetime'], df['min_long_slip']*1e4, '-', alpha=0.5, color='green', label=f"Min long/short")
    
    ## max position
    # ax.plot(df['datetime'], df['max_long_slip']*1e4, '-', alpha=0.5, color='red', label=f"Max long/short")
    # ax.plot(df['datetime'], df['max_short_slip']*1e4,  '-', alpha=0.5, color='red')
    
    # # average position
    # ax.plot(df['datetime'], df['avg_long_slip']*1e4, '-', alpha=0.3, color='black', label=f"EMA")
    # ax.plot(df['datetime'], df['avg_short_slip']*1e4, '-', alpha=0.3, color='black')
    
    # # 10,000 USD worth of index, which costs around $500 at 20x leverage (typical size)
    # ax.plot(df['datetime'], df['10k_long_slip']*1e4, ':', alpha=0.4, color='orange')
    # ax.plot(df['datetime'], df['10k_long_slip'].rolling(24 * 60).mean()*1e4, '-', alpha=0.6, color='blue', label=f"10,000 {QUOTE}")
    # ax.plot(df['datetime'], df['10k_short_slip']*1e4, '-', alpha=0.4, color='orange')
    # ax.plot(df['datetime'], df['10k_short_slip'].rolling(24 * 60).mean()*1e4, '-', alpha=0.6, color='blue')
    
    # # '100 ETH' : 24 hour rolling mean and actual in the background
    # ax.plot(df['datetime'], df['100klots_long_slip']*1e4, ':', alpha=0.4, color='purple')
    # ax.plot(df['datetime'], df['100klots_long_slip'].rolling(24 * 60).mean()*1e4, '-', alpha=0.6, color='purple', label=f"{100_000 * lot_size:.0f} {INDEX}")
    # ax.plot(df['datetime'], df['100klots_short_slip']*1e4, ':', alpha=0.4, color='purple')
    # ax.plot(df['datetime'], df['100klots_short_slip'].rolling(24 * 60).mean()*1e4, '-', alpha=0.6, color='purple')
    
    slippage = [x for x in df.columns if re.search('^perp_price', x)]
    slippage = [slippage[0], slippage[5], slippage[10], slippage[-11], slippage[-6], slippage[-1]]
    # print(perp_pnls)
    color = iter(cm.rainbow(np.linspace(0, 1, len(slippage))))
    for i, p in enumerate(slippage):
        if i % 4 != 0:
            next
        c = next(color)
        # print(p)
        ax.plot(df['datetime'], 1e4 * df[p], c=c, label=re.sub("perp_price_", "", p))
    # ax.legend()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set(xlabel="Time", ylabel="Price impact (from mid-price, bps)")

    ax.legend()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(linestyle='--', linewidth=1)

def plot_pos_sizes(ax, df):
    ax.plot(df['datetime'], df['max_long_trade'], 'r', label="max long")
    ax.plot(df['datetime'], df['max_short_trade'], 'g', label="max short")
    ax.plot(df['datetime'], df['current_trader_exposure_EMA'], 'k:', label="EMA")
    ax.plot(df['datetime'], -df['current_trader_exposure_EMA'], 'k:')

    #axs[1,0].set_xticks(df['num_trades'][mask])
    ax.legend()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))


def plot_prices(ax, df):
    ax.plot(df['datetime'], df['idx_px'], 'k-', label="Oracle price")
    ax.plot(df['datetime'], df['mark_price'], 'r:', label="Mark price")
    ax.set(xlabel="Time", ylabel=f"{INDEX}/{QUOTE}")
    ax.grid(linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
def plot_pnl(ax, df):
    # ax.plot(df['datetime'], df['trader_pnl_cc'], 'k-', label="")
    perp_pnls = [x for x in df.columns if re.search('_pnl_cc', x) and not re.search('^perp', x)]
    # print(perp_pnls)
    color = iter(cm.rainbow(np.linspace(0, 1, len(perp_pnls))))
    for p in perp_pnls:
        c = next(color)
        # print(p)
        ax.plot(df['datetime'], df[p], c=c, label=p)
    # ax.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.set(xlabel="Time", ylabel=f"{COLL}")
    ax.grid(linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))
    
def plot_earnings(ax, df):
    ax.plot(df['datetime'], (df['staker_cash']), 'b-', label='Lp')
    ax.plot(df['datetime'], df['protocol_earnings_vault'], 'k-', label="Protocol") 
    ax.plot(df['datetime'], (df['df_cash']- df['df_target']), 'k:', label='DF excess')
    ax.plot(df['datetime'], df['liquidator_earnings_vault'], 'g-', label='Liquidator')
    ax.set(xlabel="Time", ylabel=f"{COLL}")
    ax.grid(linestyle='--', linewidth=1)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))
    
def plot_analysis(df, file=None):
    print(f"Index: {INDEX}, Quote: {QUOTE}, Collateral: {COLL}")
    
    ## First plot
    
    fig, axs = plt.subplots(3, 2, sharex=True)
    # mask = df["timestamp"] % (60 * 60) == 0
    mask = df["timestamp"] % 60 == 0
   
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
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H-%M'))
    
  
    plot_earnings(axs[1,0], df[mask])
    
    plot_pnl(axs[0,1], df[mask])
    
    plot_amm_funds(axs[1,1], df[mask])

    fig = plt.gcf()
    fig.autofmt_xdate()
    fig.set_size_inches((12, 11), forward=False)
    if file is not None:
        fig.savefig(file[:-4] + '-2.png', dpi=500)

    plt.show()
# %%

    # %%

if __name__ == "__main__":
    main()
    