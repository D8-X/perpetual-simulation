import numpy as np
import matplotlib.pyplot as plt
import pricing_benchmark
import pandas as pd
from scipy.stats import norm

def plot_price_curve(
    M1, M2, M3, K2, L1, 
    S2, 
    sig2, 
    r=0, 
    S3=None, 
    sig3=None, 
    rho23=None, 
    min_spread=0.0001, 
    incentive_spread=0, 
    k_step=1e-2):

    # check args
    if not S3:
        S3 = S2
    if not sig3:
        sig3 = sig2
    if not rho23:
        rho23 = 1
    

    PCT_MAX = 0.05
    PCT_MIN = -0.05
    
    def get_price(k):
        return pricing_benchmark.calculate_perp_priceV4(
            K2, k, L1, S2, S3, sig2, sig3, rho23, r, M1, M2, M3, min_spread, incentive_spread)
    
    mid_price = 0.5 * (get_price(-k_step) + get_price(k_step))
    #k_star = pricing_benchmark.get_Kstar(S2, S3, M1, M2, M3, K2, L1, sig2, sig3, rho23, r)
    left_end = -(L1 + M1) / S2
    right_end = M2 - K2
    k_star = pricing_benchmark.get_Kstar(S2, S3, M1, M2, M3, K2, L1, sig2, sig3, rho23, r)
    k_range = [0]
    price_range = [mid_price]
    pct_range = [0]

    # long positions
    cur_pct = 0
    cur_k = k_step
    while cur_pct <= PCT_MAX or cur_k <= 1.5*right_end:
        cur_price = get_price(cur_k)
        cur_pct = cur_price / mid_price - 1

        price_range.append(cur_price)
        k_range.append(cur_k)
        pct_range.append(cur_pct)
        
        cur_k += k_step
    
    # short positions
    cur_pct = 0
    cur_k = -k_step
    while cur_pct >= PCT_MIN or cur_k >= 1.5*left_end:
        cur_price = get_price(cur_k)
        cur_pct = cur_price / mid_price - 1

        price_range.append(cur_price)
        k_range.append(cur_k)
        pct_range.append(cur_pct)

        cur_k -= k_step
    k_range = np.array(k_range)
    price_range = np.array(price_range)
    pct_range = np.array(pct_range)

    sorted_idx = np.argsort(k_range)
    k_range = k_range[sorted_idx]
    price_range = price_range[sorted_idx]
    pct_range = 100 * pct_range[sorted_idx]

    s2_loc = np.argmin((price_range - S2) ** 2)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(k_range, price_range, label='price')
    axs[0].set(xlabel="k", ylabel="$")
    axs[0].axvline(x=0, color='g', linestyle='-', label="k=0")
    # axs[0].axvline(x=left_end, color='y', linestyle='-')
    # axs[0].axvline(x=right_end, color='y', linestyle='-')

    axs[0].axvline(x=k_star, color='r', linestyle='-', label="k*")
    axs[0].axhline(y=S2, color='b', linestyle='-', label="index price")
    axs[0].axhline(y=mid_price, color='m', linestyle='-', label="mid price")
    axs[0].grid(linestyle='--', linewidth=1)
    axs[0].legend()

    axs[1].plot(k_range, pct_range, label='deviation from mid-price')
    axs[1].plot(k_range, 100*(price_range/S2 - 1), color='b', label='deviation from index')
    axs[1].set(xlabel="k", ylabel="%")
    axs[1].axvline(x=0, color='g', linestyle='-', label="k=0")
    # axs[1].axvline(x=left_end, color='y', linestyle='-')
    # axs[1].axvline(x=right_end, color='y', linestyle='-')
    axs[1].axhline(y=0, color='m', linestyle='-')

    axs[1].grid(linestyle='--', linewidth=1)
    axs[1].axvline(x=k_star, color='r', linestyle='-', label="k*")
    axs[1].legend()

    plt.show()


def plot_dd_curve(
    M1, M2, M3, K2, L1, 
    S2, 
    sig2, 
    r=0, 
    S3=None, 
    sig3=None, 
    rho23=None, 
    min_spread=0.0001, 
    incentive_spread=0, 
    k_step=1e-2):

    # check args for no-quanto case
    if not S3:
        S3 = S2
    if not sig3:
        sig3 = sig2
    if not rho23:
        rho23 = 1
    

    PD_MAX = 0.1
    DD_MAX = norm.ppf(PD_MAX)
    DD_MIN = -7
    PD_MIN = norm.cdf(DD_MIN)

    left_end = -(L1 + M1) / S2
    right_end = M2 - K2
    k_star = pricing_benchmark.get_Kstar(S2, S3, M1, M2, M3, K2, L1, sig2, sig3, rho23, r)

    def get_dd(k):
        px = pricing_benchmark.calculate_perp_priceV4(
            K2, k, L1, S2, S3, sig2, sig3, rho23, r, M1, M2, M3, min_spread, incentive_spread)
        pd = np.sign(k - k_star) * (px / S2 - 1 - np.sign(k) * min_spread, 0.0005, 1)
        return norm.ppf(pd) if pd > PD_MIN else DD_MIN

    k_range = []
    dd_range = []
    pd_range = []
    
    # long positions
    cur_k = k_step
    cur_dd = -np.Inf
    while cur_k <= 1.5*right_end or cur_dd < DD_MAX:
        cur_dd = get_dd(cur_k)
        cur_pd = norm.cdf(cur_dd)
        
        dd_range.append(cur_dd)
        k_range.append(cur_k)
        pd_range.append(cur_pd)
        
        cur_k += k_step
    
    # short positions
    cur_k = -k_step
    cur_dd = -np.Inf
    while cur_k >= 1.5*left_end or cur_dd < DD_MAX:
        cur_dd = get_dd(cur_k)
        cur_pd = norm.cdf(cur_dd)
        
        dd_range.append(cur_dd)
        k_range.append(cur_k)
        pd_range.append(cur_pd)
        
        cur_k -= k_step

    k_range = np.array(k_range)
    dd_range = np.array(dd_range)
    pd_range = 100 * np.array(pd_range)

    sorted_idx = np.argsort(k_range)
    k_range = k_range[sorted_idx]
    dd_range = dd_range[sorted_idx]
    pd_range = pd_range[sorted_idx]


    fig, axs = plt.subplots(1, 2)
    axs[0].plot(k_range, dd_range, label='dd')
    axs[0].set(xlabel="k")
    axs[0].axvline(x=0, color='g', linestyle='-', label="k=0")
    # axs[0].axvline(x=left_end, color='y', linestyle='-')
    # axs[0].axvline(x=right_end, color='y', linestyle='-')

    axs[0].axvline(x=k_star, color='r', linestyle='-', label="k*")
    axs[0].grid(linestyle='--', linewidth=1)
    axs[0].legend()

    axs[1].plot(k_range, pd_range, label='dd')
    axs[1].set(xlabel="k", ylabel="%")
    axs[1].axvline(x=0, color='g', linestyle='-', label="k=0")
    # axs[1].axvline(x=left_end, color='y', linestyle='-')
    # axs[1].axvline(x=right_end, color='y', linestyle='-')

    axs[1].axvline(x=k_star, color='r', linestyle='-', label="k*")
    axs[1].grid(linestyle='--', linewidth=1)
    axs[1].legend()

    plt.show()

def draw_cash_sample(expected_cash_qc, k=0.5):
    #return np.random.gumbel(loc=reference_cash_cc, scale=reference_cash_cc / 2)
    #return np.random.uniform(low=reference_cash_cc / 2, high=2 * reference_cash_cc)
    scale = expected_cash_qc * k
    min_cash = np.max((expected_cash_qc / 4, 500)) # no less than $500 per trader
    mean_over_min = (expected_cash_qc - min_cash) / scale
    return min_cash + np.random.gamma(mean_over_min, scale)

if __name__ == '__main__':

    # df = pd.read_csv("data/protocol_trade_stats.csv")
    # print(df['type'].unique())
    # my_filter = df['type'] == 'Margin Trade' #'Spot Trade' #~df['type'].isin(('Borrow', 'Lend', 'UnLend', 'Close Borrow'))
    # print(f"Tx count: {df[my_filter]['tx_count'].sum()}")
    # print(df[my_filter].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    # fake_samples = pd.DataFrame([draw_cash_sample(8_000, k=0.24) for _ in range(100)])
    # print(fake_samples.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    # df['avg_volume'][my_filter].plot.hist(bins=100)
    # plt.show()
    # quit()
    M1, M2, M3 = 0, 1.2169841720932666, 0
    K2 = 7.526
    sig2 = 0.01
    L1 = 363869.34905064397
    S2 = 49602.09
    min_spread = 0.00025
    incentive_spread = 0.0005
    k_step = 0.001
    for sig2 in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        plot_price_curve(M1, M2, M3, K2, L1, S2, sig2, min_spread=min_spread, k_step=k_step, incentive_spread=incentive_spread)
        #plot_dd_curve(M1, M2, M3, K2, L1, S2, sig2, min_spread=min_spread, k_step=0.001, incentive_spread=0.0001)
