#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import glob
import json
import multiprocessing
import pprint
import time
from scipy import stats
from arb_trader import ArbTrader
from noise_staker import NoiseStaker
from trader import CollateralCurrency
from noise_trader import NoiseTrader
from momentum_trader import MomentumTrader
from arb_trader import ArbTrader
from amm import AMM
import numpy as np
import pandas as pd 
from datetime import datetime, timezone
import os.path
import pickle
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed


# Perps we can simulate
INDEX = ['BTC', 'ETH', 'XAU', 'BNB', 'CHF', 'GBP', 'SPY', 'MATIC', 'LINK']

# This simulation's quote and collateral currency
QUOTE = 'USD'
COLL = 'MATIC'

# traders that join on day 1
NUM_TRADERS = {
    'BTCUSD': 10,
    'XAUUSD': 0,
    'BNBUSD': 0, 
    'ETHUSD': 10,
    'CHFUSD': 0,
    'GBPUSD': 0,
    'SPYUSD': 0,
    'TSLAUSD': 0,
    'MATICUSD': 10,
    'LINKUSD': 0,
}

# new traders join randomly - up to how many each time? 
# e.g. if we start with N traders and multiplier is M, then there will be in average (M+1) * N traders by the end of the simulation
# to keep same growth rate but reduced target (e.g. to go from 4 to 2 months and keep all other params the same, use divider = 2)
GROWTH_DIVIDER = 2
GROWTH_MULTIPLIER = {
    'BTCUSD': 20 // GROWTH_DIVIDER, 
    'XAUUSD': 20 // GROWTH_DIVIDER, 
    'BNBUSD': 20 // GROWTH_DIVIDER,  
    'ETHUSD': 20 // GROWTH_DIVIDER, 
    'CHFUSD': 20 // GROWTH_DIVIDER, 
    'GBPUSD': 20 // GROWTH_DIVIDER, 
    'SPYUSD': 20 // GROWTH_DIVIDER,
    'MATICUSD': 20 // GROWTH_DIVIDER,
    'LINKUSD': 20 // GROWTH_DIVIDER,
}

# how many arb bots we run for each perp - zero is conservative
BOTS_PER_PERP = {
    'BTCUSD': 0,
    'BNBUSD': 0,
    'XAUUSD': 0,
    'ETHUSD': 0,
    'CHFUSD': 0,
    'GBPUSD': 0,
    'SPYUSD': 0,
    'MATICUSD': 0,
    'LINKUSD': 0,
}

# Chainlink provides data from multiple networks
NETWORK = {
    'BTCUSD': 'BSC_Mainnet', 
    'XAUUSD': 'BSC_Mainnet', 
    'BNBUSD': 'BSC_Mainnet', 
    'ETHUSD': 'BSC_Mainnet',
    'GBPUSD': 'BSC_Mainnet',
    'SPYUSD': 'BSC_Mainnet',
    'CHFUSD': 'Polygon',
    'MATICUSD': 'BSC_Mainnet',
    'LINKUSD': 'BSC_Mainnet',
}

# when a trader goes bankrupt, a new one joins - up to how many?
MAX_TRADER_REPLACEMENTS = 2000

# filter out perps we are not simulating
SYMBOLS = [b + 'USD' for b in INDEX if NUM_TRADERS[b + 'USD'] > 0]

# static simulation hyperparameters, see inside main
SIM_PARAMS = dict()

# how much cash do we give each arbitrage bot?
SIM_PARAMS['usd_per_bot'] = 1_000 

# withdrawing funds automatically - just for educational purposes, we set rate to 0 for now
SIM_PARAMS['protocol_profit_withdrawal_frequency'] = 60 * 24 * 10 # we try to withdraw every n days (seconds)
SIM_PARAMS['protocol_profit_withdrawal_rate'] = 0.0 # % of the excess (if any) we withdraw ([0,1])
SIM_PARAMS['protocol_profit_withdrawal_floor'] = 1_000 # the least amount worth withdrawing in one go (QC)
SIM_PARAMS['protocol_profit_withdrawal_cap'] = 100_000 # the max amount we would withdraw in one go (QC)
SIM_PARAMS['protocol_profit_withdrawal_freeze_period'] = 60 * 24 * 30 # no withdrawals for the first n days (seconds)
SIM_PARAMS['animate_premia'] = False # just for testing, do not use

# probability that an inactive trader opens a position within 24 hours = 1 / num_trades_per_day
SIM_PARAMS['num_trades_per_day'] = 2

# prob that trader is in a tier that pays zero fees. higher is more conservative
SIM_PARAMS['prob_best_tier'] = 0 # not used when simulating zero fees

# number of liquidity providers
SIM_PARAMS['num_stakers'] = 25
# total cash brought in by each LP (randomized)
SIM_PARAMS['cash_per_staker'] = 4_000 # if num=25, then use 20k for 500k, 8k for 200k, 6k for 150k, 4k for 100k, 2k for 50k
# how often do they deposit per month, in average (randomized)
SIM_PARAMS['monthly_stakes'] = 1.5
# they withdraw after this number of months (randomized)
SIM_PARAMS['holding_period_months'] = 1

# this is just for logging on screen
SIM_PARAMS['log_every'] = 2_000


# Parallelize run?
RUN_PARALLEL = False

def main():
    all_runs_t0 = datetime.now()

    # seed for cash samples, random agent trading order, random agent preferences
    seeds = [
        42, 
        31415,
        # 66260,
    ]

    # simulation period
    simulation_horizons = [
        # (datetime(2022, 5, 15, 0, 0, tzinfo=timezone.utc), datetime(2022, 8, 16, 0, 0, tzinfo=timezone.utc)), 
        (datetime(2022, 7, 1, 0, 0, tzinfo=timezone.utc), datetime(2022, 10, 21, 0, 0, tzinfo=timezone.utc)), 
        # (datetime(2022, 6, 18, 0, 0, tzinfo=timezone.utc), datetime(2022, 9, 20, 0, 0, tzinfo=timezone.utc)),
    ]

    # given that a trader is about to open a position, with what probability will it be a long one?
    long_probs = [
        0.36,
        0.51,
        0.64,
        # 0.95,
    ]

    # how much liquidity does the protocol have at launch? in dollars
    initial_investments = [
        # 150_000_000, 
        # 50_000_000,
        # 3 * 250_000,
        len(SYMBOLS) * 350_000,
        # len(SYMBOLS) * 425_000,
    ]

    # exact cash amounts for each trader are randomized, this is an average
    # distribution is fat tailed to the right (i.e. whales exist but are rare)
    usds_per_trader = [
        # 800,
        1_000,
        # 3_000,
    ]

    run_configs = itertools.product(seeds, simulation_horizons, long_probs, initial_investments, usds_per_trader)
    total_hash = hash(run_configs)

    filename = f"./results/sim_run_id_{total_hash}.csv"
   
    def run_single_sim(run_config):
        # copy global params and assign run specific ones
        sim_params = dict(SIM_PARAMS)
        
        sim_params['group_hash'] = total_hash
        sim_params['sim_hash'] = hash(run_config)
        sim_params['seed'] = run_config[0]
        sim_params['from_date'], sim_params['to_date'] = run_config[1]
        sim_params['prob_long'] = run_config[2]
        sim_params['initial_protocol_investment'] = run_config[3]
        sim_params['usd_per_trader'] = run_config[4] 

        # print("\n" + "".join(("-" for _ in range(100))) + "\n")
        # print(f"Run details:")
        # print(f"From {sim_params['from_date']} to {sim_params['to_date']}")
        # print(f"Probability of trader taking a long position: {100 * sim_params['prob_long']:.2f}%")
        # print(f"Protocol initial funds: {to_readable_fiat(sim_params['initial_protocol_investment'])}")
        
        df = simulate(sim_params)

        # revenue summary
        res = report_stats(df, sim_params)
        res = pd.DataFrame(res, index=[0])
        res.to_csv(f"./results/sim_id_{sim_params['sim_hash']}.csv", index=False)
        return res
    
    
    print(f"Run id:\t{total_hash}\n\n")
    
    if RUN_PARALLEL:
        num_cores = multiprocessing.cpu_count() - 1
        results = Parallel(n_jobs=num_cores)(delayed(run_single_sim)(run_config) for run_config in run_configs)
    else:
        results = [run_single_sim(run_config) for run_config in run_configs]
        
    result = pd.concat(results)
    result.to_csv(filename, index=False)
    
    delta_t = (datetime.now() - all_runs_t0).total_seconds() / 60
    print(f"Finished! And it only took {delta_t / 60:.1f} hours...")
    
    return 


def report_stats(amm_state, sim):
    n = amm_state.shape[0]
    days = float((pd.to_datetime(amm_state.at[n-1,'time'], format='%y-%m-%d %H:%M') 
                  - pd.to_datetime(amm_state.at[0,'time'], format='%y-%m-%d %H:%M')).days)
    days = np.max((1, days))
    res = {
        "Run_ID": sim['group_hash'],
        "Sim_ID": sim['sim_hash'],
        "Start": amm_state.at[0,'time'], 
        "End": amm_state.at[n-1, 'time'],
        "Days": days,
        "Prob_Long": sim['prob_long'],
        "Funds_Init_CC": amm_state.at[0, "df_cash"] + amm_state.at[0, 'amm_cash'] + amm_state.at[0, 'pool_margin'], 
        "Funds_Final_CC": amm_state.at[n-1, "df_cash"] + amm_state.at[n-1, 'amm_cash'] + amm_state.at[n-1, 'pool_margin'], 
        "DF_Excess": -amm_state.at[n-1, "df_cash_to_target"], 
        "LP_Init_CC": amm_state.at[0, "staker_cash"], 
        "LP_Final_CC": amm_state.at[n-1, "staker_cash"],
        "LP_Init_QC": amm_state.at[0, "staker_cash"] * amm_state.at[0, "idx_s3"], 
        "LP_Final_QC": amm_state.at[n-1, "staker_cash"] * amm_state.at[n-1, "idx_s3"], 
        "Volume_QC": amm_state.at[n-1, "total_volume_qc"],
        "Volume_CC": amm_state.at[n-1, "total_volume_cc"],
    }
    res["Funds_Init_QC"] = res["Funds_Init_CC"] * amm_state.at[0, "idx_s3"]
    res["Funds_Final_QC"] = res["Funds_Final_CC"] * amm_state.at[n-1, "idx_s3"]
    res["Profit_CC"] = res["Funds_Final_CC"] - res["Funds_Init_CC"]
    res["Profit_QC"] = res["Funds_Final_QC"] - res["Funds_Init_QC"]
    res["Liquid_Profit_QC"] = np.min((res["Profit_QC"], np.max((0, res["DF_Excess"] * amm_state.at[n-1, "idx_s3"]))))
    res["Liquid_Profit_CC"] = np.min((res["Profit_CC"], np.max((0, res["DF_Excess"]))))
    res["Annualized_Profit_QC"] = res["Profit_QC"] * 365 / res["Days"]
    res["Annualized_Profit_CC"] = res["Profit_CC"] * 365 / res["Days"]
    # res["LP_QC_APY"] = (res["LP_Final_QC"] / res["LP_Init_QC"] - 1) * 365 / res["Days"]
    res["LP_CC_APY"] = amm_state.at[n-1, "lp_apy"] #(res["LP_Final_CC"] / res["LP_Init_CC"] - 1) * 365 / res["Days"]
    res["ProfitQC_per_VolumeQC"] = res["Profit_QC"] / res["Volume_QC"]
    res["ProfitCC_per_VolumeCC"] = res["Profit_CC"] / res["Volume_CC"]

    return res


def simulate(sim_input):
    run_t0 = datetime.now()
    sim_state = dict(sim_input)
    sim_state['from_date'], sim_state['to_date'] = sim_state['from_date'].timestamp(), sim_state['to_date'].timestamp()
    np.random.seed(sim_state['seed'])
    print(f"Sim id: {sim_state['sim_hash']}")
    (idx_s2, idx_s3, cex_ts, time_df) = init_index_data(
        sim_state['from_date'], 
        sim_state['to_date'], 
        reload=False, 
        symbol=SYMBOLS, 
        collateral=COLL
    )
    

    total_noise_traders = dict()
    total_momentum_traders = dict()
    total_arb_traders = dict()
    total_num_traders = dict()
    for symbol in SYMBOLS:
        total_noise_traders[symbol] = int(np.ceil(NUM_TRADERS[symbol] * 0.90))
        total_momentum_traders[symbol] = NUM_TRADERS[symbol] - total_noise_traders[symbol]
        total_arb_traders[symbol] = BOTS_PER_PERP[symbol]
        total_num_traders[symbol] = total_noise_traders[symbol] + total_momentum_traders[symbol] + total_arb_traders[symbol]
    
    # amm parameters
    amm_params = load_pool_params(COLL)
    amm_params['ceil_staker_pnl_share'] = 0.75
    amm_params['block_time_sec'] = 60 # not the actual block time, but the highest data frequency
    
    amm = AMM(amm_params, 0)
    state_dict = dict()
    perps = []
    traders = []
    amm_state = np.empty((time_df.shape[0], len(get_amm_state_keys())))
    protocol_cash_cc = sim_state['initial_protocol_investment'] / idx_s3[0]
    # instrument parameters
    perp_params = dict() # dict of dicts
    for symbol in SYMBOLS:
        # reset state of this perp
        sim_state[symbol] = dict()
        sim_state[symbol]['cash_per_trader_qc'] = sim_state['usd_per_trader']

        perp_params[symbol] = load_perp_params(symbol, COLL)
        perp_params[symbol]['perp_amm_cash_cc'] = np.min(
            (perp_params[symbol]['fAMMMinSizeCC'], protocol_cash_cc / len(SYMBOLS)))
        protocol_cash_cc -= perp_params[symbol]['perp_amm_cash_cc']
        
        # simulation state
        sim_state[symbol]['num_replaced_traders'] = 0
        sim_state[symbol]['max_trader_replacements'] = MAX_TRADER_REPLACEMENTS
        # active traders (starts at 0)
        sim_state[symbol]['num_momentum_traders'] = 0
        sim_state[symbol]['num_arb_traders'] = 0
        sim_state[symbol]['num_noise_traders'] = 0
        sim_state[symbol]['num_bankrupt_traders'] = 0    
        sim_state[symbol]['arb_pnl'] = 0    
        sim_state[symbol]['cex_volume_bc'] = 0
        sim_state[symbol]['num_trades'] = 0
        sim_state[symbol]['arb_capital_used'] = 0

        perp_cc_ccy = get_collateral_ccy(symbol, COLL)
        if perp_cc_ccy is CollateralCurrency.BASE:
            s3 = None
        elif perp_cc_ccy is CollateralCurrency.QUOTE:
            s3 = idx_s2[symbol] * 0 + 1
        else:
            s3 = idx_s3
        
        # print(f"\n---{symbol}---\n")
        perp_idx = amm.add_perpetual(
            perp_params[symbol]['perp_amm_cash_cc'], 
            perp_params[symbol]['initial_margin_cash_cc'],
            idx_s2[symbol], 
            s3, 
            perp_params[symbol], 
            perp_cc_ccy,
            min_spread = perp_params[symbol]['fMinimalSpread'], 
            incentive_spread = perp_params[symbol]['fIncentiveSpread'],
            max_position = np.inf,
            verbose = 0,
            symbol=f"{symbol}-{COLL}"
        )
        perps.append(perp_idx)

        # to store simulation results per perp
        state_dict[perp_idx] = np.empty((time_df.shape[0], len(get_perp_state_keys())))
        # create traders
        traders.extend(
            initialize_traders(
                total_noise_traders[symbol], 
                total_momentum_traders[symbol], 
                total_arb_traders[symbol], 
                perp_idx, 
                cex_ts[symbol], 
                amm, 
                perp_cc_ccy,
                num_trades_per_day=sim_state['num_trades_per_day'],
                cash_qc=sim_state[symbol]['cash_per_trader_qc'],
                arb_cash_qc=sim_state['usd_per_bot'],
                prob_best_tier=sim_state['prob_best_tier'],
                prob_long=sim_state['prob_long']
            )
        )
        # record_initial_endowment(traders)

    # the rest goes to DF
    amm.default_fund_cash_cc += protocol_cash_cc

    # initi stakers
    stakers = initialize_stakers(
        amm, 
        sim_state['num_stakers'], 
        sim_state['cash_per_staker'], 
        sim_state['monthly_stakes'], 
        sim_state['holding_period_months'])

    amm.export(f"./data/{COLL}_pool")
    
    
    # # plot trader cash
    # trader_cash_qc = np.array([trader.cash_cc for trader in traders]) * idx_s3[0]
    # fig_cash, ax_cash = plt.subplots()
    # ax_cash.hist(trader_cash_qc, bins='auto', density=True)
    # fig_cash.suptitle(f"Trader cash distribution ({QUOTE})")
    # fig_cash.supxlabel(f"{QUOTE}")

    # plot premium charts/AMM depth
    if sim_state['animate_premia']:
        plt.ion()
        usdrange = np.linspace(-100_000, 100_000, 1_000)    
        fig, axs = plt.subplots(len(SYMBOLS), 1, sharex=True)
        fig.suptitle("Premium over index price (%)")
        fig.supxlabel("Notional value (USD)")
        max_usd = np.max(usdrange)
        min_usd = np.min(usdrange) 


    # loop over time
    print(f"\nBeginning simulation @ {datetime.now().time()}\n")

    trade_count = 0
    liquidation_count = 0
    last_withdrawal = 0
    avg_trading_time, num_trading_measurements = 0, 0
    avg_logging_time, num_logging_measurements = 0, 0
    avg_logging_perp_time, num_logging_perp_measurements = 0, 0
    avg_joining_time, num_joining_measurements = 0, 0
    trader_pnl = 0
    monitoring_period = sim_input['log_every']
    for t in range(time_df.shape[0]):
        # trading happens
        trading_starts = time.time()
        traders_pay_funding(traders)
        liquidation_count += traders_liquidate(traders, sim=sim_state)
        trades_so_far = sim_state[symbol]['num_trades']
        traders_trade(traders, amm, sim_state, cex_ts)
        delta_trades = sim_state[symbol]['num_trades'] - trades_so_far
        trade_count += delta_trades
        trading_ends = time.time()
        # staking
        stakers_stake(stakers)
        
        avg_trading_time += (trading_ends - trading_starts)
        num_trading_measurements += 1

        record_amm_state(t, amm_state, amm, stakers, traders)

        # logging status
        logging_starts = time.time()
        for i, symbol in enumerate(SYMBOLS):
            perp_idx = perps[i]
            perp = amm.get_perpetual(perp_idx)
            logging_perp_starts = time.time()
            record_perp_state(t, state_dict, perp, sim_state[symbol], traders)
            logging_perp_ends = time.time()
            avg_logging_perp_time += logging_perp_ends - logging_perp_starts
            num_logging_perp_measurements += 1
            
            # this perpetual's status
            do_print = (
                t % monitoring_period == 0  or
                amm.get_default_fund_gap_to_target_ratio() < 0.02 or
                np.abs(perp.get_mark_price() / perp.get_index_price() - 1) > 0.05
            )
                
            if do_print:
                px = perp.get_mark_price()
                S2 = perp.get_index_price()
                S3 = perp.get_collateral_price()
                px0 = 0.5 * (perp.get_price(perp.params['fLotSizeBC']) + perp.get_price(-perp.params['fLotSizeBC']))
                
                M = perp.amm_pool_cash_cc
                M_label = "M2" if perp.collateral_currency is CollateralCurrency.BASE else "M1" if perp.collateral_currency is CollateralCurrency.QUOTE else "M3"
                delta_t = (datetime.now() - run_t0).total_seconds() / 60
                
                trade_size_ema = perp.current_trader_exposure_EMA
                trades_vec = [
                    -perp.params['fMaximalTradeSizeBumpUp'] * trade_size_ema, # largest short not considering k*
                    -0.85 * trade_size_ema, # a typical short, ema is biased up
                    -perp.min_num_lots_per_pos * perp.params['fLotSizeBC'], # smallest short
                    perp.min_num_lots_per_pos * perp.params['fLotSizeBC'], # smallest long
                    0.85 * trade_size_ema, # a typical long, ema is biased up
                    perp.params['fMaximalTradeSizeBumpUp']  * trade_size_ema, # largest long not considering k*
                ]
                slippage_summary = " ".join([f"{100 * (perp.get_price(k) - px0)/px0: .3f}({k: .3f})" for k in trades_vec])
                premium_summary = f"mark={100*(px - S2)/S2: .3f}% slip=[{slippage_summary}](%,amt) "
                price_summary = f"S2={S2:.1f} "
                funding_summary = f"{M_label}={M:.2f} ({ 100 * (M / perp.amm_pool_target_size if perp.amm_pool_target_size > 0 else np.inf):3.0f}%) "
                
                print(
                    f"{symbol} " \
                    + premium_summary + price_summary + funding_summary \
                    + f"traders: {perp.get_num_active_traders()}/{total_num_traders[symbol]}, ema={trade_size_ema:.1f}"
                )
            
            
            
            new_noise_traders = 0
            new_momentum_traders = 0
            if t < 0.75 * time_df.shape[0]:
                joining_starts = time.time()
                traders_joining = np.random.binomial(
                    int(GROWTH_MULTIPLIER[symbol]),
                    NUM_TRADERS[symbol] / (0.75 * time_df.shape[0])
                )
                if traders_joining > 0:
                    for _ in range(traders_joining):
                        noise_trader = 1 if np.random.uniform() < 0.9 else 0
                        new_noise_traders += noise_trader
                        new_momentum_traders += 1 - noise_trader

                    total_noise_traders[symbol] += new_noise_traders
                    total_momentum_traders[symbol] += new_momentum_traders
                    total_num_traders[symbol] += new_noise_traders + new_momentum_traders  

                    perp_cc_ccy = get_collateral_ccy(symbol, COLL)
                    traders.extend(initialize_traders(
                            new_noise_traders, 
                            new_momentum_traders, 
                            0, 
                            perp_idx, 
                            cex_ts[symbol], 
                            amm, 
                            perp_cc_ccy,
                            num_trades_per_day=sim_state['num_trades_per_day'],
                            cash_qc=sim_state[symbol]['cash_per_trader_qc'],
                            arb_cash_qc=sim_state['usd_per_bot'],
                            prob_long=sim_state['prob_long'],
                            prob_best_tier=sim_state['prob_best_tier']))
                joining_ends = time.time()
                avg_joining_time += (joining_ends - joining_starts)
                num_joining_measurements += 1
            
        logging_ends = time.time()
        avg_logging_time += logging_ends - logging_starts
        num_logging_measurements += 1

        # overall pool status
        if t % monitoring_period == 0 or t == time_df.shape[0] - 1: # or amm.get_default_fund_gap_to_target_ratio() < 0.02:
            # # uncomment to log compute time
            # if num_trading_measurements > 0:
            #     print(f"avg_trading_time = {avg_trading_time / num_trading_measurements}")
            # if num_logging_measurements > 0:
            #     print(f"avg_logging_time = {avg_logging_time / num_logging_measurements}")
            # if num_logging_perp_measurements > 0:
            #     print(f"avg_logging_perp_time = {avg_logging_perp_time / num_logging_perp_measurements}")
            # if num_joining_measurements > 0:
            #     print(f"avg_joining_time = {avg_joining_time / num_joining_measurements}")
            
            S3 = perp.get_collateral_price()
            delta_t = (datetime.now() - run_t0).total_seconds() / 60
            arb_pnl_cc = sum([sim_state[symbol]['arb_pnl'] for symbol in SYMBOLS]) / idx_s3[t]
            arb_locked_usd = sum([sim_state[symbol]['arb_capital_used'] for symbol in SYMBOLS])
            arb_active = sum([sim_state[symbol]['num_arb_traders'] for symbol in SYMBOLS])
            traders_active = sum([perp.get_num_active_traders() for perp in amm.perpetual_list])
            total_traders = sum([total_num_traders[symbol] for symbol in SYMBOLS])
            DF = amm.default_fund_cash_cc
            msg = (
                f"{COLL} pool @ {time_df[t]}: "\
                + f"S3={S3:.1f} "\
                # + f"stakers = {amm.staker_cash_cc:.3f} " \
                + f"LP apy = {np.nanmean([100*s.get_apy() for s in stakers if s.has_staked]):.3f}% " \
                + f"DF = {amm.default_fund_cash_cc:.3f} ({100*amm.get_default_fund_gap_to_target_ratio():.1f}%), "\
                + f"Liquidations = {amm.liquidator_earnings_vault:.3f} ({liquidation_count}) "\
                + f"Bankrupcies = {sum([sim_state[symbol]['num_replaced_traders'] for symbol in SYMBOLS])}, "\
                + f"fees = {amm.fees_earned:.3f} "\
                + f"Num traders = {traders_active}/{total_traders} "\
                + f"Arb pnl = {arb_pnl_cc:.3f} "\
                + f"(${arb_locked_usd / np.max((arb_active, 1)):.0f} locked per bot, ${arb_locked_usd:.0f} total)"
            )
            
            print(msg)
            
            # PnL Stats
            if t > 0:
                pnl_abs_cc = [f"{name}: {amm.earnings[i]: .3f}" for i, name in enumerate(SYMBOLS)]
                print(f"Cumulative pool PnL (in {COLL}): {sum(amm.earnings.values()):.2f} {COLL}, {100 *(sum(amm.earnings.values()) / sum(amm.get_perpetual(perps[i]).total_volume_cc for i in range(len(SYMBOLS)))):.3f} %")
                print(f"Cumulative PnL by perp (in {COLL}):\t" + "\t".join(pnl_abs_cc))
                pnl_rel = [
                    f"{name}: {100 * amm.earnings[i] * S3 / amm.get_perpetual(perps[i]).total_volume_qc: .3f}" 
                    for i, name in enumerate(SYMBOLS)
                ]
                print(f"Cumulative PnL by perp (%, over traded volume in {QUOTE}):\t" + "\t".join(pnl_rel))
                trader_pnl = sum(t.pnl_cc for t in traders if t.position_bc != 0)
                print(f"Active traders PnL (in {COLL}): {trader_pnl:.1f} (total), {trader_pnl / np.max((1,traders_active)):.1f} (per trader)")
                # pprint.pprint(report_stats(amm_state.loc[:t], sim_state))
                time_left = (time_df.shape[0]/t -1)*delta_t
                print(f"{delta_t:.1f} minutes elapsed, {100.0 * t / time_df.shape[0]:.1f}% complete, approx. {1.1*time_left:.1f} minutes left.")
            print("".join(("-" for _ in range(100))) + "\n")

        # 'animate' AMM depth
        # if t % 10 == 0 and animate_premia:
        #     delta_t = (datetime.now() - run_t0).total_seconds() / 60
        #     ratioDF =  np.round(amm.get_default_fund_gap_to_target_ratio()*100)
        #     S3 = perp.get_collateral_price()
        #     DF = amm.default_fund_cash_cc
           
        #     for i in range(len(index_vec)):
        #         perp = amm.get_perpetual(perps[i])
        #         pos_range = usdrange / perp.get_index_price()
        #         price_range = np.array([perp.get_price(k) for k in pos_range])
        #         max_long = np.min((max_usd, perp.get_max_signed_trade_size_for_position(0, 1) * perp.get_index_price()))
        #         max_short = np.max((min_usd, perp.get_max_signed_trade_size_for_position(0, -1) * perp.get_index_price()))
        #         axs[i].cla()
        #         axs[i].plot(usdrange, 100 * (price_range / perp.get_index_price() - 1))
        #         # axs[i].set_yscale('symlog')
        #         axs[i].axvline(x=max_long, color='r')
        #         axs[i].axvline(x=max_short, color='r')
        #         axs[i].axhline(y=0, color='b')
        #         axs[i].set_xticks(np.linspace(min_usd, max_usd, 21))
        #         axs[i].set_xticklabels(np.linspace(min_usd, max_usd, 21), rotation=45)
        #         axs[i].set(ylabel=f"{index_vec[i]}", ylim=(-1.25, 1.25))
        #         axs[i].grid(linestyle='--', linewidth=1)

        #     fig.suptitle(f"Premium over index price (%)\n[{delta_t:.1f}] {COLL} pool {time_df[t]}: S3={S3:.1f} DF={DF:.3f} ({ratioDF:3.0f}%)")
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

            # # trader cash
            # ax_cash.cla()
            # trader_cash_qc = np.array([trader.cash_cc for trader in traders]) * idx_s3[0]
            # ax_cash.hist(trader_cash_qc, bins='auto')
            # ax_cash.grid(linestyle='--', linewidth=1)
            # ax_cash.ticklabel_format(style='plain')
            # ax_cash.tick_params(rotation=45)
            # fig_cash.canvas.draw()
            # fig_cash.canvas.flush_events()
 
        if amm.is_emergency:
            print("AMM emergency state reached")
            break
        
        amm.inc_time()

        if (
            (t - last_withdrawal) > sim_state['protocol_profit_withdrawal_frequency'] and 
            t > sim_state['protocol_profit_withdrawal_freeze_period']
        ):
            pnl_before = amm.protocol_earnings_vault
            fx_q2c = 1 / amm.perpetual_list[0].get_collateral_price()
            floor_cc = sim_state['protocol_profit_withdrawal_floor'] * fx_q2c
            cap_cc = sim_state['protocol_profit_withdrawal_cap'] * fx_q2c
            amm.withdraw_profit(sim_state['protocol_profit_withdrawal_rate'], floor_cc, cap_cc)
            pnl_after = amm.protocol_earnings_vault
            if pnl_after > pnl_before:
                profit = pnl_after - pnl_before
                if profit > 0:
                    print(f"Governance withraws {profit:.2f} {COLL} @ {time_df[t]}")
            last_withdrawal = t

    end_dt = datetime.fromtimestamp(sim_state['to_date'])
    delta_t = (datetime.now() - run_t0).total_seconds() / 60
    print(f"Simulation completed in {delta_t:.1f} minutes.\n")
    # some aggregates for pool
    amm_state = pd.DataFrame(amm_state, columns=get_amm_state_keys())
    amm_state['time'] = time_df.to_numpy()
    amm_state['idx_s3'] = idx_s3
    amm_state['total_volume_qc'] = 0
    amm_state['total_volume_cc'] = 0
    for i, symbol in enumerate(SYMBOLS):
        perp_idx = perps[i]
        state_dict[perp_idx] = pd.DataFrame(state_dict[perp_idx], columns=get_perp_state_keys())
        state_dict[perp_idx]['idx_px'] = idx_s2[symbol]
        state_dict[perp_idx]['cex_px'] = cex_ts[symbol]
        amm_state['total_volume_cc'] += state_dict[perp_idx]['perp_volume_cc']
        amm_state['total_volume_qc'] += state_dict[perp_idx]['perp_volume_qc']
    for i, symbol in enumerate(SYMBOLS):
        perp_idx = perps[i]
        date_appendix = f"_{end_dt.year}{end_dt.month}{end_dt.day}-{sim_state['sim_hash']}"
        file_name = f"res{total_num_traders[symbol] - total_arb_traders[symbol]}-{total_arb_traders[symbol]}-{symbol}{COLL}{date_appendix}"
        path_to_store = "results/" + file_name + ".csv"
        inc = 1
        while os.path.isfile(path_to_store):
            path_to_store = "results/" + file_name + "-" + str(inc) + ".csv"
            inc += 1
        df = pd.concat([amm_state, state_dict[perp_idx]], axis=1)
        df.to_csv(path_to_store)
        print(f"{symbol}:\t{path_to_store}")
    return amm_state


def load_perp_params(symbol, collateral):
    perp_params = dict()
    params_pattern = f"./data/params/{symbol}-{collateral}Perpetual.json"
    params_files = glob.glob(params_pattern)
    if len(params_files) < 1:
        quit(f"Could not find a parameter file for this symbol/collateral combination: {params_pattern} not found.")
    with open(params_files[0]) as json_file:
        perp_params = json.load(json_file)

    perp_params['r'] = 0 # always
    perp_params['initial_margin_cash_cc'] = 0
    perp_params['initial_staker_cash_cc'] = 0
    
    return perp_params

def load_pool_params(collateral):
    pool_params = dict()
    params_pattern = f"./data/params/{collateral}PoolConfig.json"
    params_files = glob.glob(params_pattern)
    if len(params_files) < 1:
        quit(f"Could not find a parameter file for this collateral: {params_pattern} not found.")
    with open(params_files[0]) as json_file:
        pool_params = json.load(json_file)[0]
    return pool_params


def init_index_data(from_date, to_date, reload=False, symbol : str|list='BTCUSD', collateral='BTC'):
    if not type(symbol) is list:
        symbol = [symbol]
    df_idx_s2 = dict()
    for sym in symbol:
        df_idx_s2[sym] = load_price_data(
            sym, 
            source="chainlink", 
            network=NETWORK[sym], 
            from_date=from_date, 
            to_date=to_date
        )
        df_idx_s2[sym].rename(columns={'price': sym}, inplace=True)
    
    if collateral in ['XUSD', 'ZUSD', 'USDT', 'USDC', 'USD']:
        df_idx_s3 = df_idx_s2[symbol[0]].copy()
        df_idx_s3.rename(columns={symbol[0]: 'price'}, inplace=True)
        df_idx_s3['price'] = 1
    else:
        df_idx_s3 = load_price_data(
            f"{collateral}USD", 
            source="chainlink", 
            network=NETWORK[f"{collateral}USD"], 
            from_date=from_date, 
            to_date=to_date
        )
    df_idx_s3.rename(columns={'price': 's3'}, inplace=True)

    # TODO: arb with actual exchange data needs their TS loaded here
    df_perppx = dict()
    for sym in symbol:
        df_perppx[sym] = load_price_data(
            sym, 
            source="chainlink", 
            network=NETWORK[sym], 
            from_date=from_date, 
            to_date=to_date
        )
        df_perppx[sym].rename(columns={'price': sym + '_perp'}, inplace=True)
    
    # combine data
    df = df_idx_s3.copy()
    for sym in symbol:
        df = pd.merge(df, df_idx_s2[sym], on=['t', 'timestamp'], how='outer')
        df = pd.merge(df, df_perppx[sym], on=['t', 'timestamp'], how='outer')
        df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.fillna(method='ffill', axis=0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    s2 = {sym: df[sym].to_numpy() for sym in symbol}
    perp = {sym: df[sym + '_perp'].to_numpy() for sym in symbol}
    return (s2, df["s3"].to_numpy(), perp, df['t'])

def perturbe_perp_ts(idx_s2, perp_px):
    # hack: don't trust cex premium, use ~ 5 bps +- 20 bps
    my_mean = 0.0080
    my_std = 0.0040
    myclip_a = -0.01
    myclip_b = 0.01
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    # #  new premium = (p + s * eps) / s - 1 = p /s - 1 + eps = eps + eps' = original + biased
    perp_px = perp_px + idx_s2 * stats.truncnorm(a, b, loc=my_mean, scale=my_std).rvs(perp_px.shape[0])
    print(stats.describe(perp_px /idx_s2 - 1 ))
    return perp_px


def format_time(df, from_date, to_date):
    df['timestamp'] = df['timestamp'].astype('int64')
    df.drop_duplicates(subset='timestamp', keep="last", inplace=True)
    assert(len(df['timestamp'].unique()) == df.shape[0])
    # t is the date in UTC
    df['t'] = pd.to_datetime(df['timestamp'], unit="s", origin='unix', utc=True).dt.ceil('min')
    if len(df['t'].unique()) != df.shape[0]:
        # print(f"Warning: rounding to the nearest minute created loss of data, {df.shape[0] - len(df['t'].unique())} points lost")
        df.drop_duplicates(subset='t', keep="last", inplace=True)
    df['timestamp'] = df['t'].apply(datetime.timestamp).astype('int64')
    if not from_date is None:
        df = df[df['timestamp'] >= from_date]
    if not to_date is None:
        df = df[df['timestamp'] <= to_date]
    df.sort_values('timestamp', inplace=True)
    return df[['t', 'timestamp', 'price']]


def load_price_data(symbol: str, source: str|None=None, network: str|None=None, from_date=None, to_date=None):
    """Load index time series data into a pandas data frame

    Args:
        symbol (str): currency pair, e.g. 'BTCUSD', 'XAGUSD', 'SOVBTC', etc
        source (str): e.g. 'chainlink', 'some_api', etc
        network (str): e.g. 'BSC', 'RSK', etc
    """
    if not source:
        source = "*"
    if not network:
        network = "*"
    
    path_pattern = f"./data/index/{source}/{symbol}_{network}_*.csv"
    data_files = glob.glob(path_pattern)

    if len(data_files) < 1:
        quit(f"Could not find files with pattern {path_pattern}")
    if len(data_files) > 1:
        if source == '*':
            quit(f"Multiple sources found for symbol {symbol} - please specify")
        if network == "*":
            quit(f"Multiple networks found for symbol {symbol} - please specify")

    df = pd.DataFrame(columns=['t', 'timestamp', 'price'])
    # print(f"Looping over {data_files}...")
    # we take all files and combine them to have as much data as possible
    for filename in data_files:
        df_tmp = pd.read_csv(filename)
        assert(all((c in df_tmp.columns for c in ['timestamp', 'price'])))
        df_tmp = df_tmp[['timestamp', 'price']]
        if len(df_tmp['timestamp'].unique()) != df_tmp.shape[0]:
            # print(f"{filename} contained duplicate entries: {df_tmp.shape[0] - len(df_tmp['timestamp'].unique())} timestamps repeated")
            df_tmp.drop_duplicates(subset='timestamp', keep="first", inplace=True)
        # print(f"Processing {filename}, which contains {df_tmp.shape[0]} observations")
        df_tmp = format_time(df_tmp, from_date, to_date)
        df = pd.concat([df, df_tmp])
        df.drop_duplicates(subset='timestamp', keep="first", inplace=True)
        df.sort_values('timestamp', inplace=True)
    if df.shape[0] == 0:
        quit(f"Could not find non-empty files with pattern {path_pattern}")
    # data is cadlag constant interpolated to minute frequency (format_time already discards second granularity)
    t1 = pd.to_datetime(df['timestamp'].min(), unit="s", origin='unix', utc=True)
    t2 = pd.to_datetime(df['timestamp'].max(), unit="s", origin='unix', utc=True)
    df_out = pd.DataFrame(pd.date_range(t1, t2, freq='min'), columns=['t'])
    # df_out['t'] = df_out['t'].dt.strftime("%y-%m-%d %H:%M")
    df = pd.merge(df_out, df, on='t', how='left')
    df['timestamp'] = df['t'].apply(datetime.timestamp).astype('int64')
    df.sort_values('timestamp', inplace=True)
    df['t'] = df['t'].dt.strftime("%y-%m-%d %H:%M")
    df.reset_index(inplace=True, drop=True)
    assert(df['timestamp'].isna().sum() < 1)
    # print(f"Done with symbol {symbol}!\n")
    return df

def get_collateral_ccy(symbol, collateral):
    usds = ['USD', 'ZUSD', 'XUSD', 'USDT', 'USDC']
    if symbol in [f"{collateral}{usd}" for usd in usds]:
        ccy = CollateralCurrency.BASE
    elif collateral in usds:
        ccy = CollateralCurrency.QUOTE
    else:
        ccy = CollateralCurrency.QUANTO
    return ccy

def draw_cash_sample(expected_cash_qc, k=0.5):
    scale = expected_cash_qc * k
    min_cash = np.max((expected_cash_qc / 4, 500)) # no less than $500 per trader
    mean_over_min = (expected_cash_qc - min_cash) / scale
    return min_cash + np.random.gamma(mean_over_min, scale)


def initialize_stakers(amm, num_stakers, cash_qc, monthly_stakes=-1, holding_period_months=-1):
    stakers = []
    for _ in range(num_stakers):
        fxq2c = 1 / amm.perpetual_list[0].get_collateral_price()
        cash_cc = draw_cash_sample(cash_qc) * fxq2c
        stakers.append(NoiseStaker(amm, cash_cc, monthly_stakes, holding_period_months))
    return stakers


def stakers_stake(stakers):
    for staker in stakers:
        staker.stake()


def initialize_traders(noise_traders, momentum_traders, arb_traders, perp_idx, cex_perp_px, amm, currency, num_trades_per_day=1, cash_qc=-1, arb_cash_qc=-1, prob_best_tier=0, prob_long=-1):
    trader_list=[]
    m = np.max((noise_traders, momentum_traders, arb_traders))
    num = [0,0,0]
    fx_q2c = 1 / amm.get_perpetual(perp_idx).get_collateral_price()
    assert(not np.isnan(cash_qc))
    assert(not np.isnan(fx_q2c))
    cash_samples = []
    for k in range(m):
        if num[0]<noise_traders:
            cash_samples.append(draw_cash_sample(cash_qc))
            is_best_tier = np.random.uniform() <= prob_best_tier
            trader_list.append(NoiseTrader(amm, perp_idx, currency, cash_cc=cash_samples[-1] * fx_q2c, daily_trades=num_trades_per_day, is_best_tier=is_best_tier, prob_long=prob_long))
            num[0] += 1
        if num[1]<momentum_traders:
            cash_samples.append(draw_cash_sample(cash_qc))
            is_best_tier = np.random.uniform() <= prob_best_tier
            trader_list.append(MomentumTrader(amm, perp_idx, currency, cash_cc=cash_samples[-1] * fx_q2c, is_best_tier=is_best_tier))
            num[1] += 1
        if num[2]<arb_traders:
            trader_list.append(ArbTrader(amm, perp_idx, currency, cex_perp_px, cash_cc=arb_cash_qc * fx_q2c))
            num[2] += 1
    is_one = num[0] + num[1] == 1
    msg = f"Initialized {num[0] + num[1]} trader{'' if is_one else 's'} with {'' if is_one else 'approx.'} ${np.mean(cash_samples):.2f}"
    if not is_one:
        msg = msg + f" +- ${np.std(cash_samples):.2f} each (90% lie in the range ${np.quantile(cash_samples, 0.05):.2f} - ${np.quantile(cash_samples, 0.95):.2f})"
    if m > 4:
        print(msg)
    return trader_list


def traders_trade(traders, amm, sim, cex_ts):

    num_noise_replaced = 0
    num_momentum_replaced = 0
    for symbol in SYMBOLS:
        sim[symbol]['num_momentum_traders'] = 0
        sim[symbol]['num_noise_traders'] = 0
        sim[symbol]['num_arb_traders'] = 0
        sim[symbol]['arb_pnl'] = 0
        sim[symbol]['arb_capital_used'] = 0

    # traders trade in random (but np.random.seed-predictable) order  
    idx_arr = np.random.choice(len(traders), len(traders), replace=False)  
    for k in idx_arr:
        trader = traders[k]

        dPos, is_close = trader.query_trade_amount()
        is_active = trader.is_active

        perp_idx = trader.perp_idx
        perp = amm.get_perpetual(perp_idx)
        symbol = SYMBOLS[perp_idx]
        
        if not is_active:
            sim[symbol]['num_bankrupt_traders'] += 1
            if sim[symbol]['num_replaced_traders'] < sim[symbol]['max_trader_replacements']:
                # replace the trader
                cash_cc = draw_cash_sample(sim[symbol]['cash_per_trader_qc']) / perp.idx_s3[perp.current_time]
                if isinstance(trader, MomentumTrader):
                    traders[k] = MomentumTrader(
                        amm, 
                        perp_idx, 
                        perp.collateral_currency, 
                        cash_cc=cash_cc)
                    num_momentum_replaced += 1
                elif isinstance(trader, NoiseTrader):
                    traders[k] = NoiseTrader(
                        amm, 
                        perp_idx, 
                        perp.collateral_currency, 
                        cash_cc=cash_cc)
                    num_noise_replaced += 1

                sim[symbol]['num_replaced_traders'] += 1
        else:
            if dPos != 0:
                px = trader.trade(dPos, is_close)
                if px is not None:
                    sim[symbol]['num_trades'] += 1
                    if isinstance(trader, ArbTrader):
                        sim[symbol]['cex_volume_bc'] += np.abs(dPos)
                # else:
                #     msg = f"Warning: trade rejected quietly: {trader.__class__.__name__} pos={trader.position_bc} cash={trader.cash_cc} trade={dPos} {'closing' if is_close else 'opening'}"
                #     print(msg)
           
            if np.abs(trader.position_bc) > 0:
                # add to count of active traders this time step
                if isinstance(trader, MomentumTrader):
                    sim[symbol]['num_momentum_traders'] += 1
                elif isinstance(trader, NoiseTrader):
                    sim[symbol]['num_noise_traders'] += 1
                elif isinstance(trader, ArbTrader):
                    sim[symbol]['num_arb_traders'] += 1
                    sim[symbol]['arb_capital_used'] += 2 * np.abs(trader.position_bc) * perp.get_mark_price()

        if isinstance(trader, ArbTrader):
            sim[symbol]['arb_pnl'] += trader.pnl
    
    # if num_noise_replaced > 0:
    #     print(f"Replaced {num_noise_replaced} noise traders")
    # if num_momentum_replaced > 0:
    #     print(f"Replaced {num_momentum_replaced} momentum traders")
    

def traders_liquidate(traders, do_print=False, update_sim=True, sim: dict|None=None):
    num_liquidated = 0
    num_liquidated_long = 0
    num_liquidated_short = 0
    idx_arr = np.random.choice(len(traders), len(traders), replace=False)
    for k in idx_arr:
        perpetual = traders[k].get_perpetual()
        not_safe = not traders[k].is_maintenance_margin_safe(perpetual)
        if not_safe:
            #print(f"Trader {traders[k].id} is {'not' if not_safe else ''} margin safe")
            is_long = traders[k].position_bc > 0
            liq = perpetual.liquidate(traders[k])
            if liq:
                num_liquidated += 1
                if update_sim and not sim is None:
                    sim[SYMBOLS[traders[k].perp_idx]]['num_trades'] += 1
                if isinstance(traders[k], ArbTrader):
                    print('arb liq')
                if is_long:
                    num_liquidated_long += 1
                else:
                    num_liquidated_short += 1
                if do_print:
                    print(f"liquidated trader #{k}")

    # if num_liquidated > 0:
    #      print(f"Liquidated {num_liquidated} traders: {num_liquidated_long} long, {num_liquidated_short} short")
    return num_liquidated


def traders_pay_funding(traders):
    idx_arr = np.random.choice(len(traders), len(traders), replace=False)
    for k in idx_arr:
        traders[k].pay_funding()


def record_initial_endowment(traders):
    endowment = {'momentum': 0, 'noise': 0, 'arb':0,
        'num_momentum':0, 'num_noise':0, 'num_arb':0}
    # s0 = traders[0].amm.perpetual_list[0].get_index_price() 
    for k in range(len(traders)):
        s0 = traders[k].amm.get_perpetual(traders[k].perp_idx).get_collateral_to_quote_conversion()
        cash = traders[k].cash_cc*s0
        if isinstance(traders[k], MomentumTrader):
            endowment['momentum'] += cash
            endowment['num_momentum'] += 1
        elif isinstance(traders[k], NoiseTrader):
            endowment['noise'] += cash
            endowment['num_noise'] += 1
        elif isinstance(traders[k], ArbTrader):
            endowment['arb'] += cash
            endowment['num_arb'] += 1
    num = str(endowment['num_noise']+endowment['num_momentum'])+"-"+str(endowment['num_arb'])
    file = open('results/Endowment'+str(num)+".pkl", "wb")
    pickle.dump(endowment, file)
    return endowment


def get_amm_state_keys():
    keys = [
        "amm_cash",
        "pool_margin",
        "pricing_staked_cash",
        "cash_staked",
        "staker_cash",
        "df_cash",
        "share_token_supply",
        "protocol_earnings_vault",
        "liquidator_earnings_vault",
        "df_target",
        "amm_target",
        "df_cash_to_target",
        "trader_pnl_cc",
        "lp_apy",
    ]
    for i, name in enumerate(SYMBOLS):
        keys.append(f"{name}_pnl_cc")
    return keys

def record_amm_state(t, state, amm, stakers, traders):

    amm_cash = amm.get_amm_funds()
    new_state = [
        amm_cash,
        sum((p.amm_trader.cash_cc  for p in amm.perpetual_list)),
        sum((p.get_pricing_staked_cash_for_perp() for p in amm.perpetual_list)),
        amm.staker_cash_cc,
        sum((s.get_position_value_cc() for s in stakers if s.has_staked)),
        amm.default_fund_cash_cc,
        amm.share_token_supply,
        
        amm.protocol_earnings_vault,
        amm.liquidator_earnings_vault,
        amm.get_default_fund_gap_to_target() + amm.default_fund_cash_cc,
        amm.get_amm_pools_gap_to_target() + amm_cash,
        amm.get_default_fund_gap_to_target(),
        sum(t.pnl_cc for t in traders if t.position_bc != 0),
        np.nanmean([s.get_apy() for s in stakers if s.has_staked])
    ]
    new_state.extend([amm.earnings[i] for i, name in enumerate(SYMBOLS)])
    state[t,:] = new_state


def get_perp_state_keys():
    return ['mark_price',
        'mid_price',
        'funding_rate',
        # cash
        'perp_amm_cash',
        'perp_pricing_staked_cash',

        # targets
        'perp_amm_target',
        'perp_amm_target_baseline',
        'perp_amm_target_stress',
        'perp_df_target',
        'num_noise_traders',
        'num_momentum_traders',
        'num_arb_traders',
        'num_bankrupt_traders',

        # aggregates 
        'perp_volume_bc',
        'perp_volume_qc',
        'perp_volume_cc',
        # state.at[time_idx, 'open_interest'] = perp.open_interest
        
        'current_trader_exposure_EMA',
        'num_trades',
        
        # AMM margin
        'perp_margin',
        'perp_K2',
        'perp_L1',
        
        # trade sizes and prices
        'max_long_trade',
        'max_long_price',
        'max_short_trade',
        'max_short_price',
        'min_long_price',
        'min_short_price',
        '10k_long_price',
        '10k_short_price',
        '100klots_long_price',
        '100klots_short_price',
        'avg_long_price',
        'avg_short_price',
        
        # trader stats
        'perp_trader_pnl_cc',
    ]


def record_perp_state(time_idx, state_dict, perp, sim, traders):
    """record statistics and state"""
    
    # pre-compute some things
    # max new long position
    pos1 =  perp.get_max_signed_trade_size_for_position(0, 1)
    # max new short position
    pos2 = perp.get_max_signed_trade_size_for_position(0, -1)
    # min new long/short position
    pos3 = perp.min_num_lots_per_pos * perp.params['fLotSizeBC']
    # $10,000 worth of index at spot
    pos4 = 10_000 / perp.get_index_price()
    # 100,000 lots
    pos5 = 100_000 * perp.params['fLotSizeBC']
    # Upward-biased EMA
    pos6 = perp.current_trader_exposure_EMA
    
    # store
    state_dict[perp.my_idx][time_idx,:] = [
        perp.get_mark_price(),
        0.5 * (perp.get_price(perp.params['fLotSizeBC']) + perp.get_price(-perp.params['fLotSizeBC'])),
        perp.get_funding_rate(),
        # cash
        perp.amm_pool_cash_cc,
        perp.get_pricing_staked_cash_for_perp(),

        # targets
        perp.amm_pool_target_size,
        perp.get_amm_pool_size_for_dd(perp.params['fAMMTargetDD'][0]),
        perp.get_amm_pool_size_for_dd(perp.params['fAMMTargetDD'][1]),
        perp.default_fund_target_size,
        sim['num_noise_traders'],
        sim['num_momentum_traders'],
        sim['num_arb_traders'],
        sim['num_bankrupt_traders'],

        # aggregates 
        perp.total_volume_bc,
        perp.total_volume_qc,
        perp.total_volume_cc,
        # state.at[time_idx, 'open_interest'] = perp.open_interest
        
        perp.current_trader_exposure_EMA,
        sim['num_trades'],
        
        # AMM margin
        perp.amm_trader.cash_cc,
        perp.amm_trader.position_bc,
        perp.amm_trader.locked_in_qc,
        
        # trades and prices
        pos1,
        perp.get_price(pos1),
        pos2,
        perp.get_price(pos2),
        perp.get_price(pos3),
        perp.get_price(-pos3),
        perp.get_price(pos4),
        perp.get_price(-pos4),
        perp.get_price(pos5),
        perp.get_price(-pos5),
        perp.get_price(pos6),
        perp.get_price(-pos6),    
        
        # trader pnl
        sum(t.pnl_cc for t in traders if t.position_bc != 0 and t.perp_idx == perp.my_idx)
    ]

def to_readable_fiat(x):
    if x >= 500_000:
        return f"{x / 1_000_000:.2f} million {QUOTE}"
    if x >= 1_000:
        return f"{x / 1_000:.2f} thousand {QUOTE}"
    return f"{x} {QUOTE}"


if __name__ == "__main__":
    main()