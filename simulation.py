#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

from fileinput import filename
import glob
import json
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

# - Bear, bull
# - Prob long: 40%, 60%
# - n times each?
# - initial investment? 1M, 2M
# - total traders in system ~ 1K by the end of the sim period
#  (8 * n)

# 200/400 btc, 150/300 xau, 100/200 chf, 150/300 gbp, 100/200 eth,  

INDEX = ['BTC', 'ETH', 'XAU', 'BNB', 'CHF', 'GBP', 'SPY', 'TSLA', 'MATIC', 'LINK']
QUOTE = 'USD'
COLL = 'USD'

# traders that join on day 1
NUM_TRADERS = {
    'BTCUSD': 10,
    'XAUUSD': 10,
    'BNBUSD': 0, 
    'ETHUSD': 10,
    'CHFUSD': 10,
    'GBPUSD': 10,
    'SPYUSD': 0,
    'TSLAUSD': 0,
    'MATICUSD': 10,
    'LINKUSD': 0,
}

# new traders join randomly - up to how many each time? 
# e.g. if we start with N traders and this number is M, there will be in average (M+1) * N traders by the end of the simulation
GROWTH_DIVIDER = 3 # to keep same 'speed' but reduced target (e.g. to go from 4 to 2 months and keep all other params the same, use divider = 2)
GROWTH_MULTIPLIER = {
    'BTCUSD': 20 // GROWTH_DIVIDER, # 20 x 20
    'XAUUSD': 20 // GROWTH_DIVIDER, # 30 x 10
    'BNBUSD': 20 // GROWTH_DIVIDER,  
    'ETHUSD': 20 // GROWTH_DIVIDER, # 20 x 10
    'CHFUSD': 20 // GROWTH_DIVIDER, # 20 x 10
    'GBPUSD': 20 // GROWTH_DIVIDER, # 30 x 10
    'SPYUSD': 20 // GROWTH_DIVIDER, 
    'TSLAUSD': 20 // GROWTH_DIVIDER,
    'MATICUSD': 20 // GROWTH_DIVIDER,
    'LINKUSD': 20 // GROWTH_DIVIDER,
}

# how many arb bots we run for each perp
BOTS_PER_PERP = {
    'BTCUSD': 0,
    'BNBUSD': 0,
    'XAUUSD': 0,
    'ETHUSD': 0,
    'CHFUSD': 0,
    'GBPUSD': 0,
    'SPYUSD': 0,
    'TSLAUSD': 0,
    'MATICUSD': 0,
    'LINKUSD': 0,
}

# network
NETWORK = {
    'BTCUSD': 'BSC_Mainnet', 
    'XAUUSD': 'BSC_Mainnet', 
    'BNBUSD': 'BSC_Mainnet', 
    'ETHUSD': 'BSC_Mainnet',
    'GBPUSD': 'BSC_Mainnet',
    'SPYUSD': 'BSC_Mainnet',
    'CHFUSD': 'Polygon',
    'TSLAUSD': 'BSC_Mainnet',
    'MATICUSD': 'BSC_Mainnet',
    'LINKUSD': 'BSC_Mainnet',
}

# when a trader goes bankrupt, a new one joins - up to how many?
MAX_TRADER_REPLACEMENTS = 2000

# # protocol investment/profit
# initial_protocol_investment =  1_000_000
# usd_per_bot = 0 # the total capital needed for arbs is this times the number of bots
# liq_provider_initial_investment = 1_000 # in dollars
# # protocol profit withdrawal rules
# protocol_profit_withdrawal_frequency = 60 * 24 * 20 # we try to withdraw every n days
# protocol_profit_withdrawal_rate = 0.20 # % of the excess (if any) we withdraw
# protocol_profit_withdrawal_floor = 1_000 # the least dollar amount worth withdrawing
# protocol_profit_withdrawal_freeze_period = 60 * 24 * 30 # no withdrawals for the first 30 days after launch


# # animate price premium while running? (slows things down! use only for exploration)
# animate_premia = False

# # traders
# usd_per_trader = 1_000 # this is an average, there's a random sampling involved
# num_trades_per_day = 1
# prob_staking = 0.90 # prob that trader stakes SOV and pays zero fees / not used currently

# # simulation horizon
# from_date = datetime(2022, 6, 20, 0, 0, tzinfo=timezone.utc).timestamp()
# to_date = datetime(2022, 7, 20, 0, 0, tzinfo=timezone.utc).timestamp()

# sim_params = dict()
index_vec = [b + 'USD' for b in INDEX if NUM_TRADERS[b + 'USD'] > 0]

def main():
    run_t0 = datetime.now()
    np.random.seed(seed)
    print(f"Run hash: {run_hash}")
    (idx_s2, idx_s3, cex_ts, time_df) = init_index_data(from_date, to_date, reload=False, symbol=index_vec, collateral=COLL)
    # summary data
    # plt.ion()
    # for symbol in index_vec:
    #     num_minutes = 48 * 60 # a day? holding period?
    #     num_hours = num_minutes // 60
    #     stat_window = 30 # why?
    #     print(symbol)
    #     print(idx_s2[symbol].shape)
    #     dlog = np.log(idx_s2[symbol][num_minutes:]) - np.log(idx_s2[symbol][:-num_minutes])
    #     # print(dlog[:10])
    #     print(np.nanmean(dlog))
    #     print(np.nanstd(dlog))
    #     vol_series = pd.Series(dlog).rolling(window=stat_window).std()
    #     plt.plot(100 * vol_series)
    #     plt.xlabel('time?')
    #     plt.ylabel('%')
    #     plt.title(f"{symbol}\nRealized vol of {num_hours}h-log-returns, {stat_window}-day rolling window")
    #     plt.show()
    # return
    # note: same number of traders for each perp (this is total per perp, not overall)
    total_noise_traders = dict()
    total_momentum_traders = dict()
    total_arb_traders = dict()
    total_num_traders = dict()
    for symbol in index_vec:
        total_noise_traders[symbol] = int(np.ceil(NUM_TRADERS[symbol] * 0.95))
        total_momentum_traders[symbol] = NUM_TRADERS[symbol] - total_noise_traders[symbol]
        total_arb_traders[symbol] = BOTS_PER_PERP[symbol] # int(0 * ntrader)
        total_num_traders[symbol] = total_noise_traders[symbol] + total_momentum_traders[symbol] + total_arb_traders[symbol]
    
    
    
    # instrument parameters
    perp_params = dict() # dict of dicts
    # amm parameters
    amm_params = dict()
    # amm_params['initial_staker_cash_cc'] = liq_provider_initial_investment / idx_s3[0]
    amm_params['ceil_staker_pnl_share'] = 0.80
    amm_params['target_pool_size_update_time'] = 86400/3 # 8 hours in seconds
    amm_params['block_time_sec'] = 60 #30
    
    amm = AMM(amm_params, 0)
    state_dict = dict()
    perps = []
    traders = []


    amm_state = pd.DataFrame(
            index=np.arange(time_df.shape[0]), 
            columns=['time', 'idx_s3', 'df_cash', 'amm_cash', 'pool_margin', 'pricing_staked_cash',  'cash_staked', 'staker_cash',
                    'protocol_earnings_vault', 'df_cash_to_target', 'liquidator_earnings_vault', 
                    'share_token_supply', 'total_volume_qc',]) # 'trader_cash'])
    amm_state.loc[:,'time'] = time_df.to_numpy()
    amm_state.loc[:,'idx_s3'] = idx_s3
    protocol_cash_cc = initial_protocol_investment / idx_s3[0]

    for symbol in index_vec:
        sim_params[symbol] = dict()
        sim_params[symbol]['cash_per_trader_qc'] = usd_per_trader

        perp_params[symbol] = load_perp_params(symbol, COLL)
        perp_params[symbol]['perp_amm_cash_cc'] = np.min(
            (perp_params[symbol]['fAMMMinSizeCC'], protocol_cash_cc / len(index_vec)))
        protocol_cash_cc -= perp_params[symbol]['perp_amm_cash_cc']
        # report
        # print(f"{symbol}: AMM pool={perp_params[symbol]['perp_amm_cash_cc']:.4f}, target={perp_params[symbol]['fAMMMinSizeCC']:.4f}")

        # simulation parameters
        sim_params[symbol]['num_replaced_traders'] = 0
        sim_params[symbol]['max_trader_replacements'] = MAX_TRADER_REPLACEMENTS
        # active traders (starts at 0)
        sim_params[symbol]['num_momentum_traders'] = 0
        sim_params[symbol]['num_arb_traders'] = 0
        sim_params[symbol]['num_noise_traders'] = 0
        sim_params[symbol]['num_bankrupt_traders'] = 0    
        sim_params[symbol]['arb_pnl'] = 0    
        sim_params[symbol]['total_bitmex_btc_vol'] = 0
        sim_params[symbol]['num_trades'] = 0
        sim_params[symbol]['arb_capital_used'] = 0

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
            verbose = 0
        )
        perps.append(perp_idx)

        # dataframe to store simulation results
        state_dict[perp_idx] = pd.DataFrame(
            index=np.arange(time_df.shape[0]), 
            columns=['idx_px', 'mark_price', 'mid_price',
                    'perp_amm_cash', 'perp_margin', 'perp_pricing_staked_cash', 
                    'perp_df_target',
                    'perp_amm_target', 'perp_amm_target_baseline', 'perp_amm_target_stress', 
                    'current_trader_exposure_EMA',
                    'num_noise_traders', 'num_momentum_traders','num_arb_traders','num_bankrupt_traders', 
                    'perp_volume_qc', 'perp_K2', 'perp_L1',
                    'funding_rate', '10k_long_price', '10k_short_price', '100klots_long_price', '100klots_short_price',
                    # 'current_AMM_exposure_EMA_0','current_AMM_exposure_EMA_1','current_trader_exposure_EMA',
                    # 'protocol_btc_volume', 'locked_in_qc', 'open_interest',
                    # 'current_locked_in_value_EMA_0','current_locked_in_value_EMA_1',
                    # 'total_bitmex_btc_vol', 'arb_capital_used', 'arb_pnl', 'cex_px'
                    'max_long_trade', 'max_short_trade', 'num_trades', 'max_long_price', 'max_short_price', 
                    'avg_long_price', 'avg_short_price', 'min_long_price', 'min_short_price',
                    ])
        state_dict[perp_idx].loc[:,'idx_px'] = idx_s2[symbol]
        state_dict[perp_idx].loc[:,'cex_px'] = cex_ts[symbol]
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
                cash_qc=sim_params[symbol]['cash_per_trader_qc'],
                arb_cash_qc=usd_per_bot,
                prob_staking=prob_staking,
                prob_long=prob_long
            )
        )
        exec_time = [0,0,0]
        record_initial_endowment(traders)
        save_perp_params(symbol, COLL, amm.get_perpetual(perp_idx))

    # the rest goes to DF
    amm.default_fund_cash_cc += protocol_cash_cc

    # initi stakers
    stakers = initialize_stakers(amm, num_stakers, cash_per_staker, monthly_stakes, holding_period_months)

    amm.export(f"./data/{COLL}_pool")
    
    
    # # plot trader cash
    # trader_cash_qc = np.array([trader.cash_cc for trader in traders]) * idx_s3[0]
    # fig_cash, ax_cash = plt.subplots()
    # ax_cash.hist(trader_cash_qc, bins='auto', density=True)
    # fig_cash.suptitle(f"Trader cash distribution ({QUOTE})")
    # fig_cash.supxlabel(f"{QUOTE}")

    # plot premium charts/AMM depth
    if animate_premia:
        plt.ion()
        usdrange = np.linspace(-100_000, 100_000, 1_000)    
        fig, axs = plt.subplots(len(index_vec), 1, sharex=True)
        fig.suptitle("Premium over index price (%)")
        fig.supxlabel("Notional value (USD)")
        max_usd = np.max(usdrange)
        min_usd = np.min(usdrange) 


    # loop over time
    print(f"\nBeginning simulation...\n")

    trade_count = 0
    liquidation_count = 0
    last_withdrawal = 0
    avg_trading_time, num_trading_measurements = 0, 0
    avg_logging_time, num_logging_measurements = 0, 0

    for t in range(time_df.shape[0]):
        # trading happens
        trading_starts = time.time()
        traders_pay_funding(traders)
        liquidation_count += traders_liquidate(traders, amm)
        trades_so_far = sim_params[symbol]['num_trades']
        traders_trade(traders, amm, sim_params, cex_ts)
        delta_trades = sim_params[symbol]['num_trades'] - trades_so_far
        trade_count += delta_trades
        trading_ends = time.time()
        # staking
        stakers_stake(stakers)
        
        avg_trading_time += (trading_ends - trading_starts) / len(traders)
        num_trading_measurements += 1

        record_amm_state(t, amm_state, amm, traders, stakers)

        # logging status
        logging_starts = time.time()
        for i, symbol in enumerate(index_vec):
            perp_idx = perps[i]
            
            record_perp_state(t, state_dict[perp_idx], perp_idx, traders, sim_params[symbol], amm)

            

            if t % 15000 == 0:
                perp = amm.get_perpetual(perp_idx)
                max_long = perp.get_max_signed_trade_size_for_position(0, 1)
                max_short = perp.get_max_signed_trade_size_for_position(0, -1)
                min_pos_size = perp.min_num_lots_per_pos * perp.params['fLotSizeBC']
                
                # this perpetual's status
                do_print = (
                    True or
                    perp.amm_pool_cash_cc / perp.amm_pool_target_size < 0.50 or 
                    #amm.get_default_fund_gap_to_target_ratio() < 0.05 or
                    np.abs(perp.get_mark_price() / perp.get_index_price() - 1) > 0.02 or
                    max_long < min_pos_size or
                    max_short > -min_pos_size
                )
                
                if do_print:
                    px = perp.get_mark_price()
                    S2 = perp.get_index_price()
                    S3 = perp.get_collateral_price()
                    px0 = 0.5 * (perp.get_price(perp.params['fLotSizeBC']) + perp.get_price(-perp.params['fLotSizeBC']))

                    # SovPnL = np.round((state_dict[perp_idx].iloc[t]['protocol_earnings_vault']- initial_protocol_investment)*S2*100)/100
                    # vol = np.round(perp.total_volume*10)/10
                    # volB = np.round(sim_params[symbol]['total_bitmex_btc_vol']*10)/10
                    M = perp.amm_pool_cash_cc
                    M_label = "M2" if perp.collateral_currency is CollateralCurrency.BASE else "M1" if perp.collateral_currency is CollateralCurrency.QUOTE else "M3"
                    # K2 = -perp.amm_trader.position_bc
                    # cash_cc = perp.amm_trader.cash_cc
                    # L = -perp.amm_trader.locked_in_qc
                    # net_balance = L - S2 * K2 + M * S3
                    
                    
                    delta_t = (datetime.now() - run_t0).total_seconds() / 60
                    
                    trade_size_ema = perp.current_trader_exposure_EMA
                    trades_vec = [
                        -perp.params['fMaximalTradeSizeBumpUp'] * trade_size_ema, # largest short not considering k*
                        -0.85 * trade_size_ema, # a typical short
                        -perp.min_num_lots_per_pos * perp.params['fLotSizeBC'], # smallest short
                        perp.min_num_lots_per_pos * perp.params['fLotSizeBC'], # smallest long
                        0.85 * trade_size_ema, # a typical long
                        perp.params['fMaximalTradeSizeBumpUp']  * trade_size_ema, # largest long not considering k*
                    ]
                    slippage_summary = " ".join([f"{100 * (perp.get_price(k) - px0)/px0: .2f}({k: .3f})" for k in trades_vec])
                    premium_summary = f"mark={100*(px - S2)/S2: .2f}% slip=[{slippage_summary}](%,amt) "
                    # premium_summary = f"[{100 * (px_max_short - S2) / S2: .2f}({max_short: .2f}) {100 * (px0 - S2) / S2: .2f}(0) {100 * (px_max_long - S2) / S2: .2f}({max_long: .2f})]% "
                    price_summary = f"S2={S2:.1f} "# if perp.collateral_currency is CollateralCurrency.BASE else f"S2={S2:.1f} S3={S3:.1f} "
                    funding_summary = f"{M_label}={M:.2f} ({ 100 * (M / perp.amm_pool_target_size if perp.amm_pool_target_size > 0 else np.inf):3.0f}%) "
                    
                    print(
                        f"{symbol} " \
                        + premium_summary + price_summary + funding_summary \
                        + f"traders: {perp.get_num_active_traders()}/{total_num_traders[symbol]}"
                    )
            
            new_noise_traders = 0
            new_momentum_traders = 0
            for _ in range(GROWTH_MULTIPLIER[symbol]):
                trader_joins = np.random.uniform() < NUM_TRADERS[symbol] / (0.7 * time_df.shape[0]) and t < 0.7 * time_df.shape[0]
                if not trader_joins: continue
                
                noise_trader = 1 if np.random.uniform() < 0.8 else 0 #total_noise_traders[symbol] / (total_noise_traders[symbol] + total_momentum_traders[symbol]))
                new_noise_traders += noise_trader
                new_momentum_traders = 1 - noise_trader

            if new_noise_traders + new_momentum_traders > 0:
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
                        cash_qc=sim_params[symbol]['cash_per_trader_qc'],
                        arb_cash_qc=usd_per_bot,
                        prob_long=prob_long,
                        prob_staking=prob_staking))

        logging_ends = time.time()
        avg_logging_time += logging_ends - logging_starts
        num_logging_measurements += 1

        # overall pool status
        if t % 15000 == 0 or t == time_df.shape[0] - 1:
            S3 = perp.get_collateral_price()
            delta_t = (datetime.now() - run_t0).total_seconds() / 60
            arb_pnl_cc = sum([sim_params[symbol]['arb_pnl'] for symbol in index_vec]) / idx_s3[t]
            # arb_locked_usd = sum([sim_params[symbol]['arb_capital_used'] for symbol in index_vec])
            # arb_active = sum([sim_params[symbol]['num_arb_traders'] for symbol in index_vec])
            traders_active = sum([perp.get_num_active_traders() for perp in amm.perpetual_list])
            total_traders = sum([total_num_traders[symbol] for symbol in index_vec])
            DF = amm.default_fund_cash_cc
            msg = (
                f"{COLL} pool @ {time_df[t]}: "\
                + f"S3={S3:.1f} "\
                + f"stakers = {amm.staker_cash_cc:.3f} " \
                + f"df = {amm.default_fund_cash_cc:.3f} ({100*amm.get_default_fund_gap_to_target_ratio():.1f}%), "\
                + f"liquidations = {amm.liquidator_earnings_vault:.3f} ({liquidation_count}) "\
                + f"bankrupcies = {sum([sim_params[symbol]['num_replaced_traders'] for symbol in index_vec])}, "\
                # + f"arbitrage = {arb_pnl_cc:.3f} "\
                + f"traders = {traders_active}/{total_traders}"
                # + f"fees = {amm.fees_earned:.3f} "\
                #+ f"(${arb_locked_usd / np.max((arb_active, 1)):.0f} locked per bot, ${arb_locked_usd:.0f} total)"
            )
            # print("".join(("-" for _ in range(100))))
            print(msg)
            print(f"PnL by perp (in {COLL}):\t" + "\t".join([f"{name}: {amm.earnings[i]: .3f}" for i, name in enumerate(index_vec)]))
            print(amm_state.iloc[t,:])
            if t > 0:
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

        if (t - last_withdrawal) > protocol_profit_withdrawal_frequency and t > protocol_profit_withdrawal_freeze_period:
            pnl_before = amm.protocol_earnings_vault
            fx_q2c = 1 / amm.perpetual_list[0].get_collateral_price()
            floor_cc = protocol_profit_withdrawal_floor * fx_q2c
            cap_cc = protocol_profit_withdrawal_cap * fx_q2c
            amm.withdraw_profit(protocol_profit_withdrawal_rate, floor_cc, cap_cc)
            pnl_after = amm.protocol_earnings_vault
            if pnl_after > pnl_before:
                profit = pnl_after - pnl_before
                # reinvest in liq prov?
                # amm.staker_cash_cc += profit
                # print(f"Protocol owner withraws {profit:.2f} {COLL} @ {time_df[t]}")
            last_withdrawal = t

    end_dt = datetime.fromtimestamp(to_date)
    delta_t = (datetime.now() - run_t0).total_seconds() / 60
    print(f"Simulation completed in {delta_t:.1f} minutes.\n")
    for i, symbol in enumerate(index_vec):
        perp_idx = perps[i]
        date_appendix = f"_{end_dt.year}{end_dt.month}{end_dt.day}-{run_hash}"
        file_name = f"res{total_num_traders[symbol] - total_arb_traders[symbol]}-{total_arb_traders[symbol]}-{symbol}{COLL}{date_appendix}"
        path_to_store = "results/" + file_name + ".csv"
        inc = 1
        while os.path.isfile(path_to_store):
            path_to_store = "results/" + file_name + "-" + str(inc) + ".csv"
            inc += 1
        df = pd.concat([amm_state, state_dict[perp_idx]], axis=1)
        df.to_csv(path_to_store)
        #state_df.to_pickle('results/'+file_name+'.pkl')
        print(f"{symbol}:\t{path_to_store}")
        # try:
        #     plot_analysis(pd.read_csv(path_to_store), symbol[:(len(symbol) - len(QUOTE))])
        # except:
        #     print("Something went wrong")
    return amm_state


def load_perp_params(symbol, collateral):
    perp_params = dict()
    params_pattern = f"./data/params/{symbol}{collateral}.json"
    params_files = glob.glob(params_pattern)
    if len(params_files) < 1:
        quit(f"Could not find a parameter file for this symbol/collateral combination: {params_pattern} not found.")
    with open(params_files[0]) as json_file:
        perp_params = json.load(json_file)

    perp_params['r'] = 0 # always
    perp_params['initial_margin_cash_cc'] = 0
    perp_params['initial_staker_cash_cc'] = 0
    
    return perp_params

def save_perp_params(symbol, collateral, perp):
    return
    params = perp.export()['params']
    filename = f"./data/params/{symbol}{collateral}PerpetualParams.ts"
    f = open (filename, "w")
    f.write(params)
    f.close()

def init_index_data(from_date, to_date, reload=False, symbol='BTCUSD', collateral='BTC'):
    if not type(symbol) is list:
        symbol = [symbol]
    df_idx_s2 = dict()
    for sym in symbol:
        df_idx_s2[sym] = load_price_data(sym, source="chainlink", network=NETWORK[sym], from_date=from_date, to_date=to_date)
        df_idx_s2[sym].rename(columns={'price': sym}, inplace=True)
    
    if collateral in ['XUSD', 'ZUSD', 'USDT', 'USDC', 'USD']:
        df_idx_s3 = df_idx_s2[symbol[0]].copy()
        df_idx_s3.rename(columns={symbol[0]: 'price'}, inplace=True)
        df_idx_s3['price'] = 1
    else:
        df_idx_s3 = load_price_data(f"{collateral}USD", source="chainlink", network=NETWORK[f"{collateral}USD"], from_date=from_date, to_date=to_date)
        # df_idx_s3 = load_price_data(f"{collateral}USD", source="chainlink", network="BSC_Mainnet", from_date=from_date, to_date=to_date)
    df_idx_s3.rename(columns={'price': 's3'}, inplace=True)

    # TODO: load actual exchange data, not index price
    df_perppx = dict()
    for sym in symbol:
        df_perppx[sym] = load_price_data(sym, source="chainlink", network=NETWORK[sym], from_date=from_date, to_date=to_date)
        # df_perppx[sym] = load_price_data(sym, source="chainlink", network="BSC_Mainnet", from_date=from_date, to_date=to_date)
        df_perppx[sym].rename(columns={'price': sym + '_perp'}, inplace=True)
    
    # combine data
    # print("Combining data...")
    df = df_idx_s3.copy()
    for sym in symbol:
        df = pd.merge(df, df_idx_s2[sym], on=['t', 'timestamp'], how='outer')
        df = pd.merge(df, df_perppx[sym], on=['t', 'timestamp'], how='outer')
        df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.fillna(method='ffill', axis=0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # print("Data successfully loaded:")
    # print(df.head(3))
    # print("...")
    # print(df.tail(3))
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

def init_data(filename, from_date, to_date):
    # TODO: remove this function (it's obsolete in this script but it might be used in integration testing scenario generation?)
    df_px = pd.read_csv(filename)
    if not 'timestamp' in df_px.columns and not 'timeStamp' in df_px.columns:
        date_col = pd.to_datetime(df_px['time'], format='%Y-%m-%d %H:%M:%S.%f', utc=True)
        df_px['time'] = date_col.dt.strftime("%y-%m-%d %H:%M")
        df_px['timestamp'] = date_col.apply(datetime.timestamp)
    elif 'timestamp' in df_px.columns:
        pass # this is obsolete now
        df_px['timestamp'] /= 1000
    else:
        df_px['timestamp'] = df_px['timeStamp']
    if not from_date is None:
        df_px = df_px[df_px['timestamp']>=from_date]
    if not to_date is None:
        df_px = df_px[df_px['timestamp']<=to_date]
    # keep only full minutesif not from_date is None:
        df_px = df_px[df_px['timestamp']>=from_date]
    if not to_date is None:
        df_px = df_px[df_px['timestamp']<=to_date]
    mask = df_px['timestamp'] % 60 == 0
    df = df_px[mask]

    return df


def format_time(df, from_date, to_date):
    if 'timestamp' not in df.columns:
        quit("Case not supported - data should contain a timestamp column")
        if 'time' in df.columns:
            date_col = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S').dt.ceil('min')
            # df['time'] = date_col.dt.strftime("%y-%m-%d %H:%M")
            df['timestamp'] = date_col.apply(datetime.timestamp)
        elif 'timeStamp' in df.columns:
            df['timestamp'] = df['timeStamp']
        else:
            quit("Case not implemented: data should either have a 'timestamp' (w/ milisecs), 'timeStamp' (w/o milisecs) or 'time' (date) column")
    else:
        # df['timestamp'] /= 1000
        pass
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
        df = df[df['timestamp']>=from_date]
    if not to_date is None:
        df = df[df['timestamp']<=to_date]
    df.sort_values('timestamp', inplace=True)
    return df[['t', 'timestamp', 'price']]


def load_price_data(symbol: str, source: str=None, network: str=None, from_date=None, to_date=None):
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
    #return np.random.gumbel(loc=reference_cash_cc, scale=reference_cash_cc / 2)
    #return np.random.uniform(low=reference_cash_cc / 2, high=2 * reference_cash_cc)
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



def initialize_traders(noise_traders, momentum_traders, arb_traders, perp_idx, bitmex_perp_px, amm, currency, cash_qc=-1, arb_cash_qc=-1, prob_staking=0, prob_long=-1):
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
            is_staker = np.random.uniform() <= prob_staking
            trader_list.append(NoiseTrader(amm, perp_idx, currency, cash_cc=cash_samples[-1] * fx_q2c, daily_trades=num_trades_per_day, is_staker=is_staker, prob_long=prob_long))
            num[0] += 1
        if num[1]<momentum_traders:
            cash_samples.append(draw_cash_sample(cash_qc))
            is_staker = np.random.uniform() <= prob_staking
            trader_list.append(MomentumTrader(amm, perp_idx, currency, cash_cc=cash_samples[-1] * fx_q2c, is_staker=is_staker))
            num[1] += 1
        if num[2]<arb_traders:
            trader_list.append(ArbTrader(amm, perp_idx, currency, bitmex_perp_px, cash_cc=arb_cash_qc * fx_q2c))
            num[2] += 1
    is_one = num[0] + num[1] == 1
    msg = f"Initialized {num[0] + num[1]} trader{'' if is_one else 's'} with {'' if is_one else 'approx.'} ${np.mean(cash_samples):.2f}"
    if not is_one:
        msg = msg + f" each (90% lie in the range ${np.quantile(cash_samples, 0.05):.2f} - ${np.quantile(cash_samples, 0.95):.2f})"
    # print(msg)
    return trader_list


def traders_trade(traders, amm, sim, cex_ts):
    idx_arr = np.random.choice(len(traders), len(traders), replace=False)

    for symbol in index_vec:
        sim[symbol]['num_momentum_traders'] = 0
        sim[symbol]['num_noise_traders'] = 0
        sim[symbol]['num_arb_traders'] = 0
        sim[symbol]['arb_pnl'] = 0
        sim[symbol]['arb_capital_used'] = 0

    # traders trade in random (but np.random.seed-predictable) order    
    num_noise_replaced = 0
    num_momentum_replaced = 0
    # random.shuffle(idx_arr)

    for k in idx_arr:
        trader = traders[k]

        dPos, is_close = trader.query_trade_amount()
        is_active = trader.is_active

        perp_idx = trader.perp_idx
        perp = amm.get_perpetual(perp_idx)
        symbol = index_vec[perp_idx]
        
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
                        sim[symbol]['total_bitmex_btc_vol'] += np.abs(dPos)
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
    

def traders_liquidate(traders, amm, do_print=False, update_sim=True):
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
                if update_sim:
                    sim_params[index_vec[traders[k].perp_idx]]['num_trades'] += 1
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


def record_amm_state(t, state, amm, traders, stakers):
    #  columns=['time', 'idx_s3', 'amm_cash', 'staker_cash', 'df_cash',
    #                 'protocol_earnings_vault', 'df_cash_to_target', 'liquidator_earnings_vault', 'amm_target'])

    state.at[t, 'amm_cash'] = amm.get_amm_funds()
    state.at[t, 'pool_margin'] = sum((p.amm_trader.cash_cc  for p in amm.perpetual_list))
    state.at[t, 'pricing_staked_cash'] = sum((p.get_pricing_staked_cash_for_perp() for p in amm.perpetual_list))
    state.at[t, 'cash_staked'] = amm.staker_cash_cc
    state.at[t, 'staker_cash'] = sum((staker.get_position_value_cc() for staker in stakers))
    state.at[t, 'df_cash'] = amm.default_fund_cash_cc
    # state.at[t, 'trader_cash'] = sum((trader.get_margin_balance_cc() for trader in traders))
    state.at[t, 'share_token_supply'] = amm.share_token_supply
    
    state.at[t, 'protocol_earnings_vault'] = amm.protocol_earnings_vault
    state.at[t, 'liquidator_earnings_vault'] = amm.liquidator_earnings_vault
    state.at[t, 'df_target'] = amm.get_default_fund_gap_to_target() + amm.default_fund_cash_cc
    state.at[t, 'amm_target'] = amm.get_amm_pools_gap_to_target() + state.at[t, 'amm_cash']
    state.at[t, 'df_cash_to_target'] = amm.get_default_fund_gap_to_target()
    state.at[t, 'total_volume_qc'] = sum((p.total_volume * p.get_index_price() for p in amm.perpetual_list))



def record_perp_state(time_idx, state, perp_idx, traders, sim, amm):
    # # record statistics and state
    #  columns=['idx_px', 'mark_premium', 'mid_premium',
    #                 'amm_cash', 'df_target', 'current_trader_exposure_EMA',
    #                 'amm_target',  
    #                 'num_noise_traders', 'num_momentum_traders','num_arb_traders','num_bankrupt_traders', 
    #                 # 'current_AMM_exposure_EMA0','current_AMM_exposure_EMA1','current_trader_exposure_EMA',
    #                 # 'protocol_btc_volume', 'locked_in_qc', 'open_interest',
    #                 # 'current_locked_in_value_EMA0','current_locked_in_value_EMA1', 'total_bitmex_btc_vol', 
    #                 'max_long_trade', 'max_short_trade', 'num_trades', 'max_long_prem', 'max_short_prem', 
    #                 # 'long_1_prem', 'short_1_prem', 'long_10k_prem', 'short_10k_prem',
    #                 'arb_capital_used', 'arb_pnl', 'cex_px'])
    perp = amm.perpetual_list[perp_idx]
    
    t = perp.current_time
    # state.iloc[time_idx, 1] = perp.idx_s2[t]
    state.at[time_idx, 'mark_price'] = perp.get_mark_price()
    state.at[time_idx, 'mid_price'] = 0.5 * (perp.get_price(perp.params['fLotSizeBC']) + perp.get_price(-perp.params['fLotSizeBC']))
    state.at[time_idx, 'funding_rate'] = perp.get_funding_rate()
    # cash
    state.at[time_idx, 'perp_amm_cash'] = perp.amm_pool_cash_cc
    state.at[time_idx, 'perp_pricing_staked_cash'] = perp.get_pricing_staked_cash_for_perp()

    # targets
    state.at[time_idx,'perp_amm_target'] = perp.amm_pool_target_size
    state.at[time_idx,'perp_amm_target_baseline'] = perp.get_amm_pool_size_for_dd(perp.params['fAMMTargetDD'][0])
    state.at[time_idx,'perp_amm_target_stress'] = perp.get_amm_pool_size_for_dd(perp.params['fAMMTargetDD'][1])
    state.at[time_idx,'perp_df_target'] = perp.default_fund_target_size
    state.at[time_idx,'num_noise_traders'] = sim['num_noise_traders']
    state.at[time_idx,'num_momentum_traders'] = sim['num_momentum_traders']
    state.at[time_idx,'num_arb_traders'] = sim['num_arb_traders']
    state.at[time_idx,'num_bankrupt_traders'] = sim['num_bankrupt_traders']

    # aggregates 
    state.at[time_idx, 'perp_volume_qc'] = perp.total_volume * perp.get_index_price()
    # state.at[time_idx, 'open_interest'] = perp.open_interest
    
    state.at[time_idx,'current_trader_exposure_EMA'] = perp.current_trader_exposure_EMA
    state.at[time_idx,'num_trades'] = sim['num_trades']
    
    # AMM margin
    state.at[time_idx, 'amm_margin'] = perp.amm_trader.cash_cc
    state.at[time_idx, 'K2'] = perp.amm_trader.position_bc
    state.at[time_idx, 'L1'] = perp.amm_trader.locked_in_qc

    # EMAs
    # state.at[time_idx, 'L1_EMA-'] = perp.current_locked_in_value_EMA[0]
    # state.at[time_idx, 'L1_EMA+'] = perp.current_locked_in_value_EMA[1]
    # state.iloc[time_idx,'current_AMM_exposure_EMA_0'] = perp.current_AMM_exposure_EMA[0]
    # state.iloc[time_idx,'current_AMM_exposure_EMA_1'] = perp.current_AMM_exposure_EMA[1]

    # arbitrage
    # state.at[time_idx,'arb_pnl'] = sim['arb_pnl']
    # state.iloc[time_idx,25] = sim['total_bitmex_btc_vol']
    
    # trade sizes and premia
    # max
    state.at[time_idx,'max_long_trade'] = perp.get_max_signed_trade_size_for_position(0, 1)
    state.at[time_idx,'max_short_trade'] = perp.get_max_signed_trade_size_for_position(0, -1)
    state.at[time_idx,'max_long_price'] = perp.get_price(state.at[time_idx,'max_long_trade'])
    state.at[time_idx,'max_short_price'] = perp.get_price(state.at[time_idx,'max_short_trade'])
    # $10,000 
    pos = 10_000 / perp.get_index_price()
    state.at[time_idx,'10k_long_price'] = perp.get_price(pos)
    state.at[time_idx,'10k_short_price'] = perp.get_price(-pos)
    # 100,000 lots
    pos = 100_000 * perp.params['fLotSizeBC']
    state.at[time_idx, '100klots_long_price'] = perp.get_price(pos)
    state.at[time_idx, '100klots_short_price'] = perp.get_price(-pos)

    # average
    pos = perp.current_trader_exposure_EMA
    state.at[time_idx,'avg_long_price'] = perp.get_price(pos)
    state.at[time_idx,'avg_short_price'] = perp.get_price(-pos)
    # min
    pos = perp.min_num_lots_per_pos * perp.params['fLotSizeBC']
    state.at[time_idx,'min_long_price'] = perp.get_price(pos)
    state.at[time_idx,'min_short_price'] = perp.get_price(-pos)
    


def getNumTraderVec(id, totalId):
    np.random.seed(42)
    nVecAll = (np.array([10, 20, 40, 50, 60, 80, 100, 120, 150, 180]))
    np.random.shuffle(nVecAll)
    #idx_desma = np.concatenate(([0], np.arange(int(np.ceil(nVecAll.shape[0]/2)) , nVecAll.shape[0], 1)))
    #idx_aws = np.setdiff1d(np.arange(0, nVecAll.shape[0], 1), idx_desma)
    chunk_size = int(np.floor(nVecAll.shape[0]/totalId))
    limits = [(id-1)*chunk_size, np.min((id*chunk_size, nVecAll.shape[0]-1))]
    if limits[1] > nVecAll.shape[0]-chunk_size:
        limits[1] = nVecAll.shape[0]
    return nVecAll[np.arange(limits[0], limits[1])]


def to_readable_fiat(x):
    if x >= 500_000:
        return f"{x / 1_000_000:.2f} million {QUOTE}"
    if x >= 1_000:
        return f"{x / 1_000:.2f} thousand {QUOTE}"
    return f"{x} {QUOTE}"


if __name__ == "__main__":
    # TODO: fix this global variable approach
    all_runs_t0 = datetime.now()
    seeds = [
        124, 
        431,
    ]

    simulation_horizons = [
        #(datetime(2022, 6, 15, 0, 0, tzinfo=timezone.utc), datetime(2022, 10, 15, 0, 0, tzinfo=timezone.utc)), 
        (datetime(2022, 7, 5, 0, 0, tzinfo=timezone.utc), datetime(2022, 10, 6, 0, 0, tzinfo=timezone.utc)), 
    ]

    long_probs = [
        0.45,
        0.55,
    ]

    initial_investments = [
         # 150_000_000,
        # 50_000_000,
          2_100_000,
        #1_500_000,
    ]

    usds_per_trader = [
        3_000,
    ]

    run_configs = itertools.product(seeds, simulation_horizons, long_probs, initial_investments, usds_per_trader)
    total_hash = hash(run_configs)

    filename = f"./results/simulation_{total_hash}.csv"
    results = pd.DataFrame(
        columns=["Start", "End", "Days", "Prob_Long",
                 "Funds_Init", "Funds_Final", "DF_Excess", "Profit", "Liquid_Profit",
                 "LP_Init", "LP_Final", "LP_APY",
                 "Volume", "Profit_per_Volume", "Annualized_Profit"]
    )

    for run_config in run_configs:
        run_hash = hash(run_config)
        seed = run_config[0]
        from_date, to_date = run_config[1]
        prob_long = run_config[2]
        initial_protocol_investment = run_config[3]
        usd_per_trader = run_config[4] 

        print("\n" + "".join(("-" for _ in range(100))) + "\n")
        print(f"Run details:")
        print(f"From {from_date} to {to_date}")
        print(f"Probability of trader taking a long position: {100 * prob_long:.2f}%")
        print(f"Protocol initial funds: {to_readable_fiat(initial_protocol_investment)}")

        from_date, to_date = from_date.timestamp(), to_date.timestamp()
        usd_per_bot = 0 # the total capital needed for arbs is this times the number of bots
        # liq_provider_initial_investment = 1_000 # in dollars
        protocol_profit_withdrawal_frequency = 60 * 24 * 10 # we try to withdraw every n days (seconds)
        protocol_profit_withdrawal_rate = 0.0 # % of the excess (if any) we withdraw ([0,1])
        protocol_profit_withdrawal_floor = 1_000 # the least amount worth withdrawing in one go (QC)
        protocol_profit_withdrawal_cap = 100_000 # the max amount we would withdraw in one go (QC)
        protocol_profit_withdrawal_freeze_period = 60 * 24 * 30 # no withdrawals for the first n days (seconds)
        animate_premia = False
        # usd_per_trader = 2_000 # this is a sort of average, there's random sampling involved
        num_trades_per_day = 1
        prob_staking = 0.90 # prob that trader stakes SOV and pays zero fees / not used currently

        # 25 x 20k = 500k total cash brought in by stakers (randomized)
        # they deposit about once per month (randomized)
        # they withdraw after exactly 1.5 months (not randomized)
        num_stakers = 25
        cash_per_staker = 4_000 # 20 for 500k, 8k for 200k, 6k for 150k, 4k for 100k, 2k for 50k
        monthly_stakes = 1.5
        holding_period_months = 1.5

        sim_params = dict()
        
        df = main()
        n = df.shape[0]

        res = {
            "Start": df.at[0,'time'], 
            "End": df.at[n-1, 'time'],
            "Days": float((pd.to_datetime(df.at[n-1,'time'], format='%y-%m-%d %H:%M') - pd.to_datetime(df.at[0,'time'], format='%y-%m-%d %H:%M')).days),
            "Prob_Long": prob_long,
            "Funds_Init": df.at[0, "df_cash"] + df.at[0, 'amm_cash'] + df.at[0, 'pool_margin'], 
            "Funds_Final": df.at[n-1, "df_cash"] + df.at[n-1, 'amm_cash'] + df.at[n-1, 'pool_margin'], 
            "DF_Excess": -df.at[n-1, "df_cash_to_target"], 
            "LP_Init": df.at[0, "staker_cash"], 
            "LP_Final": df.at[n-1, "staker_cash"], 
            "Volume": df.at[n-1, "total_volume_qc"],
        }
        res["Profit"] = res["Funds_Final"] - res["Funds_Init"]
        res["Liquid_Profit"] = np.min((res["Profit"], np.max((0, res["DF_Excess"]))))
        res["Annualized_Profit"] = res["Profit"] * 365 / res["Days"]
        res["LP_APY"] = (res["LP_Final"] / res["LP_Init"] - 1) * 365 / res["Days"]
        res["Profit_per_Volume"] = res["Profit"] / res["Volume"]
       
        results.loc[results.shape[0]] = res

        print(res)
        

    delta_t = (datetime.now() - all_runs_t0).total_seconds() / 60
    results.to_csv(filename)
    print(results)
    print(f"Finished! And it only took {delta_t / 60:.1f} hours...")