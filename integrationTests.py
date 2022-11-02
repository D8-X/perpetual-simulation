#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#%%  -------------------
from pandas.core.frame import DataFrame
from arb_trader import ArbTrader
from trader import Trader, CollateralCurrency
from noise_trader import NoiseTrader
from momentum_trader import MomentumTrader
from amm import AMM
import numpy as np
import pandas as pd 
from datetime import datetime, date, timezone
import os.path
from timeit import default_timer as timer
import pickle
import platform
import simulation
import json
from matplotlib import pyplot as plt


def calculateLiquidationPriceCollateralBase(LockedInValueQC, position, cash_cc, maintenance_margin_ratio):
    #correct only if markprice = spot price
    return LockedInValueQC/(position-maintenance_margin_ratio*np.abs(position) + cash_cc)

def create_price_scenario(filename = ""):
    from_date = datetime(2020, 11, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    to_date = datetime(2020, 11, 1, 1, 0, tzinfo=timezone.utc).timestamp()
    
    (idx_px, bitmex_px, time_df) = simulation.init_index_data(from_date, to_date, reload=True)
    # manipulate data
    idx_px[7] = idx_px[6]*(1-0.1)
    idx_px[8] = idx_px[7]*(1-0.05)
    idx_px[9] = idx_px[8]*(1-0.025)
    idx_px[10] = idx_px[10]*(1-0.1)
    idx_px = idx_px[0:30]
    if filename!="":
        df = pd.DataFrame(idx_px)
        df.columns = ['priceIndex']
        df.to_csv(filename)
        
    return idx_px

def fill_dict(mydict, df, category):
    df_filtered = df[df['category']==category]
    for k in range(df_filtered.shape[0]):
        values = [df_filtered.iloc[k,1],df_filtered.iloc[k,2]]
        if np.isnan(values[1]):
            values = values[0]
        mydict[df_filtered.iloc[k,0]] = values

def load_index_data(scenario_folder):
    df = pd.read_csv(scenario_folder+"/PriceScenario.csv")
    return df["priceIndex"].to_numpy()

def get_params(scenario_folder):
    df = pd.read_csv(scenario_folder+"IntegrationTestParameters.csv")
    perp_params = dict()
    params = dict()
    endowment = dict()
    fill_dict(perp_params, df, 'perp_params')
    fill_dict(params, df, 'params')
    fill_dict(endowment, df, 'endowment')
    perp_params['initial_df_cash_cc'] = 0
    return params, perp_params, endowment

def round_to_lot(value, lot_size):
    if value<0:
        return np.ceil(value/lot_size)*lot_size
    else:
        return np.floor(value/lot_size)*lot_size

def get_max_trade_size(trade_amount, current_pos, amm, perp_idx):
    # calculate largest trade size not considering leverage
    perp = amm.get_perpetual(perp_idx)
    amt = perp.get_max_signed_trade_size_for_position(current_pos, trade_amount)
    amt = amm.shrink_to_lot(amt, perp.params['fLotSizeBC'])
    assert(amt == 0 or np.abs(amt) >= perp.params['fLotSizeBC'])
    #if np.abs(amt) < amm.params['fLotSizeBC']:
        # amt = amm.shrink_to_lot(amt)
        # if np.abs(amt) < amm.params['fLotSizeBC'] and np.abs(amt) > 0:
        #     print("Something wrong with shrink_to_lot")
        #     amt = 0
    return amt

# maxTradeSizeFromContract= -0.000100931160166099
# fLotSizeBC,0.00001,

if __name__ == "__main__":
    scenario_root = './test_scenarios'
    scenario_name = 'scenario1'
    scenario_folder = scenario_root+"/"+scenario_name+"/"
    #%%  -------------------
    #idx_px = create_price_scenario(filename="./test_scenarios/scenario1/price_scenario1.csv")
    idx_px = load_index_data(scenario_folder)
    plt.plot(idx_px,'-x')
    print("Index Price S2 (Oracle):")
    print(idx_px)

    #%%  -------------------
    # scenario parameters
    params, perp_params, endowment = get_params(scenario_folder)
    initial_staker_cash = endowment['initial_staker_cash']
    initial_default_fund_cash_cc = endowment['initial_df_cash_cc']
    # perp_params['initial_df_cash_cc'] = 0
    initial_amm_cash = endowment['initial_amm_cash']
    initial_margin_cash = endowment['initial_margin_cash']
    trader_collateral_cc = endowment['trader_collateral_cc']

    # Initialize AMM
    amm = AMM(params, 0, initial_default_fund_cash_cc)
    # add perpetual BTCUSD twice, amm thinks one is base and the other is quanto
    perp_idx = amm.add_perpetual(initial_amm_cash, initial_margin_cash, idx_px, None, perp_params, CollateralCurrency.BASE)
    perp_idx2 = amm.add_perpetual(initial_amm_cash, initial_margin_cash, idx_px, idx_px, perp_params, CollateralCurrency.QUANTO)
    #%% initialize traders -------------------
    # traderNo, time, tradePos
    # amm.export()
    trade_schedule = pd.read_csv(scenario_folder+"ScheduleTraders.csv")
    num_traders = 2 * (np.max(trade_schedule['traderNo'])+1)
    traders=[]
    for k in range(num_traders // 2):
        traders.append(NoiseTrader(amm, perp_idx, CollateralCurrency.BASE, trader_collateral_cc))
        traders.append(NoiseTrader(amm, perp_idx2, CollateralCurrency.QUANTO, trader_collateral_cc))

    #%% explore margin/liquidation
    dir = 1
    cash_cc = traders[0].cash_cc
    # pos = dir*amm.get_max_leverage_pos(cash_cc)
    # LockedInValueQC = pos*idx_px[0]
    # maintenance_margin_ratio = amm.get_maintenance_margin_rate(pos)
    # S = calculateLiquidationPriceCollateralBase(LockedInValueQC, pos, cash_cc, maintenance_margin_ratio)
    # print("Price =", idx_px[0])
    # print("S liquidation=", S)
    # print("return=", S/idx_px[0]-1)


    # [timeindex, number of active traders, [traderno/AMM, cash, lockedIn, position], defaultfund, ammfund, participationFund]
    result_data = []
    num_active_traders = num_traders
    #%% run simulation
    for t in range(idx_px.shape[0]):
        print(f"t = {t}\t S2 = {idx_px[t]:.1f}\t ret(S2) = {0 if t == 0 else 100*(idx_px[t] / idx_px[t-1] - 1):.3f}%, emergency={amm.is_emergency}, amm cash gap={amm.get_amm_pools_gap_to_target()}")
        print(f"AMM fund before trades: DF={amm.default_fund_cash_cc}, Cash={amm.get_amm_funds()}, Staker cash={amm.staker_cash_cc}")
        current_schedule = trade_schedule[trade_schedule["time"]==t]
        num_liquidated = simulation.traders_liquidate(traders, amm, True, False)
        # list for output:
        trader_status = []
        
        for kk in range(2 * current_schedule.shape[0]):
            k = kk // 2
            if kk % 2 == 0:
                idx = current_schedule.iloc[k].traderNo
                perp = amm.get_perpetual(perp_idx)
            else:
                idx = 2 * current_schedule.iloc[k].traderNo
                perp = amm.get_perpetual(perp_idx2)
            idx = current_schedule.iloc[k].traderNo
            dir = current_schedule.iloc[k].tradePos
            cash = traders[idx].cash_cc
            current_pos = traders[idx].position_bc
            L = traders[idx].locked_in_qc
            trade_amount = 0
            max_lvg_pos = amm.shrink_to_lot(np.sign(dir)*perp.get_max_leverage_position(traders[idx]), perp.params['fLotSizeBC'])
            k_star = perp.get_Kstar()
            margin_balance = traders[idx].get_margin_balance_cc(perp, at_mark=False)
            M2before = perp.amm_pool_cash_cc
            K2before = -perp.amm_trader.position_bc
            kema = perp.current_trader_exposure_EMA
            df_before = perp.default_fund_target_size
            if current_pos != 0 and np.sign(dir) != np.sign(current_pos):
                # close the trade
                trade_amount = -current_pos
                # max_trade_amount = get_max_trade_size(trade_amount, current_pos, amm, perp_idx)
                # trade_amount = np.sign(trade_amount) * np.min((np.abs(trade_amount), np.abs(max_trade_amount)))
                # trade_amount /= 10 # a tenth of the max
                # trade_amount = round_to_lot(trade_amount, amm.params['fLotSizeBC'])
                print("time ", t, " close position for trader ", idx, " pos=", current_pos)
                px = traders[idx].trade(trade_amount, True)
            else:
                # open with max size allowed by leverage
                trade_amount = max_lvg_pos
                max_trade_amount = get_max_trade_size(trade_amount, current_pos, amm, perp_idx)
                trade_amount = np.sign(trade_amount) * np.min((np.abs(trade_amount), np.abs(max_trade_amount) / 10))
                # trade_amount /= 10 # a tenth of the max
                trade_amount = amm.shrink_to_lot(trade_amount, perp.params['fLotSizeBC']) #, amm.params['fLotSizeBC'])
                print("time ", t, " open position for trader ", idx, " pos=", trade_amount)
                #print(f"max amount before trader.trade: {get_max_trade_size(trade_amount, current_pos, amm, perp_idx)}")
                px = traders[idx].trade(trade_amount, False)
            if px is None:
                print(f"Trade not executed: {traders[idx].position_bc - current_pos}")
            else:
                print(f"New position: {traders[idx].position_bc - current_pos} at price {px}")
            trader_status.append(
                {
                    "traderNo": int(idx), 
                    "cashBefore": cash, 
                    "lockedInBefore": L, 
                    "positionBefore": current_pos,
                    "tradeAmount": traders[idx].position_bc - current_pos,
                    "traderMarginBalanceBefore": margin_balance,
                    "traderMarginBalanceAfter": traders[idx].get_margin_balance_cc(perp, at_mark=False),
                    "maxTradeAmount": max_trade_amount,
                    "maxLvgPosition": max_lvg_pos,
                    "kStarBefore": k_star,
                    "kStarAfter": perp.get_Kstar(),
                    "M2before": M2before,
                    "K2before": K2before,
                    "M2after": perp.amm_pool_cash_cc,
                    "K2after": -perp.amm_trader.position_bc,
                    "kEMAbefore": kema,
                    "kEMAafter": perp.current_trader_exposure_EMA,
                    "DFTargetSizeBefore": df_before,
                    "DFTargetSizeAfter": perp.default_fund_target_size,
                    "isEmergency": amm.is_emergency,
                }
            )
        num_active_traders = num_active_traders - num_liquidated

        if amm.is_emergency:
            print("AMM emergency state reached")
            break
        amm.inc_time()

        if len(trader_status) > 0:
            result_data.append(
                {
                    "timeindex": int(t),
                    "activeTraders": int(num_active_traders),
                    "trades": trader_status,
                    "defaultfund": amm.default_fund_cash_cc,
                    "ammfund": amm.get_amm_funds(),
                    "participationFund": amm.staker_cash_cc,
                }
            )
        print(f"AMM fund after trades: DF={amm.default_fund_cash_cc}, Cash={amm.get_amm_funds()}, Staker cash={amm.staker_cash_cc}")
    print(json.dumps(result_data, indent=2))
    with open (scenario_folder+"ResultData.json", "w") as f:
        #json.dump(result_data, f)
        json.dump(result_data, f, indent=2)
        f.close()