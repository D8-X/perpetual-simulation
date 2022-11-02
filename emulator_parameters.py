# parameter for emulator
import numpy as np
from noise_trader import NoiseTrader
import simulation

def set_parameters():
    base = "BTC"

    # amm parameters
    amm_params = dict()
    amm_params['initial_staker_cash_cc'] = 0.02 
    amm_params['initial_default_fund_cash_cc'] = 0
    amm_params['fMarginRateBeta'] = 0.10
    amm_params['fInitialMarginRateCap'] = 0.10
    amm_params['fMaintenanceMarginRateAlpha'] = 0.04
    amm_params['fInitialMarginRateAlpha'] = 0.06
    
    amm_params['mark_price_ema_lambda'] = 0.70
    
    amm_params['ceil_staker_pnl_share'] = 0.80
    amm_params['DF_lambda'] = [0.999, 0.50]

    amm_params['funding_rate_clamp'] = 0.0005
    amm_params['target_pool_size_update_time'] = 86400/3
    amm_params['block_time_sec'] = 3
    
    amm_params['protocol_fee_rate'] = 0.05/100
    amm_params['protocol_fee_rate_extra'] = 0.00/100
    amm_params['LP_fee_rate'] = 0.03/100
    amm_params['liquidation_penalty_rate'] = 0.05
    amm_params['AMM_lambda'] = 0 # not used

    perp_params = dict() # dict of dicts
    perp_params[base] = dict()
    # params that are shared by all the perps with this collateral
    perp_params[base]['sig3'] = 0.1 # approx 2 * np.std(np.diff(np.log(idx_s3[::60*24])))
    perp_params[base]['r'] = 0 # always
    perp_params[base]['initial_margin_cash_cc'] = 0
    perp_params[base]['initial_staker_cash_cc'] = 0
    perp_params[base]['cover_N'] = 0.05 #5% of #total traders; parameter for default fund size

    #if base == "BTC":
    perp_params[base]['sig2'] = 0.1 # approx 2 * np.std(np.diff(np.log(idx_s2[::60*24])))
    perp_params[base]['rho23'] = 1 
    perp_params[base]['stress_return_S2'] = [-0.5, 0.2] # used for default fund size
    perp_params[base]['stress_return_S3'] = [-0.5, 0.2] # has to be the same as S2 in this case (S3==S2)
    # amm pool params
    perp_params[base]['amm_min_size'] = 1 # 1 BTC ~ 50K USD
    perp_params[base]['amm_baseline_target_dd'] = -2.9677379253417833 # 15 bps
    perp_params[base]['amm_stress_target_dd'] =  -2.4323790585844467 # 75 bps
    # 0.0015 : -2.9677379253417833
    # 0.0025 : -2.8070337683438042
    # 0.0050 : -2.575829303548901
    # 0.0075 : -2.4323790585844467
    # 0.0100 : -2.3263478740408408
    perp_params[base]['fMinimalAMMExposureEMA'] = 1 # this is BTC
    # trading params
    # trade/position size related
    perp_params[base]['fMinimalTraderExposureEMA'] = 0.2 # 0.1 BTC ~ 5_000 USD
    perp_params[base]['tradeSizeBumpUp'] = 1.1 # EMA + 50%
    # minimal premium
    perp_params[base]['fMinimalSpread'] = 0.0005 # 10 bps bid-ask (when fully funded)
    perp_params[base]['fMinimalSpreadInStress'] = 0.0010 # 20 bps bid-ask (when not fully funded) // MCDEX: >25 bps
    # lot size
    perp_params[base]['fLotSizeBC'] = 0.002 # ~ 100 USD
    # distribute initial cash
    # 10% of assigned cash
    perp_params[base]['initial_amm_cash_cc'] = 3
    # the rest goes to DF
    perp_params[base]['initial_df_cash_cc'] = 5
    return [perp_params, amm_params]

def initialize_traders(noise_traders, perp_idx, amm, price, cash_qc=10_000):
    trader_list=[]
    num = 0
    fx_q2c = 1 / price
    assert(not np.isnan(cash_qc))
    assert(not np.isnan(fx_q2c))
    cash_samples = []
    for k in range(noise_traders):
        cash_samples.append(simulation.draw_cash_sample(cash_qc))
        trader_list.append(NoiseTrader(amm, perp_idx, 1, cash_cc=cash_samples[-1] * fx_q2c))
        trader_list[k].prob_trade = np.random.uniform(1/10, 1/3)
        trader_list[k].prob_long = 0.55

    print(f"Initialized {num} traders with approx. ${np.mean(cash_samples):.2f} each (90% lie in the range ${np.quantile(cash_samples, 0.05):.2f} - ${np.quantile(cash_samples, 0.95):.2f})")
    return trader_list