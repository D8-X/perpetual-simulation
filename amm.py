#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# AMM class
# 

import numpy as np
from amm_trader import AMMTrader
from committed_staker import CommittedStaker
from perpetual import Perpetual
from trader import CollateralCurrency, Trader

class AMM:

    def __init__(self,
                params : dict,
                initial_default_fund_cash_cc : float=0, t0=0) -> None:

        self.current_time = 0
        self.t0_timestamp = t0
        self.params = params
        self.perpetual_list = []

        # liquidity provision        
        self.share_token_supply = 0
        self.staker_cash_cc = 0
        self.staker_pricing_cash_ratio = 0 # 'P'
        self.last_pricing_cash_update = 0
        self.fund_transfer_convergence_time = 60 * 60 * params['iFundTransferConvergenceHours'] # in seconds
        self.max_PtoDF_ratio = np.inf # 0.75 
        # how much in CC ccy can the pot of staker cash used in pricing increase in a given incorporation cycle (one day above)?
        self.max_liquidity_inc_delta = params['fMaxTransferPerConvergencePeriodCC'] 

        # protocol stuff
        self.default_fund_cash_cc = initial_default_fund_cash_cc
        self.protocol_earnings_vault = 0
        self.is_emergency = False
        self.last_target_pool_size_update = -params["iTargetPoolSizeUpdateTime"]/2 # in seconds
        self.liquidator_earnings_vault = 0
        self.trade_count = 0
        self.fees_earned = 0
        self.trader_dir = []
        self.lp_cc_apy = []
        self.earnings = dict()
        self.funding_earnings = dict()
        self.attacker_pnl = 0



        # self.protocol_balance_cc = 0
        # this is just for showing on screen
        self.staker_return_ema_lambda = 1 / 30 / 100
        self.staker_return_ema = 0

    def set_emergency(self):
        self.is_emergency = True
        for p in self.perpetual_list:
            p.set_emergency_state()

    def get_perpetual(self, idx) -> Perpetual:
        return self.perpetual_list[idx]
    
    def get_total_user_cash(self):
        return sum((trader.cash_cc for trader in self.trader_dir if not isinstance(trader, AMMTrader)))

    def get_total_protocol_cash(self):
        amm_cash = 0
        for p in self.perpetual_list:
            amm_cash += p.amm_pool_cash_cc + p.amm_trader.cash_cc
        return self.default_fund_cash_cc + self.staker_cash_cc + amm_cash
        
    def add_perpetual(self, initial_amm_cash:float, initial_margin_cash: float,
                     idx_s2:np.array, idx_s3 :np.array, perp_params : dict, cc: CollateralCurrency,
                     **kwargs):
        idx = len(self.perpetual_list)
        perpetual = Perpetual(self, initial_amm_cash, initial_margin_cash, idx_s2, idx_s3, perp_params, self.params, cc, idx, **kwargs)
        if "initial_df_cash_cc" in perp_params.keys():
            self.default_fund_cash_cc += perp_params['initial_df_cash_cc']
        self.perpetual_list.append(perpetual)
        self.earnings[idx] = 0
        self.funding_earnings[idx] = 0
        return idx

    def inc_time(self):
        self.current_time = self.current_time + 1
        for p in self.perpetual_list:
            p.inc_time()

    def get_amm_funds(self) -> float:
        funds = 0
        for p in self.perpetual_list:
            funds += p.amm_pool_cash_cc
        return funds

    def get_amm_pools_target(self) -> float:
        target = 0
        for p in self.perpetual_list:
            target += p.amm_pool_target_size_ema
        return target 
    
    def get_amm_pools_gap_to_target(self) -> float:
        """ missing funds to reach AMM pool target size
        Returns:
            float: missing funds. Negative if excess funds
        """
        return self.get_amm_pools_target() - self.staker_cash_cc

    def get_default_fund_gap_to_target(self) -> float:
        """ missing funds to reach default fund target size
        Returns:
            float: missing funds. Negative if excess funds
        """
        total_target = 0
        for p in self.perpetual_list:
            total_target = total_target + p.default_fund_target_size
        lp_contr = np.max((0,self.staker_cash_cc - self.get_amm_pools_target()))
        total_target = np.max((0,total_target - lp_contr))
        gap = total_target - self.default_fund_cash_cc
        return gap

    def get_default_fund_gap_to_target_ratio(self) -> float:
        """ missing funds to reach default fund target size
        Returns:
            float: funding ratio. >1 if excess funds
        """
        total_target = 0
        for p in self.perpetual_list:
            total_target = total_target + p.default_fund_target_size
        funds = self.default_fund_cash_cc
        # CHANGE
        amm_target = self.get_amm_pools_target()
        if self.staker_cash_cc > amm_target:
            funds += self.staker_cash_cc - amm_target
        gap_ratio = funds/total_target
        # if gap_ratio < 0.01:
        #     self.is_emergency = True
        return gap_ratio
    
    def record_apy(self, apy):
        self.lp_cc_apy.append(apy)
        
    def __update_DF_size_target(self):
        for p in self.perpetual_list:
            p.update_DF_size_target()
        
    def __update_AMM_pool_size_target(self):
        for p in self.perpetual_list:
            p.update_AMM_pool_size_target()

    def get_timestamp(self):
        # time elapsed in seconds, we start after one block
        return self.t0_timestamp + (self.current_time)*self.params["block_time_sec"]

    def __update_target_pool_sizes(self):
        # frequent update of AMM pool size
        # self.__update_AMM_pool_size_target()
        # rare update of default fund size
        ts_now = self.get_timestamp()
        dT = ts_now - self.last_target_pool_size_update
        if dT < self.params['iTargetPoolSizeUpdateTime']:
            return
        self.__update_DF_size_target()
        self.last_target_pool_size_update = ts_now

    def trade_with_amm(self, trader : 'Trader', amount_bc : float, is_close_only: bool):
        perp_idx = trader.perp_idx
        perpetual = self.perpetual_list[perp_idx]
        # try to trade
        px = perpetual.trade(trader, amount_bc, is_close_only)
        if not px:
            # trade failed
            return None
        # trade was successful, keep track of trader status
        perpetual.trader_status[trader.id] = trader.is_active and trader.position_bc != 0
        self.trade_count += 1
        # update default fund size and AMM pool size targets
        self.__update_target_pool_sizes()
        return px

    def round_to_lot(self, amount_bc, lot):
        """ rounds to lot (downwards in absolute value): 
        when booking a trade, any amount left over the highest lot-amount is not used"""
        rounded = np.round(np.abs(amount_bc) / lot) * lot
        assert(np.abs(rounded - np.abs(amount_bc)) <= lot)
        return rounded if amount_bc > 0 else -rounded

    def shrink_to_lot(self, amount_bc, lot):
        """ rounds to lot (downwards in absolute value): 
        when booking a trade, any amount left over the highest lot-amount is not used"""
        rounded = np.floor(np.abs(amount_bc) / lot) * lot
        return rounded if amount_bc > 0 else -rounded

    def grow_to_lot(self, amount_bc, lot):
        # lot =  self.params['fLotSizeBC']
        rounded = int(np.abs(amount_bc) / lot + 1) * lot
        assert(rounded >= np.abs(amount_bc))
        return rounded if amount_bc > 0 else -rounded

    def export(self, filename=None):
        response = dict()
        response['iPerpetualCount'] = len(self.perpetual_list)
        response['fDefaultFundCashCC'] = self.default_fund_cash_cc
        response['fPnLparticipantsCashCC'] = self.staker_cash_cc
        response['perpetuals'] = [perp.export() for perp in self.perpetual_list]
        # if filename:
        #     if len(filename) < 5 or not filename[:-5] == '.json':
        #         filename = filename + '.json'
        #     with open (filename, "w") as f:
        #         json.dump(response, f, indent=2)
        #         f.close()
        return response
    
    def withdraw_profit(self, rate, floor, cap):
        df_excess = np.max((-self.get_default_fund_gap_to_target(), 0))
        available_for_withdrawal = df_excess if df_excess >= floor else 0
        if available_for_withdrawal > 0:
            withdrawal = np.min((cap, rate * available_for_withdrawal))
            if withdrawal > 0:
                print(f"Governance withdraws from DF excess: {withdrawal:.4f}")
                self.protocol_earnings_vault += withdrawal
                self.default_fund_cash_cc -= withdrawal

    def add_liquidity(self, staker, cash_cc):
        if cash_cc <= 0:
            return
        if cash_cc > staker.cash_cc:
            cash_cc = staker.cash_cc
        
        # mint tokens
        if self.share_token_supply == 0:
            # first staker gets share tokens 1:1 with collateral token
            tokens = cash_cc
        else:
            # new stakers get tokens so that their share of the resulting total supply equals
            # their share of cash in the resulting total staker cash at the time of deposit
            # tokens / (total_supply_before + tokens) = cash / (total_cash_before + cash)
            # --> tokens = cash * total_supply_before / total_cash_before
            tokens = (cash_cc / self.staker_cash_cc) * self.share_token_supply

        self.share_token_supply += tokens
        staker.share_tokens += tokens
        # distribute cash
        staker.cash_cc -= cash_cc
        self.staker_cash_cc += cash_cc
        self.last_pricing_cash_update = self.get_timestamp()

    def remove_liquidity(self, staker, tokens):
        if tokens <= 0 or self.share_token_supply <= 0:
            return
        if tokens >= staker.share_tokens:
            tokens = staker.share_tokens
        # assert(tokens > 0 and tokens <= staker.share_tokens)
        # assert(tokens <= self.share_token_supply)
        if tokens >= self.share_token_supply:
            print(f"tokens={tokens}, staker tokens={staker.share_tokens}, total supply={self.share_token_supply}")
            tokens = self.share_token_supply
            cash_cc = self.staker_cash_cc
            staker.share_tokens = 0
            self.share_token_supply = 0
        else:
            cash_cc = (tokens / self.share_token_supply) * self.staker_cash_cc
            staker.share_tokens -= tokens
            self.share_token_supply -= tokens
        
        staker.cash_cc += cash_cc
        self.staker_cash_cc -= cash_cc
        if self.staker_cash_cc <= 0:
            self.staker_cash_cc = 0





        