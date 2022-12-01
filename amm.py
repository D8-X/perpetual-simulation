#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# AMM class
# 

import numpy as np
from amm_trader import AMMTrader
from perpetual import Perpetual
from trader import CollateralCurrency, Trader

class AMM:

    def __init__(self,
                params : dict,
                initial_default_fund_cash_cc : float=0) -> None:

        self.current_time = 0
        self.params = params
        self.perpetual_list = []

        # liquidity provision        
        self.share_token_supply = 0
        self.staker_cash_cc = 0
        self.staker_pricing_cash_ratio = 0 # 'P'
        self.last_pricing_cash_update = 0
        self.fund_transfer_convergence_time = 60 * 60 * params['iFundTransferConvergenceHours'] # 3 days in seconds
        self.max_PtoDF_ratio = 0.75 
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
        self.earnings = dict()



        # self.protocol_balance_cc = 0
        # this is just for showing on screen
        self.staker_return_ema_lambda = 1 / 30 / 100
        self.staker_return_ema = 0



    def get_perpetual(self, idx):
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
            target += p.amm_pool_target_size
        return target 

    def get_amm_pools_gap_to_target(self) -> float:
        """ missing funds to reach AMM pool target size
        Returns:
            float: missing funds. Negative if excess funds
        """
        gap = 0
        for p in self.perpetual_list:
            d = p.amm_pool_target_size - p.amm_pool_cash_cc
            gap = gap + d
        return gap

    def get_default_fund_gap_to_target(self) -> float:
        """ missing funds to reach default fund target size
        Returns:
            float: missing funds. Negative if excess funds
        """
        total_target = 0
        for p in self.perpetual_list:
            total_target = total_target + p.default_fund_target_size
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
        gap_ratio = self.default_fund_cash_cc/total_target
        return gap_ratio
    
    def __update_DF_size_target(self):
        for p in self.perpetual_list:
            p.update_DF_size_target()
        
    def __update_AMM_pool_size_target(self):
        for p in self.perpetual_list:
            p.update_AMM_pool_size_target()

    def get_timestamp(self):
        # time elapsed in seconds, we start after one block
        return (1 + self.current_time)*self.params["block_time_sec"]

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
        if px is None:
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
        if cash_cc == 0:
            return
        
        assert(cash_cc > 0 and cash_cc <= staker.cash_cc)
        
        # mint tokens
        if self.share_token_supply == 0:
            # first staker gets the first minted token
            tokens = 1.0
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
        # update pricing cash ratio: ratio_new = ratio_old * cash_old / (cash_old + delta_cash)
        self.staker_pricing_cash_ratio *= (self.staker_cash_cc - cash_cc) / self.staker_cash_cc
        self.last_pricing_cash_update = self.get_timestamp()

    def remove_liquidity(self, staker, tokens):
        if tokens == 0 or self.share_token_supply == 0:
            return

        assert(tokens > 0 and tokens <= staker.share_tokens)
        # assert(tokens <= self.share_token_supply)
        if tokens > self.share_token_supply:
            print(f"tokens={tokens}, staker tokens={staker.share_tokens}, total supply={self.share_token_supply}")
            tokens = self.share_token_supply
            cash_cc = self.staker_cash_cc
            staker.share_tokens = 0
            self.share_token_supply = 0
        else:
            cash_cc = (tokens / self.share_token_supply) * self.staker_cash_cc
            staker.share_tokens -= tokens
            self.share_token_supply -= tokens
        
        staker_pricing_cash_cc = self.staker_pricing_cash_ratio * self.staker_cash_cc
        # cash_cc = (tokens / self.share_token_supply) * self.staker_cash_cc
        # burn tokens
        # staker.share_tokens -= tokens
        # self.share_token_supply -= tokens
        # # distribute cash
        staker.cash_cc += cash_cc
        self.staker_cash_cc -= cash_cc
        # check if there wasn't enough virtual cash
        if staker_pricing_cash_cc > self.staker_cash_cc:
            # print(f"Liquidity removed exceeds virtual cash: {cash_cc} > {self.staker_cash_cc + cash_cc - staker_pricing_cash_cc}")
            # print(f"Total LP={self.staker_cash_cc + cash_cc} = {staker_pricing_cash_cc} + {self.staker_cash_cc + cash_cc - staker_pricing_cash_cc} (P + V)")
            gap = staker_pricing_cash_cc - self.staker_cash_cc
            max_gap_to_fill =  self.default_fund_cash_cc # np.max((0, -self.get_default_fund_gap_to_target())) #np.sqrt(self.max_PtoDF_ratio) * self.default_fund_cash_cc
            if gap > max_gap_to_fill:
                print(f"DANGER: liquidity removed has price impact! (we're not going to refill everything)")
                gap = max_gap_to_fill
            # give each amm pool as much cash as they were using 
            # from the portion of the pricing pot that was withdrawn
            for p in self.perpetual_list:
                delta_cash_cc =  gap / len(self.perpetual_list)
                p.amm_pool_cash_cc += delta_cash_cc
                self.default_fund_cash_cc -= delta_cash_cc
            staker_pricing_cash_cc = self.staker_cash_cc

        if self.staker_cash_cc <= 0:
            self.staker_cash_cc = 0
            self.staker_pricing_cash_ratio = 0
        else:
            if staker_pricing_cash_cc > self.staker_cash_cc:
                print("Really shouldn't be here....")
                staker_pricing_cash_cc = self.staker_cash_cc
            self.staker_pricing_cash_ratio = staker_pricing_cash_cc / self.staker_cash_cc
        self.last_pricing_cash_update = self.get_timestamp()

    
    def transfer_from_df_to_amm(self, perp, amount, target):
        if amount <= 0:
            # no gap to fill
            return 0
        # once per block
        ts_now = self.get_timestamp()
        if ts_now <= perp.last_df_transfer:
            return 0
        scale = np.min((1, (ts_now - perp.last_df_transfer) / self.fund_transfer_convergence_time))
        # cap amount to transfer
        delta_cash_cc = scale * np.min((target, self.max_liquidity_inc_delta))
        if delta_cash_cc > amount:
            delta_cash_cc = amount
        self.default_fund_cash_cc -= delta_cash_cc
        perp.amm_pool_cash_cc += delta_cash_cc
        perp.last_df_transfer = ts_now
        return delta_cash_cc
        
        
    
    def increment_pricing_staker_cash(self):
        ts_now = self.get_timestamp()
        if ts_now <= self.last_pricing_cash_update or self.staker_cash_cc == 0:
            return

        z = (ts_now - self.last_pricing_cash_update) / self.fund_transfer_convergence_time
        P_max = self.max_PtoDF_ratio * self.default_fund_cash_cc
        staker_pricing_cash_cc = self.staker_pricing_cash_ratio * self.staker_cash_cc
        gap_to_max = P_max - staker_pricing_cash_cc
        dP = 0
        if gap_to_max < 0:
            dP = -z * staker_pricing_cash_cc
            if dP < gap_to_max:
                dP = gap_to_max
        else:
            target = self.staker_cash_cc if P_max > self.staker_cash_cc else P_max
            gap_to_target = target - staker_pricing_cash_cc
            if gap_to_target > 0:
                dP = z * target
                if dP > gap_to_target:
                    dP = gap_to_target
        max_dP = z * self.max_liquidity_inc_delta
        if np.abs(dP) > max_dP:
            dP = np.sign(dP) * max_dP
        
        staker_pricing_cash_cc += dP
        assert(staker_pricing_cash_cc <= self.staker_cash_cc)
        self.staker_pricing_cash_ratio = staker_pricing_cash_cc / self.staker_cash_cc
        self.last_pricing_cash_update = ts_now






        