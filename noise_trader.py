#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np

class NoiseTrader(Trader):
    def __init__(
        self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, 
        cash_cc=np.nan, daily_trades=None, is_best_tier=False, prob_long=0.5, slip_tol=0.0010):
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_best_tier=is_best_tier)

        # fix a trade initialization probability randomly
        # one trade every one to two days
        if not daily_trades:
            self.prob_trade = np.random.uniform(0.5/(60*24), 1/(60*24)) #np.random.uniform(1/(60*24), 5/(60*24))
        else:
            self.prob_trade = np.min((1, daily_trades / (60 * 24)))
        # fix a probability that the trader is long
        self.prob_long = prob_long
        # holding period: 15min-48h
        self.holding_period_seconds = 60 * np.random.uniform(15, 48 * 60)
        self.time_last_trade = -self.holding_period_seconds
        self.time_last_pnl_check = -self.holding_period_seconds
        
        # slippage tolerance:
        self.slippage_tol  =  np.random.uniform(0.0003, 0.0013)  #np.random.uniform(np.max((5 / 10_000, slip_tol - 5 / 10_000)), slip_tol + 5 / 10_000) 
        
        # when to close?
        self.cash_to_open_cc = 0 # to track pnl
        # stop loss at somewhere between 5% and 50% loss
        self.stop_loss = np.random.uniform(0.05, 0.50)
        # take profit at somewhere between 10% and 100% profit
        self.take_profit = np.random.uniform(0.10, 1)
        # still need this?
        self.deviation_tol = 0.02
        # funding rate tol
        self.funding_rate_tol = np.random.uniform(0.0005, 0.0015)


    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
        
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        super().query_trade_amount()
        if not self.is_active:
            return (0, False)

        perp = self.amm.get_perpetual(self.perp_idx)


        if self.position_bc != 0:
            assert(self.cash_to_open_cc > 0)
            # try to close with some probability, but not if it hasn't been long enough
            if np.random.uniform(0, 1) < self.prob_trade and self.amm.current_time - self.time_last_trade > self.holding_period_seconds:
                # print("Noise trader randomly closes")
                return (-self.position_bc, True)
            
            # check rough TP/SL closing condition
            if self.amm.current_time - self.time_last_pnl_check > self.holding_period_seconds / 3:
                self.time_last_pnl_check = self.amm.current_time
                exit_balance_cc = (self.position_bc * perp.get_price(-self.position_bc) - self.locked_in_qc)/perp.get_collateral_to_quote_conversion() + self.cash_cc
                rough_pnl = exit_balance_cc/self.cash_to_open_cc - 1
                
                if rough_pnl > self.take_profit or rough_pnl < -self.stop_loss:
                    # close position, slippage is accounted for
                    return (-self.position_bc, True)
            # otherwise, no trade
            return (0, False)
            

         
        # no open position: randomly open one
        if np.random.uniform(0, 1) > self.prob_trade:
            # no trade
            return (0, False)
        
        
        dir = 1 if np.random.uniform(0, 1) < self.prob_long else -1
        # don't open if funding rate is too bad for the chosen direction
        if dir * perp.get_funding_rate() > self.funding_rate_tol:
            print(f"dir: {dir}")
            print(f"fuding rate: {perp.get_funding_rate()}")
            print(f"tol: {self.funding_rate_tol}")
            return (0, False)
        
        # maximal position
        pos = dir * perp.get_max_leverage_position(self)
        # shrink randomly, not everyone wants to try to trade at max leverage
        pos *= np.random.beta(a=5, b=1)
        # check if the price we get is not deviating too much from the index price
        count = 0
        exceeds = True
        while count<3 and exceeds:
            exceeds = not self.is_below_max_deviation(pos, self.slippage_tol)
            if exceeds:
                pos = pos/2
            count = count + 1
        if exceeds:
            return (0, False)
        # shrink pos subject to slippage
        # pos = self.get_max_slippage_size(max_slippage=self.slippage_tol, trade_amount_target=pos, tol=0.0010)
        # scale down, maybe
        pos = perp.scale_to_max_signed_trader_position(pos) * 0.99
        
        if pos != 0:
            return (pos, False)
        # no trade
        return (0, False)
    
    def trade(self, dPos, is_close):
        cash_before = self.cash_cc
        px = super().trade(dPos, is_close)
        if px:
            self.time_last_trade = self.amm.current_time
            if not is_close:
                self.cash_to_open_cc = cash_before
            self.time_last_pnl_check = self.amm.current_time
        return px
       
        

    