#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np

class Attacker(Trader):
    def __init__(
        self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, 
        cash_cc=np.nan, is_best_tier=False):
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_best_tier=is_best_tier)

        # timeline for one cycle:
        # 1) open long here at t0
        # 2) open long position in CEX at t1
        # 3) close long here at t2
        # 4) open short here at t3
        # 5) close position in CEX at t4
        # 6) close short position here at t5
        
        # timestamps trades, TODO: replace by timestamps
        self.trade_times = [
            0,
            #1, # this happens on cex 
            2,  
            3,  
            #4, # this happens on cex
            5,
        ]
        # which trade are we waiting for
        self.next_trade_idx = 0
        # list of trades to perform
        self.trades = (
            # t0, open long here
            (10, False),
            # t1 open long at cex, price will go up soon
            # t2 close long here, make profit
            (-10, True),
            # t3 open short here 
            (-5, False),
            # t4 close long on cex, price will go down soon
            # t0 close short here, make profit
            (5, True),
        )

    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
        
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        super().query_trade_amount()
        if not self.is_active:
            return (0, False)

        if self.amm.get_timestamp() >= self.trade_times[self.next_trade_idx]:
            # time to trade:
            return self.trades[self.next_trade_idx]
    
    def trade(self, dPos, is_close):
        px = super().trade(dPos, is_close)
        if px:
            self.next_trade_idx += 1
        else:
            print(f"Attack failed: ({dPos}, {is_close})")
        return px
       
        

    