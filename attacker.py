#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
import datetime
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np
import json

class Attacker(Trader):
    def __init__(
        self, amm: AMM, perp_idx : int, cc: CollateralCurrency, 
        cash_cc=np.nan, is_best_tier=False):
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_best_tier=is_best_tier)
        
        with open("./data/GMXAttack.json") as json_file:
            self.attack_vector = json.load(json_file)
        
        # timeline for one cycle:
        # 1) open long here at t0
        # 2) open long position in CEX at t1
        # 3) close long here at t2
        # 4) open short here at t3
        # 5) close position in CEX at t4
        # 6) close short position here at t5
        
        # which cycle are we waiting for/currently executing, starts at 1
        self.current_cycle = 1
        # which trade in this cycle are we waiting for/currently executing
        self.next_trade_idx = 0
           

    def get_next_trade_timestamp(self):
        
        if self.current_cycle <= len(self.attack_vector):
            time_vector = self.attack_vector[self.current_cycle - 1]["t"]
            if len(time_vector) > self.next_trade_idx:
                # current cycle has more trades left
                return time_vector[self.next_trade_idx]
            elif len(self.attack_vector) < self.current_cycle:
                # finished cycle, go to next one
                self.current_cycle += 1
                self.next_trade_idx = 0
                return self.get_next_trade_timestamp()
        return np.inf
                
            
    
    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
        
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        super().query_trade_amount()
        if not self.is_active:
            return (0, False)

        cur_ts = self.amm.get_timestamp()
        attack_ts = self.get_next_trade_timestamp()
        # print(f"AMM time = {cur_ts}, Attack time = {attack_ts}")
        if cur_ts >= attack_ts:
            print(f"{datetime.datetime.fromtimestamp(cur_ts)} Attacker wants to trade:")
            # time to trade:
            trade = self.attack_vector[self.current_cycle-1]["trades"][self.next_trade_idx]
            if self.position_bc == 0:
                # opening, so we scale down if needed
                perp = self.amm.get_perpetual(self.perp_idx)
                pos = perp.scale_to_max_signed_trader_position(trade[0])
                if np.abs(pos) < np.abs(trade[0]):
                    trade[0] = 0.99 * pos
            else:
                trade[0] = -self.position_bc
            print(trade)
            return (trade[0], bool(trade[1]))
        return (0, False)
    
    
    def trade(self, dPos, is_close):
        perp = self.amm.get_perpetual(self.perp_idx)
        mid_price = 0.5*(perp.get_price(0.01) + perp.get_price(-0.01))
        mark_price = perp.get_mark_price()
        idx_price = perp.get_index_price()
        px = super().trade(dPos, is_close)
        if px:
            
            print(f"Attacker has {'closed' if is_close else 'opened'}: pos={dPos}")
            print(f"Mid premium = {100*(mid_price/idx_price - 1):.3f}%, Mark premium = {100*(mark_price/idx_price - 1):.3f}")
            print(f"Slippage: from idx = {100*(px/idx_price - 1):.3f}, from mid = {100*(px/mid_price - 1):.3f}")
            if is_close:
                print(f"Cumulative PnL = {self.pnl_cc}")
            self.next_trade_idx += 1
            if self.next_trade_idx == len(self.attack_vector[self.current_cycle-1]["trades"]):
                # no more trades in this cycle
                self.current_cycle += 1
                self.next_trade_idx = 0
            if len(self.attack_vector) < self.current_cycle:
                # no more cycles left, we're done
                print(f"Attack complete, resulting pnl = {self.pnl_cc}")
        else:
            print(f"Attack failed: ({dPos}, {is_close})")
        return px
       
        

    