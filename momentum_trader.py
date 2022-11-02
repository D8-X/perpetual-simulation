#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# The momentum trader initiates a long trade if log(spot)/log(EMA)-1>1.5%,
# short trade if <5%. 
# EMA is chosen so we have an approximate moving average of 5 days:
# EMA = alpha*Spot + (1-alpha)*EMA, and alpha = 2/(N+1), 
# N = 60*24*5 --> alpha=0.0001
#
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np

class MomentumTrader(Trader):
    def __init__(self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, cash_cc=np.nan, is_staker=False) -> None:
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_staker=is_staker)
        
        self.EMA = amm.get_perpetual(perp_idx).idx_s2[0]
        self.alpha = 0.0001
         # higher probability of trading the larger the indicator
        self.trade_threshold = 0.01+np.abs(np.random.normal(0, 0.015))
        self.slippage_tol = 0.02
        self.devaition_tol = 0.05

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
        self.EMA = self.alpha * perp.get_index_price() + (1 - self.alpha) * self.EMA
        indicator = np.log(perp.get_index_price()) - np.log(self.EMA)
        
        # the price is roughly at the EMA
        if np.abs(indicator)<self.trade_threshold:
            if self.position_bc == 0:
                # no open position, nothing to do
                return (0, False)
            else:
                # close existing position
                if not self.is_below_max_deviation(-self.position_bc, self.devaition_tol):
                    return (0, False)
                return (-self.position_bc, True)

        # the price deviates from the EMA
        if self.position_bc == 0:
            # no open position -> try to open one
            # pos = self.get_max_trade_amount(np.sign(indicator))
            pos = np.sign(indicator) * perp.get_max_leverage_position(self)
            if not self.is_below_max_deviation(pos, self.slippage_tol):
                return (0, False)
            pos = perp.scale_to_max_signed_trader_position(pos)
            # assert(pos == 0 or np.abs(pos) >= perp.min_num_lots_per_pos * perp.params['fLotSizeBC'])
            return (pos, False)
        else:
            # there is an open position
            if self.position_bc * indicator < 0:
                # position is opposite to the observed price move -> close it
                if not self.is_below_max_deviation(-self.position_bc, self.devaition_tol):
                    return (0, False)
                return (-self.position_bc, True)
            return (0, False)
            
    def get_max_trade_amount(self, dir):
        perp = self.amm.get_perpetual(self.perp_idx)
        max_lvg_pos = dir * perp.get_max_leverage_position(self)
        max_trade_size = perp.get_max_signed_trade_size_for_position(self.position_bc, max_lvg_pos) * 0.99
        trade_amount = dir * np.min((np.abs(max_lvg_pos), np.abs(max_trade_size)))
        return trade_amount
       