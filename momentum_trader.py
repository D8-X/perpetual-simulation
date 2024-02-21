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
    def __init__(self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, cash_cc=np.nan, is_best_tier=False, slip_tol=0.0010) -> None:
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_best_tier=is_best_tier)
        
        self.EMA = amm.get_perpetual(perp_idx).idx_s2[0]
        self.alpha = 0.0001
        
        self.slippage_tol  = np.random.uniform(np.max((10 / 10_000, slip_tol - 5 / 10_000)), slip_tol + 10 / 10_000) 
         # higher probability of trading the larger the indicator
        self.trade_threshold = 2* self.slippage_tol + np.abs(np.random.normal(0, 0.015))
        self.deviation_tol = 0.0200

    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
        
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        super().query_trade_amount()
        if not self.is_active:
            return (0, False)
        
        # if self.get_margin_balance_cc() < 0:
        #     if self.position_bc != 0:
        #         print(f"{self.__name__} should have been liquidated! position:{self.position_bc}, cash={self.cash_cc}, bal={self.get_margin_balance_cc()}.")
        #     return (0, False)
        
        

        perp = self.amm.get_perpetual(self.perp_idx)
        self.EMA = self.alpha * perp.get_index_price() + (1 - self.alpha) * self.EMA
        indicator = np.log(perp.get_index_price()) - np.log(self.EMA)
        
        # the price is roughly at the EMA
        if np.abs(indicator) < self.trade_threshold:
            if self.position_bc == 0:
                # no open position, nothing to do
                return (0, False)
            else:
                # close existing position
                if not self.is_below_max_deviation(-self.position_bc, self.deviation_tol):
                    return (0, False)
                return (-self.position_bc, True)
        else:
        # the price deviates from the EMA
            if self.position_bc == 0:
                # no open position -> try to open one
                pos = np.sign(indicator) * perp.get_max_leverage_position(self)
                pos = perp.scale_to_max_signed_trader_position(pos) * 0.99
                if self.is_below_max_deviation(pos, self.slippage_tol):
                    return (pos, False)
                return (0, False)
            else:
                # there is an open position
                if self.position_bc * indicator < 0:
                    # position is opposite to the observed price move -> close it
                    if not self.is_below_max_deviation(-self.position_bc, self.deviation_tol):
                        return (0, False)
                    return (-self.position_bc, True)
                return (0, False)
               