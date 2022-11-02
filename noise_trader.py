#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np

class NoiseTrader(Trader):
    def __init__(
        self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, 
        cash_cc=np.nan, daily_trades=None, is_staker=False, prob_long=0.5):
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc, is_staker=is_staker)

        # fix a trade initialization probability randomly
        # one trade every one to two days
        if not daily_trades:
            self.prob_trade = np.random.uniform(0.5/(60*24), 1/(60*24)) #np.random.uniform(1/(60*24), 5/(60*24))
        else:
            self.prob_trade = np.min((1, daily_trades / (60 * 24)))
        # fix a probability that the trader is long
        self.prob_long = prob_long
        # holding period: 15min-12h
        self.holding_period_blocks = 60*np.random.uniform(1/4, 12)
        self.time_last_trade = -self.holding_period_blocks
        self.time_last_pnl_check = -self.holding_period_blocks
        # slippage tolerance: 0.5%-2%
        self.slippage_tol  = np.random.uniform(0.0050, 0.0200)
        
        # when to close?
        self.cash_to_open_cc = 0 # to track pnl
        # stop loss at somewhere between 5% and 10% loss
        self.stop_loss = np.random.uniform(0.05, 0.10)
        # take profit at somewhere between 10% and 20% profit
        self.take_profit = np.random.uniform(0.10, 0.20)
        # still need this?
        self.deviation_tol = 0.02


    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
        
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        super().query_trade_amount()
        if not self.is_active:
            return (0, False)
            
        # if self.position_bc != 0 and self.amm.current_time - self.time_last_trade < self.holding_period_blocks:
        #     # open position and holding period not fullfilled, don't trade yet
        #     return (0, False)

        perp = self.amm.get_perpetual(self.perp_idx)

        # check rough TP/SL closing condition
        if self.position_bc != 0 and self.amm.current_time - self.time_last_pnl_check > 0.5*self.holding_period_blocks:
            self.time_last_pnl_check = self.amm.current_time
            exit_balance_cc = (self.position_bc * perp.get_price(-self.position_bc) - self.locked_in_qc)/perp.get_collateral_to_quote_conversion() + self.cash_cc
            rough_pnl = exit_balance_cc/self.cash_to_open_cc - 1
            
            if rough_pnl > self.take_profit and self.is_below_max_deviation(-self.position_bc, self.slippage_tol):
                # print(f"{self.__class__.__name__}'s rough pnl is {100*rough_pnl:.2f}%, TP/SL thresholds set to {100*self.take_profit:.2f}%/{-100*self.stop_loss:.2f}%")
                # print(f"Profit should be {exit_balance_cc-self.cash_to_open_cc}")
                return (-self.position_bc, True)
            elif rough_pnl < -self.stop_loss and self.is_below_max_deviation(-self.position_bc, self.deviation_tol):
                return (-self.position_bc, True)

         
        # otherwise random trading:
        if np.random.uniform(0, 1) > self.prob_trade:
            # no trade
            return (0, False)

        # randomly open
        if self.position_bc == 0:
            # open maximal position
            dir = 1 if np.random.uniform(0, 1) < self.prob_long else -1
            # pos *(1+fee) * marginratio > cash/fx 
            # fee = perp.params['fTreasuryFeeRate'] + perp.params['fPnLPartRate']
            pos = dir * perp.get_max_leverage_position(self)
            # check if the price is not deviating too much
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
            pos = self.get_max_slippage_size(max_slippage=self.slippage_tol, trade_amount_target=pos, tol=0.0010)
            # shrink randomly so that trades are not always at max leverage
            pos *= np.random.beta(a=5, b=1)
            # shrink pos subject to amm restrictions
            pos = perp.scale_to_max_signed_trader_position(pos) 
            if pos != 0:
                self.time_last_trade = self.amm.current_time
                self.cash_to_open_cc = self.cash_cc
                self.time_last_pnl_check = self.amm.current_time
            # assert(pos == 0 or np.abs(pos) >= perp.min_num_lots_per_pos * perp.params['fLotSizeBC'])
            return (pos, False)
        return (0, False)
        # else:
        # # if self.position_bc != 0:
        #     if not self.is_below_max_deviation(-self.position_bc, self.deviation_tol):
        #         return (0, False)
        #         # we don't account for max position size 
        #         # because we are closing
        #         # close as much of the existing position subject to slippage
        #     pos = -self.position_bc #self.get_max_slippage_size(max_slippage=self.slippage_tol, trade_amount_target=-self.position_bc, tol=0.0010)
        #     return (pos, True)
        

    