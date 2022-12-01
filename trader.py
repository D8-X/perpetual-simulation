#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Abstract trader class 
# 

import numpy as np
from abc import ABC, abstractmethod

from enum import Enum

class CollateralCurrency(Enum):
    BASE = 1
    QUOTE = 2
    QUANTO = 3

class Trader(ABC):
    _counter = 0
    def __init__(self, amm : 'AMM', perp_idx : int, cc: CollateralCurrency, cash_cc=np.nan, lot_bc=1e-10, is_best_tier=False) -> None:
        self.perp_idx = perp_idx
        self.amm = amm
        self.position_bc = 0 # position in base currency
        self.cash_cc = 0 if np.isnan(cash_cc) else cash_cc # cash in collateral currency
        self.locked_in_qc = 0 # locked-in value in quote currency
        self.premium_qc = 0 # premium compared to spot owed/paid in quote currency
        self.collateral_currency = cc
        self.is_active = True
        self.min_trade_pos = lot_bc
        self.slippage_tol = 0.05
        self.id = Trader._counter
        self.is_best_tier = is_best_tier
        self.pnl_cc = 0
        Trader._counter += 1
        self.amm.trader_dir.append(self)

    def query_trade_amount(self) -> "tuple[float, bool]":
        """ Query how much the trader would trade given the current market."""
        if self.cash_cc <= 0:
            self.set_active_status(False)
            return
        # parent generically checks if trader is bankrupt, actual trade amounts must be determined by derived classes
        perp = self.amm.get_perpetual(self.perp_idx)
        if self.position_bc == 0:
            # no open position: bankrupt == not enough cash to open a position at mark and max leverage
            lev_cash_bc = self.cash_cc / perp.get_base_to_collateral_conversion(is_mark_price=True) / perp.params['fInitialMarginRate']
            if lev_cash_bc < perp.min_num_lots_per_pos * perp.params['fLotSizeBC'] or self.cash_cc < 0:
                # strict because slippage exists
                self.set_active_status(False)
        # logic for existing position should be implemented: if trader doesn't close in time she is liquidated
        pass
    def get_perpetual(self):
        return self.amm.get_perpetual(self.perp_idx)

    def set_active_status(self, status: bool):
        self.is_active = status
        # set status in perp (active if a position is open and not bankrupt)
        self.amm.get_perpetual(self.perp_idx).trader_status[self.id] = status and self.position_bc != 0

    def set_initial_cash(self, amountUSD : float, amm: 'AMM', perp_idx: int):
        perp = amm.get_perpetual(perp_idx)
        fx_q2c = 1 / perp.get_collateral_to_quote_conversion()
        return amountUSD * fx_q2c

    def pay_funding(self):
        c0 = self.cash_cc
        perp = self.get_perpetual()
        perp.pay_funding(self)
        self.pnl_cc += self.cash_cc - c0

    def notify_liquidation(self, liq_amount_bc : float, px : float, cost_cc : float):
        self.pnl_cc -= cost_cc

    def trade(self, dPos, is_close):
        c0 = self.cash_cc
        try:
            px = self.amm.trade_with_amm(self, dPos, is_close)
        except:
            px = None
        if px:
            self.pnl_cc += self.cash_cc - c0
        return px

    def get_margin_balance_cc(self, perpetual : 'Perpetual', at_mark=True) -> float:
        """
        Get the margin balance of the trader in collateral currency. 
        Open position valued at_mark price by default.
        """

        fx_b2c = perpetual.get_base_to_collateral_conversion(at_mark)
        fx_q2c = 1/perpetual.get_collateral_to_quote_conversion()
        pnl_cc = self.position_bc * fx_b2c - self.locked_in_qc*fx_q2c
        margin_cc = pnl_cc + self.cash_cc
        assert(~np.isnan(margin_cc))
        return margin_cc

    def get_maintenance_margin(self, perpetual : 'Perpetual', at_mark=True) -> float:
        """Get maintenance margin in collateral currency

        Args:
            amm (AMM): AMM instance

        Returns:
            float: maintenance margin in collateral currency
        """
        fx_b2c = perpetual.get_base_to_collateral_conversion(at_mark)
        mgn_rate = perpetual.get_maintenance_margin_rate(self.position_bc)
        margin_cc = self.position_bc*fx_b2c * mgn_rate
        return np.abs(margin_cc)

    def get_initial_margin(self, perpetual : 'Perpetual', at_mark=True) -> float:
        """Get initial margin in collateral currency

        Args:
            amm (AMM): AMM instance

        Returns:
            float: initial margin in collateral currency
        """
        fx_b2c = perpetual.get_base_to_collateral_conversion(at_mark)
        mgn_rate = perpetual.get_initial_margin_rate(self.position_bc)
        margin_cc = self.position_bc*fx_b2c * mgn_rate
        return np.abs(margin_cc)

    def get_available_margin(self, perpetual : 'Perpetual', is_initial_margin: bool, at_mark=True) -> float:
        """ Get margin balance - initial margin
            or margin balance - maintenance margin if is_initial_margin=True
        Args:
            perpetual (Perpetual): [description]
            is_initial_margin (bool): [description]

        Returns:
            float: difference to requested margin
        """
        mgn_balance = self.get_margin_balance_cc(perpetual, at_mark)
        if is_initial_margin:
            mgn = self.get_initial_margin(perpetual, at_mark)
        else:
            mgn = self.get_maintenance_margin(perpetual, at_mark)
        assert(~np.isnan(mgn))
        assert(~np.isnan(mgn_balance))
        return mgn_balance - mgn


    def is_maintenance_margin_safe(self, perpetual : 'Perpetual') -> bool:
        """check if trader is margin safe

        Args:
            amm (AMM): AMM instance

        Returns:
            bool: true if margin safe
        """
        if self.position_bc==0:
            return True
        m = self.get_margin_balance_cc(perpetual)
        required_margin = self.get_maintenance_margin(perpetual)
        return m>required_margin

    def get_slippage(self, trade_amount):
        """Calculate slippage for given trade amount

        Args:
            trade_amount ([type]): desired trade amount

        Returns:
            [type]: slippage for given amount (relative, always positive)
        """
        perp = self.amm.perpetual_list[self.perp_idx]
        px = perp.get_price(trade_amount)
        px_idx = perp.get_index_price()
        return np.abs((px-px_idx)/px_idx)

    def is_below_max_deviation(self, trade_amount, max_slippage):
        """[summary]

        Args:
            trade_amount ([float]): amount the trader wishes to trade
            max_slippage ([float]): maximum tolerance for the trader to enter the position

        Returns:
            [bool]: true if deviation below max_slippage
        """
        dev = self.get_slippage(trade_amount)
        return dev<max_slippage

    def get_max_slippage_size(self, max_slippage, trade_amount_target, tol=0.0010):
        """[summary]

        Args:
            max_slippage ([type]): percentage max slippage
            trade_amount_target ([type]): we want to trade no more than this amount
            tol: max out trade amount up to slippage+tol<max_slippage
        """
        assert(max_slippage>0)
        slip_rate = self.get_slippage(trade_amount_target)
        if slip_rate<max_slippage:
            return trade_amount_target

        s = np.sign(trade_amount_target)
        posR = np.abs(trade_amount_target)
        posL = self.min_trade_pos
        count = 0
        while posL+self.min_trade_pos<posR and count<10:
            slip_rate = self.get_slippage(s*0.5*(posL+posR))
            if slip_rate > max_slippage:
                posR = 0.5*(posL+posR)
            elif slip_rate+tol < max_slippage:
                posL = 0.5*(posL+posR)
            else:
                # done
                return s*0.5*(posL+posR)
            count += 1
        return 0
    