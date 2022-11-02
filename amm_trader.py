#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Trader class for the AMM
# 
from trader import Trader, CollateralCurrency
import numpy as np

class AMMTrader (Trader):
    def __init__(self, amm: 'AMM', perp_idx: int, cc: CollateralCurrency, initial_cash_cc : float, **kwargs) -> None:
        super().__init__(amm, perp_idx, cc)
        self.amm = amm
        self.cash_cc = initial_cash_cc
    
    def query_trade_amount(self, fee : float) -> "tuple[float, bool]":
        return (0, False)