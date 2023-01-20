#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
import numpy as np
from staker import Staker

class CommittedStaker(Staker):
    def __init__(self, amm, cash_cc):
        super().__init__(amm, cash_cc)

    def stake(self):
        if self.share_tokens == 0 and self.cash_cc > 0:
            # always enter if cash is positive
            self.deposit(self.cash_cc)
        
    