#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
import numpy as np
from staker import Staker

class NoiseStaker(Staker):
    def __init__(self, amm, cash_cc, monthly_stakes, holding_period_months):
        super().__init__(amm, cash_cc)
        assert(monthly_stakes > 0 and holding_period_months > 0)
        one_month_in_seconds = (60 * 60 * 24 * 30)
        self.prob_stake_per_block = np.min((1, monthly_stakes / one_month_in_seconds * self.amm.params["block_time_sec"]))
        self.holding_period_seconds = holding_period_months * one_month_in_seconds 
        self.time_last_stake = -self.holding_period_seconds

        # to compute pnl
        self.initial_stake = 0
        # stop loss at somewhere between 5% and 10% loss
        self.stop_loss = np.random.uniform(0.10, 0.20)
        # take profit at somewhere between 10% and 20% profit
        self.take_profit = np.random.uniform(0.20, 0.40)

        
        


    def stake(self):

        if self.share_tokens == 0:
            # randomly add liquidity
            if np.random.uniform() < self.prob_stake_per_block:
                self.initial_stake = self.cash_cc
                self.deposit(self.cash_cc)
                self.time_last_stake = self.amm.get_timestamp()
                self.has_staked = True
        else:
            assert(self.initial_stake > 0 and self.amm.share_token_supply > 0)
            if self.amm.get_timestamp() - self.time_last_stake > self.holding_period_seconds:
                self.exit()
            return
            # unstake if it's been long enough or SL/TP criteria are met
            stake_value = (self.share_tokens / self.amm.share_token_supply) * self.amm.staker_cash_cc
            pnl = stake_value / self.initial_stake -1
            if pnl < -self.stop_loss or pnl > self.take_profit or self.amm.current_time - self.time_last_stake > self.holding_period_blocks:
                self.exit()
    