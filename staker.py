import numpy as np
from abc import ABC, abstractmethod


class Staker(ABC):
    def __init__(self, amm, cash_cc) -> None:
        self.amm = amm
        self.cash_cc = cash_cc
        self.share_tokens = 0
        self.has_staked = False
        self.deposit_cash_cc = 0

    def deposit(self, cash_cc):
        # checks are done by amm
        self.deposit_cash_cc = self.cash_cc
        self.deposit_time = self.amm.get_timestamp()
        self.amm.add_liquidity(self, cash_cc)
        self.has_staked = True
    
    def withdraw(self, tokens):
        self.amm.remove_liquidity(self, tokens)
        
    def exit(self):
        if self.share_tokens > 0:
            self.withdraw(self.share_tokens)
    
    def get_apy(self):
        if not self.has_staked or self.deposit_cash_cc <= 0:
            return np.nan
        delta_t = self.amm.get_timestamp() - self.deposit_time
        if delta_t < (24 * 60 * 60): # at least a day
            return np.nan
        pos_value_cc = self.get_position_value_cc()
        if np.isnan(pos_value_cc): 
            return np.nan
        if pos_value_cc < 1:
            return -1
        N = (365 * 24 * 60 * 60) / delta_t
        r_nom = self.get_position_value_cc() / self.deposit_cash_cc - 1
        return (1 + r_nom / N) ** N - 1
        
        return np.exp((365 * 24 * 60 * 60 / delta_t) * np.log(pos_value_cc / self.deposit_cash_cc)) - 1
    # c_t_sec = e^(t_sec * r) c0
    # r = 1/t_sec  * ln(c_t/c_0)
    # c_1year = e^{1 year * r} c_0 = e^{1 year / n seconds} * r} ^ {n_seconds} c_0
    # apy = c_1year / c_0 - 1 = e^{1 year in seconds  * 1/delta_seconds * ln(c_t / c_0)} - 1

    def get_position_value_cc(self):
        if self.share_tokens > 0 and (self.share_tokens / self.amm.share_token_supply) > 0.001:
            return np.nan
        else:
            return self.cash_cc
        
        value = self.cash_cc
        if self.share_tokens > 0 and self.amm.share_token_supply > 1:
            value += (self.share_tokens / self.amm.share_token_supply) * self.amm.staker_cash_cc
        return value

    @abstractmethod
    def stake(self):
        """ Query how much the staker would deposit to the pnl participation pool given the current market."""
        pass




