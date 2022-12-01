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
        if delta_t < 60 * 60: # at least an hour
            return np.nan
        pos_value_cc = self.get_position_value_cc()
        if pos_value_cc <= 0:
            return -1
        
        return (365 * 24 * 60 * 60 / delta_t) * (self.get_position_value_cc() / self.deposit_cash_cc - 1)

    def get_position_value_cc(self):
        value = self.cash_cc
        if self.share_tokens > 0 and self.amm.share_token_supply > 0:
            value += (self.share_tokens / self.amm.share_token_supply) * self.amm.staker_cash_cc
        return value

    @abstractmethod
    def stake(self):
        """ Query how much the staker would deposit to the pnl participation pool given the current market."""
        pass




