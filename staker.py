import numpy as np
from abc import ABC, abstractmethod


class Staker(ABC):
    def __init__(self, amm, cash_cc) -> None:
        self.amm = amm
        self.cash_cc = cash_cc
        self.share_tokens = 0
        self.has_staked = False
        self.initial_cash_cc = cash_cc

    def deposit(self, cash_cc):
        # assert(cash_cc >= 0)
        # assert(cash_cc <= self.cash_cc)
        # checks are done by amm
        self.amm.add_liquidity(self, cash_cc)
    
    def withdraw(self, tokens):
        # assert(tokens <= self.share_tokens)
        self.amm.remove_liquidity(self, tokens)
    
    def exit(self):
        if self.share_tokens > 0:
            self.withdraw(self.share_tokens)
    
    def get_pnl(self):
        if not self.has_staked:
            return np.nan
        return self.get_position_value_cc() / self.initial_cash_cc - 1

    def get_position_value_cc(self):
        value = self.cash_cc
        if self.share_tokens > 0 and self.amm.share_token_supply > 0:
            value += (self.share_tokens / self.amm.share_token_supply) * self.amm.staker_cash_cc
        return value

    @abstractmethod
    def stake(self):
        """ Query how much the staker would deposit to the pnl participation pool given the current market."""
        pass




