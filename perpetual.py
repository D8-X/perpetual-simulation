#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Perpetual class
#

import numpy as np
from trader import CollateralCurrency, Trader
from amm_trader import AMMTrader
import pricing_benchmark
from scipy.stats import norm

EPS = 1E-12
GAS_FEE = 0.0 # in CC
class Perpetual:
    def __init__(self,
                 amm: "AMM",
                 initial_amm_cash: float,
                 initial_margin_cash: float,
                 idx_s2: np.array,
                 idx_s3: np.array,
                 params: dict,
                 glbl_params: dict,
                 cc: CollateralCurrency,
                 my_idx: int,
                 min_spread=None,
                 incentive_spread=None,
                 max_position=np.inf,
                 verbose=0, 
                 symbol=""):
        """[summary]

        Args:
            amm (AMM): [description]
            initial_amm_cash (float): initial cash in AMM pool (in CC)
            initial_margin_cash (float): initial cash in AMM trader (in CC)
            idx_s2 (np.array): time series of index prices (BC)
            idx_s3 (np.array): time series of collateral prices (CC - defaults to s2 if collateral currency = base currency)
            params (dict): this perp parameters
            glbl_params (dict): global parameters
            cc (CollateralCurrency): collateral currency (quote (M1), base (M2) or quanto (M3))
            my_idx (int): this perp's index
            min_spread (float, optional): Minimal (bid-ask) spread. Defaults to 0.00025.
            incentive_rate (float, optional): Incentive spread per unit of trade size EMA. Defaults to 0.
        """
        ## properties
        self.my_amm = amm
        self.my_idx = my_idx
        self.symbol = symbol
        self.collateral_currency = cc
        self.amm_trader = AMMTrader(amm, my_idx, cc, initial_margin_cash, lot_bc=params['fLotSizeBC'])
        ## params
        self._set_params(glbl_params, params, min_spread, incentive_spread)
        ## data
        self._set_index_data(cc, idx_s2, idx_s3)
        self.max_idx_slippage = 0.50 # 50%
        self.min_num_lots_per_pos = 10
        self.max_position = max_position
        ## state
        # global
        self.is_emergency = False
        self.current_time = amm.current_time
        # funds
        self.amm_pool_cash_cc = 0
        self.my_amm.default_fund_cash_cc += initial_amm_cash # re-route
        # traders
        self.trader_status = dict()
        self.total_volume_bc = 0  # notional traded in base currency
        self.total_volume_qc = 0 # notional traded in quote currency
        self.total_volume_cc = 0 # notional traded in quote currency
        self.open_interest = 0
        # EMAs
        self.current_AMM_exposure_EMA = np.array([1.0, 1.0]) * params['fMinimalAMMExposureEMA']
        self.current_trader_exposure_EMA = params['fMinimalTraderExposureEMA']
        self.current_locked_in_value_EMA = [-params['fMinimalAMMExposureEMA'], params['fMinimalAMMExposureEMA']]
        self.premium_rate = idx_s2*0
        # targets
        self.amm_pool_target_size = self.get_amm_pool_size_for_dd(self.params['fAMMTargetDD'])
        self.amm_pool_target_size_ema = self.amm_pool_target_size # equal at the start
        self.update_DF_size_target()

    def _set_params(self, glbl_params, params, min_spread, incentive_spread):
        self.glbl_params = glbl_params
        self.params = params
        if min_spread is not None:
            # override
            if min_spread < 0:
                raise ValueError("Minimal spread shold be non-negative")
            self.params['fMinimalSpread'] = min_spread
        if incentive_spread is not None:
            # override
            if incentive_spread < 0:
                raise ValueError("Incentive rate should be non-negative")
            self.params['fIncentiveSpread'] = min_spread
        
        if incentive_spread < 0:
            raise ValueError("Incentive rate should be non-negative")

    def _set_index_data(self, cc, idx_s2, idx_s3):
        self.idx_s2 = idx_s2
        # reuse s2 if collateral ccy == base ccy
        if cc is CollateralCurrency.BASE:
            self.idx_s3 = idx_s2
            self.params['fRho23'] = 0
            self.params['fSigma3'] = 0
            self.params['fStressReturnS3'] = [0, 0]
        elif cc is CollateralCurrency.QUANTO:
            self.idx_s3 = idx_s3
        elif cc is CollateralCurrency.QUOTE:
            self.idx_s3 = idx_s2 * 0 + 1
            self.params['fRho23'] = 0
            self.params['fSigma3'] = 0
            self.params['fStressReturnS3'] = [0, 0]
        else:
            NameError(' currency type not implemented')


    def set_emergency_state(self):
        print(f"--------- {self.symbol} is in emergency ---------")
        self.is_emergency = True
        if all([p.is_emergency for p in self.my_amm.perpetual_list]):
            self.my_amm.is_emergency = True
            
    def inc_time(self):
        # move the EMA of the premium forward
        self.current_time = self.current_time + 1
        # increment time
        if self.current_time < self.premium_rate.shape[0]:
            self.premium_rate[self.current_time] = self.premium_rate[self.current_time - 1]
            return True
        else:
            return False

    def set_idx_s2(self, s2, time_idx):
        self.idx_s2[time_idx] = s2

    def pay_funding(self, trader):
        """ Trader pays funding
            a) Funding rate is for a given 'tick' (e.g. one minute)
            b) This function should called with every tick
            c) Payment is made to/from the AMM margin directly
        """
        if self.is_emergency:
            return
        rate = self.get_funding_rate()
        assert(rate * self.amm_trader.position_bc <= 0)
        coupon_abs = rate * np.abs(trader.position_bc)
        if coupon_abs == 0:
            return
        fxb2c = self.get_base_to_collateral_conversion(False)
        coupon = np.sign(trader.position_bc) * coupon_abs * fxb2c
        # if coupon > trader.cash_cc:
        #     print(f"Funding fee higher than trader margin!! rate = {rate}, fee = {coupon}, margin = {trader.cash_cc}, trader = {trader.id}")
        #     coupon = trader.cash_cc
            
        trader.cash_cc -= coupon
        self.transfer_cash_to_margin(coupon)
        self.my_amm.funding_earnings[self.my_idx] += coupon

    def get_price(self, amount: float):
        """Returns AMM price for a given trade amount
            a) price(0) = 1/2 (price(1 lot long) + price(1 lot short))
            b) price(k!=0) = index price * (1 + signed premium)
                i) k > 0 and premium < 0 (or k < 0 and premium > 0) implies a rebate
            c) If there are no funds, price is undefined (M>=0 but not negative, unlike margin cash)
        """
        if amount == 0:
            return 0.5*(self.get_price(self.params['fLotSizeBC'])+self.get_price(-self.params['fLotSizeBC']))
        minSpread = 0 if amount == 0 else self.params['fMinimalSpread']
        incentiveSpread = 0 if amount == 0 else self.params['fIncentiveSpread']
        s2 = self.idx_s2[self.current_time]
        s3 = self.idx_s3[self.current_time]
        M1, M2, M3 = self.get_amm_pools()
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        if M1 + M2 + M3 < 0:
            self.set_emergency_state()
            print(f"warning : pricing with negative cash! M1={M1}, M2={M2}, M3={M3}, amm_pool={self.amm_pool_cash_cc}, df={self.my_amm.default_fund_cash_cc}")
        px = pricing_benchmark.calculate_perp_priceV4(K2, amount, L1, s2, s3,
                                                    self.params['fSigma2'],
                                                    self.params['fSigma3'],
                                                    self.params['fRho23'],
                                                    self.params['r'],
                                                    M1, M2, M3, minSpread, incentiveSpread, self.current_trader_exposure_EMA)
        if px is None:
            print(f"Price is undefined! {self.symbol}")
        return px

    def update_mark_price(self):
        """
            Update the mark-price and the EMA for the position size of the
            AMM used to calculate the target size of the AMM pool
        Returns:
            [self]: [this]
        """
        L = self.params['fMarkPriceEMALambda']
        px_mid = self.get_price(0)
        curr_premium = px_mid/self.idx_s2[self.current_time]-1
        ema_mark = self.premium_rate[self.current_time]*L + (1-L) * curr_premium
        self.premium_rate[self.current_time] = ema_mark
        px_mark = (1+ema_mark) * self.idx_s2[self.current_time]
        return px_mark

    def get_mark_price(self):
        px_mark = (1 + self.premium_rate[self.current_time]) * \
            self.idx_s2[self.current_time]
        return px_mark

    def get_index_price(self):
        """Returns the spot index price s_2(t) 
        """
        return self.idx_s2[self.current_time]

    def get_collateral_price(self):
        """Returns the spot index price s_3(t) if defined (quanto), else s_2(t) for base and 1 for quote 
        """
        return self.idx_s3[self.current_time]

    def get_max_signed_position_size(self, is_long):
        """How large can a new position be. Depends on the sign of the position
        New positions are sized in different ways depending on their direction:
        1) if they increase the AMM risk, they are sized with respect to the trade size EMA
        2) if they decrease the AMM risk, they can be as large as needed

        Args:
            is_long (bool): true if the position is long

        Returns:
            float: signed size of the largest allowed position given the current state
        """
        scale = self.params['fMaximalTradeSizeBumpUp']
        k_ema = self.current_trader_exposure_EMA
        k_star = self.get_Kstar()
        pos_size = scale * k_ema
        if (is_long and k_star < 0) or (not is_long and k_star > 0):
            # adverse position: rescale according to DF depletion level
            df_ratio =  np.min((1, self.my_amm.get_default_fund_gap_to_target_ratio()))
            pos_size *= df_ratio
            # new; ensure default won't happen
            #  (C * s3 - (s2 e^r  K2 -L1)) / (s2 |e^r-1|) >  |k|
            if is_long:
                s2 = self.get_index_price() * np.exp(self.params['fStressReturnS2'][1])
                s3 = 1
                if self.collateral_currency is CollateralCurrency.BASE:
                    s3 = s2
                elif self.collateral_currency is CollateralCurrency.QUANTO:
                    s3 = self.get_collateral_to_quote_conversion() * np.exp(self.params['fStressReturnS3'][0])
                cash = self.amm_trader.cash_cc + self.get_LP_cash_for_perp()
                max_pos = (cash * s3  + (self.amm_trader.position_bc * s2 - self.amm_trader.locked_in_qc)) / s2 / (np.exp(self.params['fStressReturnS2'][1])-1)
                # print(f"max_pos = {pos_size}, max_max_pos = {max_pos}")
                pos_size = max_pos if pos_size > max_pos else pos_size
            else:
                s2 = self.get_index_price() * np.exp(self.params['fStressReturnS2'][0])
                s3 = 1
                if self.collateral_currency is CollateralCurrency.BASE:
                    s3 = s2
                elif self.collateral_currency is CollateralCurrency.QUANTO:
                    s3 = self.get_collateral_to_quote_conversion() * np.exp(self.params['fStressReturnS3'][0])
                cash = self.amm_trader.cash_cc + self.get_LP_cash_for_perp()
                max_pos = (cash * s3 + (self.amm_trader.position_bc * s2 - self.amm_trader.locked_in_qc)) / s2 / (1 - np.exp(self.params['fStressReturnS2'][0]))
                # print(f"max_pos = {pos_size}, max_max_pos = {max_pos}")
                pos_size = max_pos if pos_size < max_pos else pos_size


        return pos_size * (1 if is_long else -1)


    def get_max_signed_trade_size_for_position(self, position, trade_amount):
        new_position = position + trade_amount
        max_position = self.get_max_signed_position_size(new_position > 0)
        exceeds_cap = np.abs(new_position) > self.max_position
        if exceeds_cap: # perp level cap not implemented
            max_position = np.sign(new_position) * self.max_position
        max_signed_trade_amount = np.sign(max_position - position) * np.min(
            (np.abs(max_position - position), self.get_absolute_max_trade_size()))
        k_star = self.get_Kstar()

        if not exceeds_cap and (
            (
                (max_signed_trade_amount > 0 and k_star >= max_signed_trade_amount) or 
                (max_signed_trade_amount < 0 and k_star <= max_signed_trade_amount))
        ):
            max_signed_trade_amount = k_star   
        res = self.my_amm.shrink_to_lot(max_signed_trade_amount, self.params['fLotSizeBC'])
        res = np.sign(res) * np.min(
            (np.abs(res), self.get_absolute_max_trade_size()))
        return res

    def scale_to_max_signed_trader_position(self, pos):
        """Scales position down to meet AMM max-position size constraints

        Args:
            pos (number): position to scale

        Returns:
            number: down-scaled position
        """
        max_signed_size = self.get_max_signed_trade_size_for_position(0, pos)
        # scale down
        abs_pos = np.min((np.abs(pos), np.abs(max_signed_size)))
        pos = self.my_amm.shrink_to_lot(abs_pos, self.params['fLotSizeBC'])  * (1 if pos > 0 else -1)
        # if below absolute min, then it's just zero
        if np.abs(pos) < self.min_num_lots_per_pos * self.params['fLotSizeBC'] or self.is_emergency:
            pos = 0
        return pos
    
    def get_LP_cash_for_perp(self):
        """Returns funds allocated to this perpetual from external LPs
            a) Perp funds = target size * min( total LP funds / sum of targets , 1)
            b) Excess funds (if any) are allocated to the default fund
        """
        T = np.sum([perp.amm_pool_target_size_ema for perp in self.my_amm.perpetual_list])
        P = self.my_amm.staker_cash_cc
        p = P/T if P < T else 1
        return p * self.amm_pool_target_size_ema

    def get_amm_pools(self):
        # get cash from LP
        staker_cash_cc = self.get_LP_cash_for_perp()
        # combine with perp margin 
        M = staker_cash_cc + self.amm_trader.cash_cc
        # assign according to collateral type
        if self.collateral_currency is CollateralCurrency.BASE:
            M1, M3 = 0, 0
            M2 = M
        elif self.collateral_currency is CollateralCurrency.QUANTO:
            M1, M2 = 0, 0
            M3 = M
        elif self.collateral_currency is CollateralCurrency.QUOTE:
            M1 = M
            M2, M3 = 0, 0
        else:
            raise NameError("Collateral currency not valid")
        return M1, M2, M3

    def get_Kstar(self):
        """Returns a k* such that Q(k*) = 0

        Returns:
            [type]: [description]
        """
        time_idx = np.min((self.current_time, self.idx_s2.shape[0] - 1))
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        s2 = self.idx_s2[time_idx]
        s3 = self.idx_s3[time_idx]
        rho = self.params['fRho23']
        sig2 = self.params['fSigma2']
        sig3 = self.params['fSigma3']
        r = self.params['r']
        M1, M2, M3 = self.get_amm_pools()
        k_star = pricing_benchmark.get_Kstar(s2, s3, M1, M2, M3, K2, L1, sig2, sig3, rho, r)
        return k_star

    def get_funding_rate(self):
        """Calculates the current funding rate (for one block/tick)
            a) f = max(m, c) + min(m, -c) + sign(K2) * base_rate
            b) m = max(mark prem,0) if K2 > 0, min(mark prem, 0) if K2 < 0
            c) c, base_rate are params
            d) capped to 90% of (initial - maintenance) rate so that it can't drive a trader to liquidation in one block
            e) The sign of f is such that the AMM always gets paid
        """
        time_idx = np.min((self.current_time, self.premium_rate.shape[0] - 1))
        premium_rate = self.premium_rate[time_idx]
        c = self.params['fFundingRateClamp']
        K2 = -self.amm_trader.position_bc
        rate = np.max((premium_rate, c)) + np.min((premium_rate, -c)) + np.sign(K2)*0.0001
        rate = np.max((rate, 0.0001)) if K2 > 0 else np.min((rate, -0.0001))
        rate = rate * self.glbl_params['block_time_sec']/(8*60*60)
        max_rate = (self.params['fInitialMarginRate']-
                    self.params['fMaintenanceMarginRate'])*0.9
        min_rate = -max_rate
        rate = np.max((rate, min_rate)) if rate < 0 else np.min(
            (rate, max_rate))
        if (rate > 0 and self.amm_trader.position_bc) > 0 or (rate < 0 and self.amm_trader.position_bc < 0):
            print(f"Bad funding rate sign!! rate = {rate}, K2 = {-self.amm_trader.position_bc}")
        return rate

    def get_base_to_collateral_conversion(self, is_mark_price: bool):
        """
        Base to collateral FX: 1 if base, s_2 if quote, s_2/s_3 if quanto
        """
        if is_mark_price:
            return self.get_mark_price() / self.idx_s3[self.current_time]
        else:
            return self.idx_s2[self.current_time] / self.idx_s3[self.current_time]

    def get_collateral_to_quote_conversion(self):
        """
        Collateral to quote FX: s_2 if base, 1 if quote, s_3 if quanto
        """
        return self.idx_s3[self.current_time]

    def get_base_to_quote_conversion(self, is_mark_price: bool):
        """
        Base to quote FX: s_2 always
        """
        if is_mark_price:
            return self.get_mark_price()
        else:
            return self.idx_s2[self.current_time]

    def get_rebalance_margin(self):
        """
        Margin to rebalance = margin - initial margin
        """
        return self.amm_trader.get_available_margin(self, True, False)

    def is_exposure_covered(self):
        s2 = self.idx_s2[self.current_time]
        s3 = self.idx_s3[self.current_time]
        M1, M2, M3 = self.get_amm_pools()
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        return M1 + s2 * M2 + s3 * M3 + L1 - s2 * K2 > EPS

    def transfer_cash_to_margin(self, amount_cc):
        # amount_cc could be negative, but adding over all traders is should end up positive
        self.amm_trader.cash_cc += amount_cc
        self.my_amm.earnings[self.my_idx] += amount_cc

    def rebalance_perpetual(self, rebalance_another=True):
        """Rebalance margin of the perpetual to initial margin
            a) Positive PnL is distributed to LPs and DF proportionally
            b) Negative PnL:
                i) If AMM balance (allocated cash + margin + pnl) is negative:
                    -> perp has defaulted and it enters emergency state
                    -> allocated funds are used up, and DF covers the rest
                    -> if DF is used up, exchange has defaulted and AMM enters emergency statey
                ii) If AMM cannot be brought to initial margin with allocated funds, we limit the losses
                iii) Otherwise LPs and DF pay proportionally
            c) Mark price is updated
            d) A random perpetual in the pool is also rebalanced     
            e) Called before and after trading
        """
        if self.is_emergency:
            # we do not rebalance perps that are in emergency state 
            # (simulates that margin cash is "lost")
            return
        
        rebalance_amnt_cc = self.get_rebalance_margin()

        if rebalance_amnt_cc > 0:
            # transfer From AMM Margin To Pool
            (amount_staker, amount_amm) = self.__split_amount(rebalance_amnt_cc, False)
            self.amm_trader.cash_cc = self.amm_trader.cash_cc - rebalance_amnt_cc
            self.my_amm.staker_cash_cc += amount_staker
            self.my_amm.default_fund_cash_cc += amount_amm
        elif rebalance_amnt_cc < 0:
            # transfer from pool to AMM margin
            amt = -rebalance_amnt_cc # withdraw (switch sign for convenience)
            # need to get funds from pools
            M1, M2, M3 = self.get_amm_pools()
            # total funds = pool + margin cash
            pool_funds = M1 + M2 + M3 - self.amm_trader.cash_cc
            perp_balance = self.amm_trader.get_margin_balance_cc(self, False)
            # Three cases: 
            # 1) can't bring AMM to zero margin -> perp default, possible pool default
            # 2) can't bring AMM to init margin -> limit the losses (no extra cash for this perp)
            # 3) losses can be covered
            if pool_funds + perp_balance < 0:
                # perp has defaulted
                print(f"\t---- {self.symbol} in emergency ----")
                self.set_emergency_state()
                # df covers just enough to cover traders' pnl, not restore initial margin
                amount_df = -(pool_funds + perp_balance)
                amount_lp = pool_funds
                # check for pool default
                if amount_df > self.my_amm.default_fund_cash_cc:
                    print(f"\t---- {self.symbol} caused pool-wide emergency ----")
                    self.my_amm.set_emergency()
                    return
            elif pool_funds + rebalance_amnt_cc < 0:
                # perp made a large loss, pools will cover what they can but no more cash until it makes a profit
                # TODO: re-visit later
                (amount_lp, amount_df) = self.__split_amount(pool_funds, True)
            else:
                # business as usual: distribute loss
                (amount_lp, amount_df) = self.__split_amount(amt, True)
                # check for default (but this shouldn't happen, just to be safe)
                if amount_df > self.my_amm.default_fund_cash_cc:
                    print(f"\t---- {self.symbol} caused unexpected pool-wide emergency ----")
                    self.my_amm.set_emergency()
                    return
            # pay 
            self.my_amm.staker_cash_cc -= amount_lp
            self.my_amm.default_fund_cash_cc -= amount_df
            self.amm_trader.cash_cc += amount_lp + amount_df
                
        self.update_mark_price()
        # rebalance another perp randomly
        if rebalance_another:
            other_perp_idx = np.random.randint(0, len(self.my_amm.perpetual_list))
            self.my_amm.perpetual_list[other_perp_idx].rebalance_perpetual(rebalance_another=False)


    def get_amm_pool_size_for_dd(self, dd):
        """ Calculate the target funds for a given target distance to default
            a) Funds here are in the context of PD: all available cash, pool and margin
            b) PD is computed for a "typical" trade size
            c) Direction of this trade size is taken as opposite to k-star (risk-increasing)
            d) Result is floored: margin_cash + cash_from_pools >= floor
            e) So cash_from_pools target could possibly be zero if margin is already enough
        """
        s2 = self.idx_s2[self.current_time]
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        dir = 0 if K2 * L1 == 0 else np.sign(self.get_Kstar())
        kappa = self.current_trader_exposure_EMA
        if dir < 0:
            K2 = K2 + kappa
            L1 = L1 + kappa*self.get_index_price()
        else:
            K2 = K2 - kappa
            L1 = L1 - kappa*self.get_index_price()

        if self.collateral_currency is CollateralCurrency.BASE:
            if dd > 10:
                size_0 = K2 - L1 / s2
            else:
                size_0 = pricing_benchmark.get_target_collateral_M2(
                    K2, s2, L1,
                    self.params["fSigma2"],
                    dd
                )
        elif self.collateral_currency is CollateralCurrency.QUANTO:
            s3 = self.idx_s3[self.current_time]
            if dd > 10:
                size_0 = (s2 * K2 - L1) / s3
            else:
                size_0 = pricing_benchmark.get_target_collateral_M3(
                    K2, s2, s3, L1,
                    self.params["fSigma2"],
                    self.params["fSigma3"],
                    self.params["fRho23"],
                    self.params["r"],
                    dd
                )
        elif self.collateral_currency is CollateralCurrency.QUOTE:
            if dd > 10:
                size_0 = K2 * s2 - L1
            else:
                size_0 = pricing_benchmark.get_target_collateral_M1(
                    K2, s2, L1,
                    self.params["fSigma2"],
                    dd
                )
        else:
            raise NameError('not implemented')
        
        size_0 = np.max((size_0, self.params['fAMMMinSizeCC']))
        size_0 -= self.amm_trader.cash_cc
        return size_0 if size_0 > 0 else 0


    def __split_amount(self, amount, is_withdraw=False):
        """Split PnL between external LPs (PnL part fund) and protocol (Default Fund)
        Returns (amount_LP , amount_DF) such that
            a) amount_LP + amount_DF == amount 
            b) is_withdraw == True:
                i) amount_LP <= LP
                ii) there is NO guarantee that amount_DF <= DF
        """
        if amount == 0:
            return (0, 0)
        # assert(self.my_amm.staker_cash_cc + self.amm_pool_cash_cc > 0)
        avail_cash_cc = self.my_amm.staker_cash_cc + self.my_amm.get_amm_funds() + self.my_amm.default_fund_cash_cc
        w_staker = self.my_amm.staker_cash_cc / avail_cash_cc

        # TODO Re-introduce this cap: 
        # a) Tried using initial ratio and volatility was still high. 
        # b) Should probably use a lower number.
        
        # w_staker = np.min((self.glbl_params['ceil_staker_pnl_share'], w_staker))
        # if not is_withdraw:
        #     w_staker = np.min(
        #         (self.glbl_params['ceil_staker_pnl_share'], w_staker))

        amount_staker = w_staker * amount
        amount_amm = amount - amount_staker
            
        return (amount_staker, amount_amm)

    def __update_exposure_ema(self, pos):
        """update the EMA for trader exposure, AMM exposure, and AMM locked-in value

        Args:
            pos ([float]): [trader position]
        """
        L = self.params['fDFLambda']
        idx_lambda = 1 if np.abs(pos) > np.abs(
            self.current_trader_exposure_EMA) else 0
        self.current_trader_exposure_EMA = self.current_trader_exposure_EMA * \
            L[idx_lambda] + (1-L[idx_lambda]) * np.abs(pos)
        self.current_trader_exposure_EMA = np.max(
            (self.current_trader_exposure_EMA,
             self.params['fMinimalTraderExposureEMA'])
        )

        # AMM exposures for default fund
        pos_amm = -self.amm_trader.position_bc
        idx = 0 if pos_amm < 0 else 1
        locked_in = -self.amm_trader.locked_in_qc

        idx_lambda = 1 if pos_amm < (
            -self.current_AMM_exposure_EMA[0]) or pos_amm > self.current_AMM_exposure_EMA[1] else 0
        self.current_AMM_exposure_EMA[idx] = self.current_AMM_exposure_EMA[idx] * \
            L[idx_lambda] + (1-L[idx_lambda]) * np.abs(pos_amm)
        self.current_AMM_exposure_EMA[idx] = np.max(
            (self.current_AMM_exposure_EMA[idx], self.params['fMinimalAMMExposureEMA']))
        self.current_locked_in_value_EMA[idx] = self.current_locked_in_value_EMA[idx] * \
            L[idx_lambda] + (1-L[idx_lambda]) * locked_in

    def update_AMM_pool_size_target(self, dd=None):
        """Updates amm_pool_target_size and amm_pool_target_size_ema
        """
        if not dd:
            dd = self.params['fAMMTargetDD']

        self.amm_pool_target_size = self.get_amm_pool_size_for_dd(dd)
        # ema_1 = L ema_0 + (1-L) * obs_1
        # -> ema_n = L^n ema_0 + (1-L) (obs_1 + L obs_2 + ... L^n-1 obs_n)
        # -> E(ema_n) = L^n E(ema_0) + (1 - L^n) * E(obs)
        L = self.params['fMarkPriceEMALambda'] # e.g. L = 70%
        self.amm_pool_target_size_ema = L * self.amm_pool_target_size_ema + (1-L) * self.amm_pool_target_size
    
    def get_num_active_traders(self):
        """Number of traders with an open position in this perpetual"""
        return sum(self.trader_status.values())

    def update_DF_size_target(self):
        """Updates default_fund_target_size"""
        K2_pair = self.current_AMM_exposure_EMA
        k2_trader = self.current_trader_exposure_EMA
        fCoverN = np.max((5, self.params["fDFCoverNRate"] * sum(self.trader_status.values())))
        r2pair = self.params['fStressReturnS2']
        r3pair = self.params['fStressReturnS3']
        s3 = self.idx_s3[self.current_time]
        s2 = self.idx_s2[self.current_time]
        s = pricing_benchmark.get_DF_target_size(K2_pair, k2_trader, r2pair, r3pair, fCoverN,
                                                 s2, s3, self.collateral_currency, 1/self.params['fInitialMarginRate'])
        self.default_fund_target_size = s

    def get_pd(self, k, p, M=None):
        M1, M2, M3 = self.get_amm_pools()
        if M:
            M1 = (M1 > 0) * M
            M2 = (M2 > 0) * M
            M3 = (M3 > 0) * M
        if not self.collateral_currency is CollateralCurrency.QUANTO:
            pd, _ = pricing_benchmark.prob_def_no_quanto(
                -self.amm_trader.position_bc + k,
                -self.amm_trader.locked_in_qc + k * p,
                self.idx_s2[self.current_time],
                self.idx_s3[self.current_time],
                self.params['fSigma2'],
                self.params['fSigma3'],
                self.params['fRho23'],
                self.params['r'],
                M1, M2, M3
            )
        else:
            pd, _ = pricing_benchmark.prob_def_quanto(
                -self.amm_trader.position_bc + k,
                -self.amm_trader.locked_in_qc + k * p,
                self.idx_s2[self.current_time],
                self.idx_s3[self.current_time],
                self.params['fSigma2'],
                self.params['fSigma3'],
                self.params['fRho23'],
                self.params['r'],
                M1, M2, M3
            )

        assert(pd >= 0 and pd <= 1)
        return pd

    def get_absolute_max_trade_size(self):
        return np.inf #60_000 / self.get_base_to_quote_conversion(is_mark_price=False)

    def is_new_position_margin_safe(self, trader, amount_bc, price_qc):
        # this only makes sense if this is an opening trade, closing incurs on different pnl
        position_bc = trader.position_bc + amount_bc
        s2 = self.get_index_price()
        premium = amount_bc * (price_qc - s2)
        lockedin_qc = trader.locked_in_qc + amount_bc * s2
        fx_q2c = 1 / self.get_collateral_to_quote_conversion()
        cash_cc = trader.cash_cc - premium * fx_q2c
       
        fx_b2c = self.get_base_to_collateral_conversion(True)
        pnl_cc = position_bc * fx_b2c - lockedin_qc*fx_q2c
        margin_balance_cc = pnl_cc + cash_cc
        mgn_rate = self.get_initial_margin_rate(position_bc)
        initial_margin_cc = np.abs(position_bc*fx_b2c * mgn_rate)

        is_safe = margin_balance_cc >= initial_margin_cc
        if not is_safe:
            print(f"{trader.__class__.__name__} not initial margin safe! {margin_balance_cc} < {initial_margin_cc}")
        return is_safe
        
    def trade(self, trader: 'Trader', amount_bc: float, is_close_only: bool):
        if self.is_emergency:
            return None
        # if trader.cash_cc <= 0 and not is_close_only:
        #     print(f"Trade rejected:  {trader.__class__.__name__} does not have cash left: {trader.cash_cc}")
        #     # can't trade without cash
        #     return None
        if np.abs(amount_bc) < self.params['fLotSizeBC']:
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} tried to trade less than one lot: {amount_bc} < {self.params['fLotSizeBC']}")
            return None
        
        new_position_bc = trader.position_bc + amount_bc
        # if closing, either there's at least one lot left, or the entire position is closed
        if is_close_only:
            if np.abs(new_position_bc) >= self.min_num_lots_per_pos * self.params['fLotSizeBC']:
                amount_bc = self.my_amm.round_to_lot(amount_bc, self.params['fLotSizeBC'])
                new_position_bc = trader.position_bc + amount_bc
            else:
                amount_bc = -trader.position_bc
                new_position_bc = 0
        else:
            amount_bc = self.my_amm.round_to_lot(amount_bc, self.params['fLotSizeBC'])
            new_position_bc = trader.position_bc + amount_bc
            

        is_trying_to_exit = trader.position_bc != 0 and np.abs(new_position_bc) < self.min_num_lots_per_pos * self.params['fLotSizeBC']
        # if amount is less than one lot, revert
        if np.abs(amount_bc) < self.params['fLotSizeBC'] and not is_trying_to_exit:
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} trade below lot size {np.abs(amount_bc):.4f} < {self.params['fLotSizeBC']}")
            return None
        

        # if resulting position is smaller than minimal size, revert
        if new_position_bc != 0 and np.abs(new_position_bc) < self.min_num_lots_per_pos * self.params['fLotSizeBC'] and not is_trying_to_exit:
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} resulting position below minimal size {np.abs(new_position_bc):.6f} < {self.min_num_lots_per_pos * self.params['fLotSizeBC']}")
            return None
        self.rebalance_perpetual()
        px = self.get_price(amount_bc)
        if px <= 0:
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} price for amount {amount_bc} is undefined")
        
        is_opening = (new_position_bc > trader.position_bc and trader.position_bc >= 0) or (new_position_bc < trader.position_bc and trader.position_bc <= 0)
        if is_opening and not self.is_new_position_margin_safe(trader, amount_bc, px):
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} not enough margin")
            return None

        k_star = self.get_Kstar()
        is_direction_adverse = amount_bc * k_star <= 0
        
        # adverse trades are canceled if too large or too much slippage
        if is_direction_adverse:
            max_trade_size = self.get_max_signed_trade_size_for_position(trader.position_bc, amount_bc)
            
            if np.abs(amount_bc) > np.abs(max_trade_size) and not is_trying_to_exit:
                action_msg = f"{'fully' if is_trying_to_exit else ''} close" if is_close_only else "open"
                msg = f"{trader.__class__.__name__} was trying to {action_msg} a trade"
                print(f"Trade rejected: {self.symbol} trade size violation {np.abs(amount_bc):.4f} > {np.abs(max_trade_size):.4f} ({msg})")
                return None
           
        # trade with AMM:
        (delta_cash, is_open) = self.book_trade_with_amm(
            trader, px, amount_bc, is_close_only)

        # update AMM state
        self.rebalance_perpetual()
        # fees
        self.__distribute_fees(trader, np.abs(amount_bc))
        
        return px

    def __distribute_fees(self, trader, amount_bc):
        """Distributes fee and fill AMM gap from default fund

        Args:
            trader ([type]): [description]
            amount_bc ([type]): [description]
        """
        amount_bc = np.abs(amount_bc)
        (staker_fee, protocol_fee) = self.__calc_fees(trader, amount_bc)
        if self.my_amm.staker_cash_cc == 0:
            protocol_fee += staker_fee
            staker_fee = 0
        total_fee = protocol_fee + staker_fee
        
        # distribute amounts
        trader.cash_cc = trader.cash_cc - total_fee
        # assert(staker_fee >= 0)
        # assert(staker_fee == 0 or self.my_amm.staker_cash_cc > 0)
        self.my_amm.staker_cash_cc += staker_fee
        # self.my_amm.default_fund_cash_cc += protocol_fee - amm_pool_contribution

        # CHANGE
        # self.amm_pool_cash_cc += protocol_fee
        self.my_amm.default_fund_cash_cc += protocol_fee
        
        # record keeping
        self.my_amm.fees_earned += protocol_fee # total
        self.my_amm.earnings[self.my_idx] += protocol_fee # this perp

    def __calc_fees(self, trader, amount_bc):
        """calculates the fee amounts in collateral currency

        Args:
            trader (Trader):    Trader object that is to be charged
            amount_bc ([float]):amount being traded

        Returns:
            [Tuple]: [staker fees and protocol fees]
        """
        if trader.is_best_tier or (self.params['fTreasuryFeeRate'] + self.params['fPnLPartRate']) < 1e-5:
            return (0, 0)
        amount_bc = np.abs(amount_bc)
        fx_b2c = self.get_base_to_collateral_conversion(False)
        fee_rate = self.params['fTreasuryFeeRate']
        # if self.my_amm.get_default_fund_gap_to_target_ratio() < 0.5:
        #     # extra fee
        #     fee_rate += self.params['protocol_fee_rate_extra']
        protocol_fee = amount_bc*fee_rate*fx_b2c
        staker_fee = amount_bc*self.params['fPnLPartRate']*fx_b2c
        total_fee = protocol_fee + staker_fee
        # mgn_cc = balance - maintenance_margin, at mark
        mgn_cc = trader.get_available_margin(self, False)
        if mgn_cc < 0:
            # no fees
            return (0, 0)
        if total_fee > mgn_cc:
            scale = mgn_cc/total_fee
            protocol_fee = protocol_fee * scale
            staker_fee = staker_fee * scale
        return (staker_fee, protocol_fee)

    def book_trade_with_amm(self, trader: 'Trader', price_qc: float, amount_bc: float, is_close_only: bool) -> None:
        """
        book the trade between AMM and trader
        """
        assert(not np.isnan(price_qc))
        assert(not np.isnan(amount_bc) and np.abs(amount_bc) > 0)
        if is_close_only:
            max_amount = np.abs(trader.position_bc)
            assert(np.abs(amount_bc) <= max_amount)

        # is_close = trader.position_bc != 0 and np.sign(trader.position_bc) != np.sign(amount_bc)
        new_pos_bc = trader.position_bc + amount_bc
        is_close = (
            (trader.position_bc > new_pos_bc and  new_pos_bc >= 0) or
            (trader.position_bc < new_pos_bc and  new_pos_bc <= 0)
        )

        s2 = self.get_index_price()
        premium = amount_bc * (price_qc - s2)
        delta_locked_value = amount_bc * s2
        fx_q2c = 1 / self.get_collateral_to_quote_conversion()
        delta_cash = -premium * fx_q2c
        if is_close:
            # calculate pnl and cash out the trader
            avg_price = trader.locked_in_qc/trader.position_bc
            pnl = avg_price * amount_bc - delta_locked_value
            delta_locked_value += pnl
            delta_cash += fx_q2c * pnl
        assert(not np.isnan(delta_locked_value))
        assert(not np.isnan(delta_cash))
        assert(not np.isnan(amount_bc))


        self.updateMargin(trader, amount_bc, delta_cash,
                          delta_locked_value)
        self.total_volume_bc += np.abs(amount_bc)
        self.total_volume_qc += np.abs(amount_bc) * self.get_index_price()
        self.total_volume_cc += np.abs(amount_bc) * self.get_base_to_collateral_conversion(False)
        # update average trade sizes for AMM pool and default fund
        # only account for opening trades
        if not is_close:
            self.__update_exposure_ema(trader.position_bc)
        # at this point the trade was successful: gas fees are paid from the amm margin account
        self.transfer_cash_to_margin(-GAS_FEE)
        self.update_AMM_pool_size_target()
        return (delta_cash, is_close)

    def updateMargin(self, trader, amount_bc, delta_cash, delta_locked_value):
        assert(not np.isnan(delta_locked_value))
        assert(not np.isnan(delta_cash))
        assert(not np.isnan(amount_bc))
        if not trader is AMMTrader and trader.cash_cc + delta_cash < 0:
            # the trader can't lose more than he owns so the cash delta is capped
            # (this is probably a liquidation)
            delta_cash = -trader.cash_cc
        # trader margin
        self.__updateTraderMargin(
            trader, amount_bc, delta_cash, delta_locked_value)
        # amm margin
        assert(np.abs(amount_bc) > 0)
        self.__updateTraderMargin(
            self.amm_trader, -amount_bc, -delta_cash, -delta_locked_value)
        # record pnl
        self.my_amm.earnings[self.my_idx] -= delta_cash

    def __updateTraderMargin(self, trader, amount_bc, delta_cash, delta_locked_value):
        old_pos = trader.position_bc
        trader.locked_in_qc = trader.locked_in_qc + delta_locked_value
        trader.position_bc = trader.position_bc + amount_bc
        trader.cash_cc = trader.cash_cc + delta_cash
        # adjust open interest
        delta_oi = 0
        if old_pos > 0:
            delta_oi = -old_pos
        if trader.position_bc > 0:
            delta_oi = delta_oi + trader.position_bc
        self.open_interest = self.open_interest + delta_oi

    def get_maintenance_margin_rate(self, pos):
        diff = self.params['fInitialMarginRate'] - \
            self.params['fMaintenanceMarginRate']
        return self.get_initial_margin_rate(pos) - diff

    def get_initial_margin_rate(self, pos):
        return self.params['fInitialMarginRate']

    def __getPositionAmountToLiquidate(self, trader: "Trader") -> float:
        """ For partial liquidations, we calculate
            how much we have to liquidate for the trader
            to be maintenance margin safe.

        Args:
            trader ([Trader]): [trader instance]
        """
        # f: fee rate
        if trader.is_best_tier:
            f = self.params['fLiquidationPenaltyRate']
        else:
            f = self.params['fLiquidationPenaltyRate'] +\
                self.params['fTreasuryFeeRate'] + \
                self.params['fPnLPartRate']
        # b0: margin balance
        b0 = trader.get_margin_balance_cc(self)
        # S3/S2
        # fx_c2b = 1 / self.get_base_to_collateral_conversion(False)
        sm = self.get_mark_price()
        s2 = self.get_index_price()
        s3 = self.get_collateral_to_quote_conversion()
        # p: trader position
        pos = trader.position_bc
        # m: margin rate
        m = self.get_initial_margin_rate(pos)
        # if we are here it's because b0 < |p| * m, but we check anyway
        num = np.abs(pos) * m * sm - b0 * s3
        den = m * sm - f * s2
        if num <= 0 or den <= 0:
            return pos
        delta = np.sign(pos) * np.abs(num / den)

        if np.abs(pos) < np.abs(delta) or np.abs(pos - delta) < self.min_num_lots_per_pos * self.params['fLotSizeBC']:
            delta = pos
        else:
            delta = self.my_amm.grow_to_lot(delta, self.params['fLotSizeBC'])
        delta = np.sign(delta) * np.min((np.abs(pos), np.abs(delta)))

        return delta
    
    def pay_liquidation_penalty(self, liq_amount_bc, trader: Trader):
        remaining_mgn = trader.get_available_margin(self, False, True)
        if remaining_mgn < 0:
            remaining_mgn = 0
        b2c = self.get_base_to_collateral_conversion(False)
        penalty_cc = np.abs(liq_amount_bc) * self.params['fLiquidationPenaltyRate'] * b2c
        if penalty_cc > remaining_mgn:
            penalty_cc = remaining_mgn
        self.__updateTraderMargin(trader, 0, -penalty_cc, 0)
        # distribute penalty
        amount_liquidator = penalty_cc / 2
        amount_default_fund = penalty_cc - amount_liquidator
        
        self.my_amm.liquidator_earnings_vault += amount_liquidator
        # pay amount to default fund
        self.my_amm.default_fund_cash_cc += amount_default_fund
        
        # record keeping
        self.my_amm.earnings[self.my_idx] += amount_default_fund
        return (penalty_cc, remaining_mgn)


    def liquidate(self, trader) -> bool:

        # rebalance perpetual because of price moves since last rebalance
        # rebalance perpetual updates the mark price
        self.rebalance_perpetual()
        
        if trader.is_maintenance_margin_safe(self):
            return False

        liq_amount_bc = -self.__getPositionAmountToLiquidate(trader)

        # liquidate at mark-price
        px = self.get_mark_price()
        trade_cash_before = trader.cash_cc
        (delta_cash, is_open) = self.book_trade_with_amm(
            trader, px, liq_amount_bc, True)

        penalty_cc, remaining_mgn_cc = self.pay_liquidation_penalty(liq_amount_bc, trader)
        
        # rebalance perpetual because of margin account changes since last rebalance
        self.rebalance_perpetual()
        # pay regular trading fees/rebalance AMM cash
        if remaining_mgn_cc > penalty_cc:
            self.__distribute_fees(trader, liq_amount_bc)
        
        trade_cash_after = trader.cash_cc
        trader.notify_liquidation(liq_amount_bc, px, np.abs(trade_cash_after - trade_cash_before))

        if sum(self.trader_status.values()) < 1:
            assert(np.abs(self.amm_trader.position_bc) < self.params['fLotSizeBC'])

        return True

    def get_max_leverage_position(self, trader):
        # cash_cc * s3 = pos * (price - sm) + f |pos| s2 + |pos| sm / leverage
        # pos * (price - sm) = pos*(price - s2) + pos*(s2 - sm)<= |pos| * (slip  + prem) * s2)
        # --> |pos| >= cash * (s3 / s2) /  (m_r * sm/s2 + f + slip + prem )
        if trader.cash_cc <= 0:
            return 0
        fee_rate = 0 if trader.is_best_tier else (self.params['fTreasuryFeeRate'] + self.params['fPnLPartRate'])
        s2 = self.get_index_price()
        sm = self.get_mark_price()
        s3 = self.get_collateral_price()
        premium_rate = np.abs(sm - s2) / s2
        slip_tol = trader.slippage_tol
        mr = self.params['fInitialMarginRate']
        return  (trader.cash_cc * s3 / s2) / (mr * sm / s2 + fee_rate + slip_tol + premium_rate)


    def export(self):
        state = {
            'fCurrentFundingRate': self.get_funding_rate(),
            'fOpenInterest': self.open_interest,
            'fAMMFundCashCC': self.amm_pool_cash_cc,
            'fkStar': self.get_Kstar(),
            'fkStarSide': np.sign(-self.get_Kstar()),
            'fTargetAMMFundSize': self.amm_pool_target_size,
            'fTargetDFSize': self.default_fund_target_size,
            'fCurrentAMMExposureEMA': list(self.current_AMM_exposure_EMA),
            'fCurrentTraderExposureEMA': self.current_trader_exposure_EMA,
        }

        response = {'idx': self.my_idx, 'state': state}

        return response

