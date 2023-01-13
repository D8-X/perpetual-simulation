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
                 min_spread=0.00025,
                 incentive_spread=0,
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
        # print("Creating perpetual:")
        self.my_amm = amm
        self.amm_pool_cash_cc = initial_amm_cash
        self.collateral_currency = cc
        self.idx_s2 = idx_s2
        self.verbose = verbose
        self.params = params
        self.symbol = symbol
        # reuse s2 if collateral ccy == base ccy
        if cc is CollateralCurrency.BASE:
            # print(
            #     f"AMM pool initialized with {initial_amm_cash:.2f} held in base currency")
            self.idx_s3 = idx_s2
            self.params['fRho23'] = 0
            self.params['fSigma3'] = 0
            self.params['fStressReturnS3'] = [0, 0]
        elif cc is CollateralCurrency.QUANTO:
            # print(
            #     f"AMM pool initialized with {initial_amm_cash:.2f} held in quanto currency")
            self.idx_s3 = idx_s3
        elif cc is CollateralCurrency.QUOTE:
            # print(
            #     f"AMM pool initialized with {initial_amm_cash:.2f} held in quote currency")
            self.idx_s3 = idx_s2 * 0 + 1
            self.params['fRho23'] = 0
            self.params['fSigma3'] = 0
            self.params['fStressReturnS3'] = [0, 0]
        else:
            NameError(' currency type not implemented')
        # premium above/below spot (mark-price premium)
        self.premium_rate = idx_s2*0
        self.glbl_params = glbl_params
        self.min_spread = min_spread
        self.incentive_rate = incentive_spread
        if min_spread > 0:
            # print(
            #     f"Minimal bid/ask spread set to {10_000 * 2*min_spread:.2f} bps")
            pass
        elif min_spread < 0:
            raise ValueError("Minimal spread shold be non-negative")
        if incentive_spread > 0:
            # print(
            #     f"Incentive spread set to {10_000 * incentive_spread:.2f} bps")
            pass
        elif incentive_spread < 0:
            raise ValueError("Incentive rate should be non-negative")
        self.mark_price_history = np.zeros((3,))
        self.max_idx_slippage = 0.50 # 50%
        self.my_idx = my_idx
        self.amm_trader = AMMTrader(amm, my_idx, cc, initial_margin_cash, lot_bc=params['fLotSizeBC'])
        self.current_time = amm.current_time
        self.total_volume_bc = 0  # notional traded in base currency
        self.total_volume_qc = 0 # notional traded in quote currency
        self.total_volume_cc = 0 # notional traded in quote currency
        self.open_interest = 0
        self.amm_pool_target_size = self.amm_pool_cash_cc
        self.default_fund_target_size = self.amm_pool_cash_cc
        self.current_AMM_exposure_EMA = np.array([1.0, 1.0]) * params['fMinimalAMMExposureEMA']
        self.current_trader_exposure_EMA = params['fMinimalTraderExposureEMA']
        self.current_locked_in_value_EMA = [-params['fMinimalAMMExposureEMA'], params['fMinimalAMMExposureEMA']]
        self.trader_status = dict()
        self.max_position = max_position
        self.min_num_lots_per_pos = 10
        self.last_df_transfer = amm.get_timestamp()
        self.is_emergency = False

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
        # self.rebalance_perpetual()
        rate = self.get_funding_rate()
        coupon_abs = rate * np.abs(trader.position_bc)
        if coupon_abs == 0:
            return
        fxb2c = self.get_base_to_collateral_conversion(False)
        coupon = np.sign(trader.position_bc) * coupon_abs * fxb2c
        if coupon > trader.cash_cc:
            coupon = trader.cash_cc
        trader.cash_cc -= coupon
        self.transfer_cash_to_margin(coupon)

    def get_price(self, pos: float):
        minSpread = self.min_spread
        incentiveSpread = self.incentive_rate
        s2 = self.idx_s2[self.current_time]
        s3 = self.idx_s3[self.current_time]
        M1, M2, M3 = self.get_amm_pools()
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        if self.collateral_currency is CollateralCurrency.QUANTO and M3 <= 0:
            self.is_emergency = True
            px = None
        else:                
            px = pricing_benchmark.calculate_perp_priceV4(K2, pos, L1, s2, s3,
                                                        self.params['fSigma2'],
                                                        self.params['fSigma3'],
                                                        self.params['fRho23'],
                                                        self.params['r'],
                                                        M1, M2, M3, minSpread, incentiveSpread, self.current_trader_exposure_EMA)

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
        self.mark_price_history = [self.mark_price_history[1],
                                   self.mark_price_history[2],
                                   curr_premium]
        # curr_premium = np.median(self.mark_price_history)
        ema_mark = self.premium_rate[self.current_time]*L + (1-L) * curr_premium
        self.premium_rate[self.current_time] = ema_mark
        px_mark = (1+ema_mark) * self.idx_s2[self.current_time]

        return px_mark

    def get_mark_price(self):
        px_mark = (1 + self.premium_rate[self.current_time]) * \
            self.idx_s2[self.current_time]
        return px_mark

    def get_index_price(self):
        return self.idx_s2[self.current_time]

    def get_collateral_price(self):
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
        # if not exceeds_cap and (
        #     (
        #         (max_signed_trade_amount > 0 and 2*k_star > max_signed_trade_amount) or 
        #         (max_signed_trade_amount < 0 and 2*k_star < max_signed_trade_amount))
        # ):
        if not exceeds_cap and (
            (
                (max_signed_trade_amount > 0 and k_star >= max_signed_trade_amount) or 
                (max_signed_trade_amount < 0 and k_star <= max_signed_trade_amount))
        ):
            max_signed_trade_amount = 2*k_star   
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
        if np.abs(pos) < self.min_num_lots_per_pos * self.params['fLotSizeBC']:
            pos = 0
        return pos

    def get_pricing_staked_cash_for_perp(self):
        ## same for all perps in pool
        if len(self.my_amm.perpetual_list) > 0:
            return (self.my_amm.staker_pricing_cash_ratio * self.my_amm.staker_cash_cc) / len(self.my_amm.perpetual_list)
        else:
            return 0

    def get_amm_pools(self):
        # get cash from LP
        
        staker_cash_cc = self.get_pricing_staked_cash_for_perp()
        # combine with protocol owned cash
        pricing_cash_cc = self.amm_pool_cash_cc + staker_cash_cc + self.amm_trader.cash_cc
        # assign according to collateral type
        if self.collateral_currency is CollateralCurrency.BASE:
            M1, M3 = 0, 0
            M2 = pricing_cash_cc
        elif self.collateral_currency is CollateralCurrency.QUANTO:
            M1, M2 = 0, 0
            M3 = pricing_cash_cc
        elif self.collateral_currency is CollateralCurrency.QUOTE:
            M1 = pricing_cash_cc
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
        # return 0
        time_idx = np.min((self.current_time, self.premium_rate.shape[0] - 1))
        premium_rate = self.premium_rate[time_idx]
        c = self.params['fFundingRateClamp']
        K2 = -self.amm_trader.position_bc
        rate = np.max((premium_rate, c)) + np.min((premium_rate, -c)) + np.sign(K2)*0.0001
        rate = rate * self.glbl_params['block_time_sec']/(8*60*60)
        max_rate = (self.params['fInitialMarginRate']-
                    self.params['fMaintenanceMarginRate'])*0.9
        min_rate = -max_rate
        rate = np.max((rate, min_rate)) if rate < 0 else np.min(
            (rate, max_rate))
        return rate

    def get_base_to_collateral_conversion(self, is_mark_price: bool):
        """
        BTCUSD collateralized in BTC returns 1
        """
        if is_mark_price:
            return self.get_mark_price() / self.idx_s3[self.current_time]
        else:
            return self.idx_s2[self.current_time] / self.idx_s3[self.current_time]

    def get_collateral_to_quote_conversion(self):
        return self.idx_s3[self.current_time]

    def get_base_to_quote_conversion(self, is_mark_price: bool):
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
        self.amm_trader.cash_cc += amount_cc
        self.my_amm.earnings[self.my_idx] += amount_cc

    def rebalance_perpetual(self, rebalance_another=True):
        """Rebalance margin of the perpetual to initial margin
        """
        if self.is_emergency:
            return
        # excess/defect in the amm's margin account
        rebalance_amnt_cc = self.get_rebalance_margin()
        # account for withdrawals/deposits from LP before distribution
        self.my_amm.increment_pricing_staker_cash()
        amt_df = 0

        if rebalance_amnt_cc > 0:
            # transfer From AMM Margin To Pool
            (amount_staker, amount_amm) = self.__split_amount(rebalance_amnt_cc, False)
            self.amm_trader.cash_cc = self.amm_trader.cash_cc - rebalance_amnt_cc
            assert(amount_staker == 0 or self.my_amm.staker_cash_cc > 0)
            self.my_amm.staker_cash_cc += amount_staker
            # send amount to AMM fund (or default fund...)
            # self.my_amm.protocol_balance_cc += amount_amm
            self.__distribute_amm_cash(amount_amm)
        elif rebalance_amnt_cc < 0:
            # withdraw (switch sign for convenience)
            amt = -rebalance_amnt_cc
            # need to get funds from pools
            max_amount = self.my_amm.staker_cash_cc / len([p for p in self.my_amm.perpetual_list if not p.is_emergency]) + self.amm_pool_cash_cc
            if amt >= max_amount:
                print(f"WARNING: {self.symbol} in emergency!")
                self.is_emergency = True
                if len([p for p in self.my_amm.perpetual_list if not p.is_emergency]) < 1:
                    self.my_amm.is_emergency = True
            else:
                max_amount = 0.95 * max_amount
                # try to leave enough funds in the pool
                # max_amount = np.min((0.95 * max_amount, np.max((0, max_amount - self.params['fAMMMinSizeCC']))))
                
            if amt > max_amount:
                # print(f"WARNING: {self.symbol} Borrowing {amt-max_amount:.4f} from default fund to cover AMM trader margin")
                # preemptively cap amount withdrawn from perp funds
                # amount to withdraw from default fund
                amt_df = amt - max_amount
                # amount to withdraw from pools
                amt = max_amount
                if amt_df > self.my_amm.default_fund_cash_cc:
                    # not enough cash in default fund
                    amt_df = self.my_amm.default_fund_cash_cc
                    self.my_amm.is_emergency = True
                    for p in self.my_amm.perpetual_list:
                        p.is_emergency = True
                self.my_amm.default_fund_cash_cc -= amt_df
            (amount_staker, amount_amm) = self.__split_amount(amt, True)
            # withdraw from AMM fund and staker fund
            assert(amount_staker == 0 or self.my_amm.staker_cash_cc > amount_staker) # strict because of 95% cap
            self.my_amm.staker_cash_cc -= amount_staker
            self.amm_pool_cash_cc -= amount_amm
            # self.my_amm.protocol_balance_cc -= (amount_amm + amt_df)
            # update margin
            feasible_mgn = amt_df + amt
            self.amm_trader.cash_cc += feasible_mgn
            # if self.amm_trader.cash_cc < 0:
            #     print(self.__dict__)
        self.rebalance_amm()
        self.update_mark_price()
        # rebalance another perp randomly
        if rebalance_another:
            other_perp_idx = np.random.randint(0, len(self.my_amm.perpetual_list))
            self.my_amm.perpetual_list[other_perp_idx].rebalance_perpetual(rebalance_another=False)

    def rebalance_amm(self):
        """
        Rebalance amm cash to target-dd cash using default fund
        Used to prevent price impacts at liquidation. 
        It has no effect if AMM pool is already full.
        """
        # baseline target (1%) > stress target (2%) > AMM pool size > critical target (100%)

        # if we are above baseline target, we do nothing, make sure we know that's the target
        baseline_target_size = self.get_amm_pool_size_for_dd(
            self.params['fAMMTargetDD'][0])
        
        is_baseline_target = baseline_target_size <= self.amm_pool_cash_cc
        
        stress_target_size = self.get_amm_pool_size_for_dd(self.params['fAMMTargetDD'][1])
        
        is_baseline_target = is_baseline_target or stress_target_size <= self.amm_pool_cash_cc
        
        if is_baseline_target:
            # adjust baseline target in case DF needs cash
            df_gap_ratio = self.my_amm.get_default_fund_gap_to_target_ratio()
            if df_gap_ratio < 1:
                baseline_target_size = stress_target_size + (baseline_target_size - stress_target_size) * df_gap_ratio
            self.amm_pool_target_size = baseline_target_size
            self.last_df_transfer = 0
            return
        
        if self.last_df_transfer == 0:
            self.last_df_transfer = self.my_amm.get_timestamp()
            
        # we are here so amm_pool_target_size < stress_target_size
        self.amm_pool_target_size = stress_target_size
        # draw funds in relation to available size from default fund
        # If default fund is funded at rate r we withdraw at most min(1, r%) from it
        gap = 0.75*(stress_target_size - self.amm_pool_cash_cc)
        # assert(gap > 0)
        if gap <= 0:
            return
        
        gap_fill_df = np.min(
            (gap, 0.75 * self.my_amm.default_fund_cash_cc)
        )
        
        gap_fill_df_adjusted = self.my_amm.transfer_from_df_to_amm(self, gap_fill_df, stress_target_size)
        
        
        # draw funds from pnl participants who don't otherwise contribute to the default fund
        if gap_fill_df > 0:
            # proportionally contributes the same as the df
            gap = np.max((0, gap - gap_fill_df)) * (gap_fill_df_adjusted / gap_fill_df)
        else:
            gap = 0
        
        gap_fill_staker = np.min(
            (gap, 0.75 * self.my_amm.staker_cash_cc)
        )
        
        # self.my_amm.default_fund_cash_cc -= gap_fill_df
        self.my_amm.staker_cash_cc -= gap_fill_staker
        self.amm_pool_cash_cc += gap_fill_staker
        # self.amm_pool_cash_cc += gap_fill_df + gap_fill_staker


    def get_amm_pool_size_for_dd(self, dd):
        s2 = self.idx_s2[self.current_time]
        K2 = -self.amm_trader.position_bc
        L1 = -self.amm_trader.locked_in_qc
        dir = np.sign(self.get_Kstar())
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


        # account for staker cash
        staker_cash_cc = self.get_pricing_staked_cash_for_perp()
        # M =  amm fund + amm margin + staker_cash >= target, and this target is for amm fund, so:
        size_0 = size_0 - staker_cash_cc - self.amm_trader.cash_cc
        # target should be non-negative
        size_0 = np.max((0, size_0))
        # size_0 = np.max((size_0, self.params['fAMMMinSizeCC']))
        self.amm_pool_target_size_ema = size_0

        # if size_0 > self.amm_pool_target_size_ema:
        #     self.amm_pool_target_size_ema = size_0
        # else:
        #     self.amm_pool_target_size_ema = size_0
        #     L = 0 #self.glbl_params['AMM_lambda']
        #     self.amm_pool_target_size_ema = (
        #         L * self.amm_pool_target_size_ema + (1 - L) * size_0
        #     )

        return self.amm_pool_target_size_ema

    def __distribute_amm_cash(self, cash: float):
        """Distribute AMM cash:
            - first fill up own pool
            - second, socialize to other perpetuals

        Args:
            cash (float): [amount to be distributed]
        """
        assert(cash >= 0)
        if cash == 0:
            return
        amm_gap = self.amm_pool_target_size - self.amm_pool_cash_cc

        if amm_gap > cash:
            # amm gap is larger than cash being distributed
            # -> amm pool receives all the cash
            amm_contribution = cash
        elif amm_gap < 0:
            # no gap to fill, all cash goes to df
            amm_contribution = 0
            # TODO: should we subtract a bit of cash from the amm?
            # if so, the amount should be based on the df gap, not the amm gap (too volatile)
        else:
            # fill the gap
            amm_contribution = amm_gap
    
        self.amm_pool_cash_cc += amm_contribution
        self.my_amm.default_fund_cash_cc += cash - amm_contribution
        return

    def __split_amount(self, amount, is_withdraw=False):
        if amount == 0:
            return (0, 0)
        # assert(self.my_amm.staker_cash_cc + self.amm_pool_cash_cc > 0)
        avail_cash_cc = self.my_amm.staker_cash_cc + self.my_amm.get_amm_funds() + self.my_amm.default_fund_cash_cc
        w_staker = self.my_amm.staker_cash_cc / avail_cash_cc
        if not is_withdraw:
            w_staker = np.min(
                (self.glbl_params['ceil_staker_pnl_share'], w_staker))
        amount_staker = w_staker * amount
        amount_amm = amount - amount_staker
        if is_withdraw:
            amount_staker = amount_staker if amount_staker < self.my_amm.staker_cash_cc else self.my_amm.staker_cash_cc
            amount_amm = amount_amm if amount_amm < self.amm_pool_cash_cc else self.amm_pool_cash_cc
        # test log
        # earning = amount_staker * (-1 if is_withdraw else 1)
        # print(f"Stakers earn {earning: .4f}, which is {100 * w_staker:.2f}% of the cash being distributed")
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
        # idx_lambda = 1 if np.abs(pos_amm) > np.abs(
        #     self.current_trader_exposure_EMA) else 0
        idx_lambda = 1 if pos_amm < (
            -self.current_AMM_exposure_EMA[0]) or pos_amm > self.current_AMM_exposure_EMA[1] else 0
        self.current_AMM_exposure_EMA[idx] = self.current_AMM_exposure_EMA[idx] * \
            L[idx_lambda] + (1-L[idx_lambda]) * np.abs(pos_amm)
        self.current_AMM_exposure_EMA[idx] = np.max(
            (self.current_AMM_exposure_EMA[idx], self.params['fMinimalAMMExposureEMA']))
        self.current_locked_in_value_EMA[idx] = self.current_locked_in_value_EMA[idx] * \
            L[idx_lambda] + (1-L[idx_lambda]) * locked_in

    def update_AMM_pool_size_target(self, dd=None):
        if not dd:
            dd = self.params['fAMMTargetDD'][0]

        self.amm_pool_target_size = self.get_amm_pool_size_for_dd(dd)
    
    def get_num_active_traders(self):
        return sum(self.trader_status.values())

    def update_DF_size_target(self):

        K2_pair = self.current_AMM_exposure_EMA
        k2_trader = self.current_trader_exposure_EMA
        fCoverN = np.max((5, self.params["fDFCoverNRate"] * sum(self.trader_status.values())))
        r2pair = self.params['fStressReturnS2']
        r3pair = self.params['fStressReturnS3']
        s3 = self.idx_s3[self.current_time]
        s2 = self.idx_s2[self.current_time]
        s = pricing_benchmark.get_DF_target_size(K2_pair, k2_trader, r2pair, r3pair, fCoverN,
                                                 s2, s3, self.collateral_currency)
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
        if trader.cash_cc <= 0:
            print(f"Trade rejected:  {trader.__class__.__name__} does not have cash left: {trader.cash_cc}")
            # can't trade without cash
            return None
        if np.abs(amount_bc) < self.params['fLotSizeBC']:
            print(f"Trade rejected: {self.symbol} {trader.__class__.__name__} tried to trade less than one lot: {amount_bc} < {self.params['fLotSizeBC']}")
            return None
        # threshold = 10
        # reject opening trade if total balance in perp exceeds threshold
        # if False and self.get_total_account_balances() > threshold and not is_close_only:
        #     print(f"BREACH: Total trader margin threshold exceeded: {self.get_total_account_balances()} > {threshold}")
        #     #return None
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
        # if self.verbose > 0 and self.my_amm.get_default_fund_gap_to_target_ratio() < df_gap:
        #     print(f"Default funds: {trader.position_bc - amount_bc:.4f} -> {trader.position_bc:.4f} ({amount_bc:.3f}@{px:.1f}) k*={k_star: .4f} reduced pool size target ratio: {self.my_amm.get_default_fund_gap_to_target_ratio():.3f} < {df_gap:.3f} ({trader.__class__.__name__} {trader.id})")
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

        # AMM fund += fee
        # if AMM fund < target and Default Fund > target
        #   fill AMM fund gap with amount DF above target
        # check if AMM pool needs funds

        # gap = target - pool cash
        gap_amm = self.my_amm.get_amm_pools_gap_to_target()
        gap_amm_after_fee = gap_amm - protocol_fee
        gap_df = self.my_amm.get_default_fund_gap_to_target()
        if gap_amm_after_fee < 0:
            # amm fund exceeds target, add fee to AMM fund
            amm_pool_contribution = protocol_fee
        else:
            # AMM pool below target after adding fee

            if gap_df > 0:
                # default fund underfunded, add fee to AMM fund only
                # and leave default fund unchanged
                amm_pool_contribution = protocol_fee
            else:
                # default fund exceeds target, we can use this
                # excess amount to fill AMM fund
                df_max_withdrawal = -gap_df
                amm_pool_contribution = np.min((gap_amm_after_fee, df_max_withdrawal))

        # assert(self.my_amm.default_fund_cash_cc >= 0)
        # distribute amounts
        trader.cash_cc = trader.cash_cc - total_fee
        assert(staker_fee >= 0)
        assert(staker_fee == 0 or self.my_amm.staker_cash_cc > 0)
        self.my_amm.staker_cash_cc += staker_fee
        self.my_amm.default_fund_cash_cc += protocol_fee - amm_pool_contribution
        self.amm_pool_cash_cc += amm_pool_contribution
        self.my_amm.fees_earned += protocol_fee
        self.my_amm.earnings[self.my_idx] += protocol_fee
        # self.my_amm.protocol_balance_cc += protocol_fee

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
        assert(not np.isnan(amount_bc))
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
        if(not is_close):
            # if self.symbol == 'ETHUSD-MATIC' and trader.position_bc > 100:
            #     print(trader.__class__.__name__)
            #     print(vars(trader))
            self.__update_exposure_ema(trader.position_bc)
        # at this point the trade was successful: gas fees are paid from the amm margin account
        self.transfer_cash_to_margin(-GAS_FEE)
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
        self.__updateTraderMargin(
            self.amm_trader, -amount_bc, -delta_cash, -delta_locked_value)
        # record pnl
        self.my_amm.earnings[self.my_idx] -= delta_cash

    def __updateTraderMargin(self, trader, amount_bc, delta_cash, delta_locked_value):
        old_pos = trader.position_bc
        trader.locked_in_qc = trader.locked_in_qc + delta_locked_value
        trader.position_bc = trader.position_bc + amount_bc
        trader.cash_cc = trader.cash_cc + delta_cash
        # assert(not trader is AMMTrader or trader.cash_cc >= 0)
        # adjust open interest
        delta_oi = 0
        if old_pos > 0:
            delta_oi = -old_pos
        if trader.position_bc > 0:
            delta_oi = delta_oi + trader.position_bc
        self.open_interest = self.open_interest + delta_oi

    def get_maintenance_margin_rate(self, pos):
        # 0.06 - 0.04
        diff = self.params['fInitialMarginRate'] - \
            self.params['fMaintenanceMarginRate']
        return self.get_initial_margin_rate(pos) - diff

    def get_initial_margin_rate(self, pos):
        return self.params['fInitialMarginRate']
        # beta = self.params['fMarginRateBeta']
        # cap = self.params['fInitialMarginRateCap']
        # #params['fMaintenanceMarginRateAlpha'] = 0.04
        # alpha = self.params['fInitialMarginRateAlpha']# = 0.06
        # return np.min((cap, alpha + beta*np.abs(pos)))

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

    def liquidate(self, trader) -> None:

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

        # pay liquidation penalty
        penalty_bc = np.abs(liq_amount_bc) * self.params['fLiquidationPenaltyRate']
        penalty_cc = self.get_base_to_collateral_conversion(False) * penalty_bc
        mgn = np.max((0, trader.get_margin_balance_cc(self)))
        penalty_to_trader = np.min((penalty_cc, mgn))
        

        gap = penalty_cc - mgn
        if gap > 0:
            trader.set_active_status(False)

        # update trader margin
        trader.cash_cc = trader.cash_cc - penalty_to_trader
        # reward liquidator
        amount_liquidator = penalty_to_trader/2
        self.my_amm.liquidator_earnings_vault += amount_liquidator
        # pay amount to default fund
        self.my_amm.default_fund_cash_cc += penalty_to_trader - amount_liquidator
        # this counts as earnings for the amm
        self.my_amm.earnings[self.my_idx] += penalty_to_trader - amount_liquidator
        # rebalance perpetual because of margin account changes since last rebalance
        self.rebalance_perpetual()
        # pay regular trading fees/rebalance AMM cash
        self.__distribute_fees(trader, liq_amount_bc)
        
        trade_cash_after = trader.cash_cc
        trader.notify_liquidation(liq_amount_bc, px, np.abs(trade_cash_after - trade_cash_before))

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

