#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
from os import stat
from trader import Trader, CollateralCurrency
from amm import AMM
import numpy as np


class ArbitrageurBankruptError(Exception):
    """Arb Traders should not go bankrupt in this world"""
    pass

class ArbTrader(Trader):
    def __init__(self, amm: 'AMM', perp_idx : int, cc: CollateralCurrency, bitmex_perp_px: np.array, cash_cc=np.nan) -> None:
        super().__init__(amm, perp_idx, cc, cash_cc=cash_cc)

        #self.slippage_tol = 0.001
        self.bitmex_perp_px = bitmex_perp_px
        self.basis_trade_thresh = 0.0030 # 0.0075+np.random.uniform(0, 0.0025)
        self.close_threshold = 0.0001
        self.bitmex_fee = 0.0004
        self.bitmex_slippage = 0.0005 #5bps
        self.protocol_fee = 0.0008

        self.target_margin = 1 # no leverage 

        # status
        self.px_entry = [0,0,0]
        self.pnl = 0 # PnL in USD
        self.open_basis = 0

    def set_active_status(self, status):
        super().set_active_status(status)
        if not status:
            print("arb bankrupt")

    def pay_liq_fee(self, fee_amount_cc):
        """Override parent method to also calculate PnL

        Args:
            fee_amount_cc ([type]): fee in collateral currency
        """
        perp = self.amm.get_perpetual(self.perp_idx)
        fx_c2q = perp.get_collateral_to_quote_conversion()
        self.pnl -= fee_amount_cc * fx_c2q
        self.cash_cc -= fee_amount_cc

    def pay_funding(self):
        """Override parent method to also calculate PnL
        """
        rate = self.get_perpetual().get_funding_rate()
        if self.position_bc*rate==0:
            return
        cash_before = self.cash_cc
        super().pay_funding()
        coupon_cc = self.cash_cc-cash_before
        fx_c2q = self.amm.perpetual_list[self.perp_idx].get_collateral_to_quote_conversion()
        fx_b2q = self.amm.perpetual_list[self.perp_idx].get_base_to_quote_conversion(False)
        # assume have a premium rate 1 bps on BitMEX side that we pay if long, receive if short
        dT = self.amm.perpetual_list[self.perp_idx].glbl_params['block_time_sec']/(8*60*60)
        bit_fee = -self.position_bc*0.0001*fx_b2q*dT # pay/rec 1bps on BitMEX
        # add to PnL (USD)
        self.pnl += coupon_cc*fx_c2q - bit_fee

    def notify_liquidation(self, liq_amount_bc : float, px : float, penalty_cc : float):
        """Override parent for pnl

        Args:
            liq_amount_bc (float): amount to be liquidated (signed)
            px (float): liquidation price
            penalty_cc (float) : liquidation penalty in collateral currency
        """
        self.calc_pnl_from_trade(px, liq_amount_bc, True)
        fx_c2q = self.amm.perpetual_list[self.perp_idx].get_collateral_to_quote_conversion()
        fee = penalty_cc*fx_c2q
        self.pnl -= fee


    def trade(self, dPos, is_close):
        """Override parent method to calculate PnL

        Args:
            dPos ([float]): position size change
            is_close (bool): true if close-only
        """
        px = super().trade(dPos, is_close)
        if px:
            self.calc_pnl_from_trade(px, dPos, is_close)
        return px

    def calc_pnl_from_trade(self, px, dPos, is_close):
        my_perp = self.amm.perpetual_list[self.perp_idx]
        current_time = my_perp.current_time
        px_perp = self.bitmex_perp_px[current_time]
        if not is_close:
            # store entry prices
            self.px_entry[0] = px
            # if dPos>0 we buy on protocol and sell on BitMEX, hence slippage in -sign(pos) direction
            self.px_entry[1] = px_perp*(1+np.sign(-dPos)*self.bitmex_slippage)
            self.px_entry[2] = my_perp.idx_s2[current_time]
        elif is_close:
            # calculate PnL
            K = -dPos
            
            """ pnlprotocol = K*(px - self.px_entry[0])
            feePos = np.abs(K)*(self.px_entry[2] + my_perp.idx_s2[current_time])
            fees = feePos * self.protocol_fee * self.bitmex_fee
            pnlBitMex = -K*(px_perp*(1+np.sign(-dPos)*self.bitmex_slippage) - self.px_entry[1])
            pnl = pnlprotocol + pnlBitMex - fees """
            pnl = self.calc_pnl(px, K)
            if False and pnl<0:
                # --
                s = (1+np.sign(-dPos)*self.bitmex_slippage)
                f = (1+self.bitmex_fee+self.protocol_fee)
                basis = (px/(px_perp*s)-1)*f
                print('>',basis, ', @px=', px,', perp*s=',px_perp*s, ', f=', f, ", pos =",dPos)
                #--
                print('ops')
            self.pnl += pnl

    def calc_pnl(self, px, pos):
        my_perp = self.amm.perpetual_list[self.perp_idx]
        current_time = my_perp.current_time
        px_perp = self.bitmex_perp_px[current_time]
        pnlprotocol = pos*(px - self.px_entry[0])
        feePos = np.abs(pos)*(self.px_entry[2] + my_perp.idx_s2[current_time])
        fees = feePos * (self.protocol_fee + self.bitmex_fee)
        pnlBitMex = -pos*(px_perp*(1+np.sign(pos)*self.bitmex_slippage) - self.px_entry[1])
        pnl = pnlprotocol + pnlBitMex - fees
        return pnl

    def query_trade_amount(self) -> 'tuple(float, bool)':
        """ Query how much the trader would trade given the current market.
            The trader never partially opens but partially closes (easier for PnL)
        fee : relative to position size (e.g., 0.06%)
        Returns:
            (amount to trade in base currency, close-only flag) 
        """ 
        if not self.is_active:
            return (0, False)
        
        my_perp = self.amm.perpetual_list[self.perp_idx]
        current_time = my_perp.current_time
        
        px_perp = self.bitmex_perp_px[current_time]

        # open basis trade (only if no open trade)
        basis = my_perp.get_price(0)/px_perp-1
        r = self.amm.get_perpetual(self.perp_idx).get_funding_rate()
        trade_dir = -np.sign(basis)
        is_favorable = np.sign(r) == -trade_dir
        if self.position_bc == 0 and np.abs(basis) > self.basis_trade_thresh and is_favorable:
            # enter position
            price_threshold = px_perp*(1-trade_dir*2*(self.bitmex_fee+self.protocol_fee))
            # threshold compatible with parameters?
            if not ((trade_dir>0 and px_perp>price_threshold) or \
                (trade_dir<0 and px_perp<price_threshold)):
                return (0, False)
            pos = self.find_position_size(my_perp, available_funds=self.cash_cc, dir=trade_dir, price_threshold = price_threshold)
            if np.abs(pos) < 10 * my_perp.params['fLotSizeBC']:
                return (0, False)
            #print(f"ArbTrader opening basis trade: Pos={pos} Sov price={my_perp.get_price(pos):.1f} CEX price={px_perp:.1f}")
            return (pos, False)
        
        
        # close basis trade
        is_favorable = np.sign(r) == np.sign(-self.position_bc)
        is_basis_favorable = np.abs(basis) < self.close_threshold
        if self.position_bc != 0 and (not is_favorable or is_basis_favorable):
            pos = -self.position_bc
            # we only close a trade amount that does not destroy
            # our pnl. So we try 5 different trade sizes
            minabspos = np.min((10 * my_perp.params['fLotSizeBC'], np.abs(self.position_bc)))
            count = 0
            px = my_perp.get_price(pos)
            pnl = self.calc_pnl(px, -pos)
            while pnl<0 and count<5:
                pos = pos/2
                px = my_perp.get_price(pos)
                pnl = self.calc_pnl(px, -pos)
                count += 1
            if pnl<0 and is_favorable:
                return (0, False)
            pos = np.sign(pos)*np.max((np.abs(pos), minabspos))
            return (pos, True)
        
        return (0, False)
       

    def find_position_size(self, perp : 'Perpetual', available_funds: float, dir: int, price_threshold: float) -> float:
        """Finds the position size so that available funds are not exceeded and the price is 'better' than the threshold

        Args:
            perp (Perpetual): reference to perpetual object
            available_funds (float): available funds should not be exceeded
            dir (int): direction of trade -1 short, 1 long
            price_threshold (float): perp price should not exceed this price

        Returns:
            float: [description]
        """
        max_size = available_funds / self.target_margin / perp.get_base_to_collateral_conversion(False)
        max_size = perp.scale_to_max_signed_trader_position(dir*max_size)
        px = perp.get_price(max_size)
        count = 0
        while ((dir>0 and px>price_threshold) or (dir<0 and px<price_threshold)) and count<10:
            max_size = self.amm.round_to_lot(max_size * 0.5, perp.params['fLotSizeBC'])
            px = perp.get_price(max_size)
            count = count+1
        if count==10:

            return 0
        return max_size
