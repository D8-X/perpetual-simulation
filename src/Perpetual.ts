import { LiquidityPool } from "./LiquidityPool";
import {
  CollateralCurrency,
  IPerpetualOrder,
  LiquidityPoolData,
  MarginAccount,
  PerpetualParams,
  PerpetualState,
  AMMVariables,
  MarketVariables,
} from "./types";
import {
  ema,
  hasOpenedPosition,
  hasTheSameSign,
  isCloseOnly,
  isMarketOrder,
  isStopOrder,
  keepPositionLeverageOnClose,
  require,
  roundToLot,
  shrinkToMaxPositionToClose,
  tbpsToABDK,
  tdrToABDK,
  validatePrice,
  validateStopPrice,
} from "./utils";

export class Perpetual extends LiquidityPool {
  paused: boolean = false;
  address: string;
  marginAccounts: Map<string, MarginAccount>;
  activeAccounts: Set<string> = new Set();
  fkStar: number;
  state: PerpetualState;

  premiumRatesEMA: number;
  iTradeDelaySec: number = 5;
  fSettlementS2PriceData: number;
  fSettlementS3PriceData: number;
  fCurrentTraderExposureEMA: number;
  currentMarkPremiumRate: { time: number; fPrice: number };
  fTargetAMMFundSize: number;
  fUnitAccumulatedFunding: number;
  fCurrentAMMExposureEMA: [number, number];

  fOpenInterest: number;

  params: PerpetualParams;
  iLastFundingTime: number;
  fCurrentFundingRate: number;
  FUNDING_INTERVAL_SEC = 3600 * 8;
  BASE_RATE: number;
  LAMBDA_SPREAD_CONVERGENCE = 0.8758875939388329;
  PNL_PART_FEE_RATIO = 0.25;
  iLastPriceJumpTimestamp: number;
  jumpSpreadTbps: any;
  MIN_NUM_LOTS_PER_POSITION: number;

  constructor(poolData: LiquidityPoolData, params: PerpetualParams) {
    super(poolData);
    this.params = params;
  }

  /**
   *
   * @param _order
   * @param _isApprovedExecutor
   * @returns
   */
  tradeViaOrderBook(_order: IPerpetualOrder, _isApprovedExecutor: boolean) {
    require(!this.paused, "paused");
    this.updateFundingAndPricesBefore(true);

    require(this.block.timestamp >=
      _order.submittedTimestamp + this.iTradeDelaySec ||
      _isApprovedExecutor, "delay required");

    if (isStopOrder(_order.flags)) {
      // reverts if trigger not met
      validateStopPrice(
        _order.fAmount > 0,
        this.getPerpetualMarkPrice(false),
        _order.fTriggerPrice
      );
    }
    if (!this.checkTradePreCond(_order)) {
      this.updateFundingAndPricesAfter();
      return true;
    }

    // bytes32 digest = _order._getDigest(address(this), true);
    //     executedOrCancelledOrders[digest] = MASK_ORDER_EXECUTED;
    //     // if the broker has not purchased a lot, the broker is
    // not allowed to charge fees.
    // if (
    //     _order.brokerAddr != address(0) &&
    //     brokerMap[perpetualPoolIds[_order.iPerpetualId]][_order.brokerAddr] == 0
    // ) {
    //     _order.brokerAddr = address(0);
    // }
    if (!this.trade(_order)) {
      // executedOrCancelledOrders[digest] = MASK_ORDER_CANCELLED;
      this.rebateExecutor(_order);
      this.updateFundingAndPricesAfter();
      return false;
    }
    this.updateFundingAndPricesAfter();
    return true;
  }

  /**
   *
   * @param _bUseOracle
   * @returns
   */
  getPerpetualMarkPrice(_bUseOracle: boolean) {
    const fPremiumRate = this.currentMarkPremiumRate.fPrice;
    const markPrice =
      _bUseOracle && this.state == PerpetualState.NORMAL
        ? this.getSafeOraclePriceS2() * (1 + fPremiumRate)
        : this.fSettlementS2PriceData * (1 + fPremiumRate);
    return markPrice;
  }

  /**
   *
   * @param _order
   * @returns
   */
  checkTradePreCond(_order: IPerpetualOrder) {
    if (this.state == PerpetualState.EMERGENCY) {
      // perpetual should be in NORMAL state
      // modifier might set the state to emergency, so in this case we return but do not
      // revert
      return false;
    }
    require(this.state == PerpetualState.NORMAL, "state should be NORMAL");
    return true;
  }

  trade(_order: IPerpetualOrder) {
    let isClose = false;

    let fPrice = 0;
    let fPnLCC = 0;
    let fNewPos = 0;
    {
      // shrink trade-amount if close-only trade and amount too large
      // override trade amount in order
      [fPrice, _order.fAmount] = this.preTrade(_order);
      if (fPrice == 0) {
        return false;
      }
      // override trade amount in order
      // if the traders closes/shrinks the position, we immediately exchange P&L into
      // collateral currency, and update the margin-cash
      let fTraderPos = this.marginAccounts.get(_order.traderAddr)!.fPositionBC;
      fNewPos = fTraderPos + _order.fAmount;
      // ensure no rounding issues:
      if (Math.abs(fNewPos) < this.params.fLotSizeBC) {
        fNewPos = 0;
      }
      isClose = hasOpenedPosition(fNewPos, _order.fAmount);

      // calculate and withdraw/deposit margin collateral depending on trade leverage choice
      // if deposit fails, the call reverts
      this.doMarginCollateralActions(fTraderPos, fPrice, _order);

      if (!isClose || Math.abs(_order.fAmount) > Math.abs(fTraderPos)) {
        // we are flipping the position sign or increasing the trade size
        // update k star, ensure resulting position not too large
        this.doOpeningTradeActions(fTraderPos, fPrice, _order);
      }
      fPnLCC = this.executeTrade(
        _order.traderAddr,
        fTraderPos,
        _order.fAmount,
        fPrice,
        isClose
      );
    }
    // distribute fees
    let fFee = this.distributeFees(_order, !isClose);
    fPnLCC = fPnLCC - fFee;
    require(this.isTraderMarginSafe(
      _order.traderAddr,
      !isClose
    ), "trader margin unsafe");
    // update AMM state, then distribute fees
    this.rebalance();

    if (fNewPos == 0) {
      // trader closed position, set positionId to zero
      if (this.marginAccounts.has(_order.traderAddr)) {
        this.marginAccounts.get(_order.traderAddr)!.fLockedInValueQC = 0;
      }
      // remove trader from active accounts and withdraw deposits
      this.withdrawDepositFromMarginAccount(_order.traderAddr);
    }
    // _getUpdateLogic().updateDefaultFundTargetSize(perpetual.id);
    return true;
  }

  /**
   *
   * @param _traderAddr
   */
  withdrawDepositFromMarginAccount(_traderAddr: string) {
    const account = this.marginAccounts.get(_traderAddr);
    require(!!account &&
      account.fPositionBC == 0, "pos must be 0 to withdraw all");
    let fCashToMove = account!.fCashCC;
    account!.fCashCC = 0;
    this.activeAccounts.delete(_traderAddr);
    this.transferFromVaultToUser(_traderAddr, fCashToMove);
  }

  /**
   *
   * @param _traderAddr
   * @param _hasOpened
   * @returns
   */
  isTraderMarginSafe(_traderAddr: string, _hasOpened: boolean): boolean {
    return _hasOpened
      ? this.isInitialMarginSafe(_traderAddr)
      : this.isMarginSafe(_traderAddr);
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  isMarginSafe(_traderAddr: string): boolean {
    if ((this.marginAccounts.get(_traderAddr)?.fPositionBC ?? 0) == 0) {
      return true;
    }

    return this.getMarginBalance(_traderAddr) >= 0;
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  isInitialMarginSafe(_traderAddr: string): boolean {
    return this.getAvailableMargin(_traderAddr, true) >= 0;
  }

  /**
   *
   * @param _traderAddr
   * @param _isInitialMargin
   * @returns
   */
  getAvailableMargin(_traderAddr: string, _isInitialMargin: boolean) {
    let fInitialMarginCC = 0;
    if (
      this.marginAccounts.has(_traderAddr) &&
      this.marginAccounts.get(_traderAddr)?.fPositionBC != 0
    ) {
      // if the position remains open, we reserve the initial/maintenance margin at the current price
      if (_isInitialMargin) {
        fInitialMarginCC = this.getInitialMargin(_traderAddr);
      } else {
        fInitialMarginCC = this.getMaintenanceMargin(_traderAddr);
      }
    }
    let fAvailableMargin = this.getMarginBalance(_traderAddr);
    fAvailableMargin = fAvailableMargin - fInitialMarginCC;
    return fAvailableMargin;
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  getMaintenanceMargin(_traderAddr: string): number {
    // base to collateral currency conversion
    const atMark = _traderAddr != this.address;
    const fConversionB2C = this.getBaseToCollateralConversionMultiplier(
      atMark,
      false
    );
    const pos = this.marginAccounts.get(_traderAddr)?.fPositionBC ?? 0;
    const m = this.params.fMaintenanceMarginRate;
    return Math.abs(pos * fConversionB2C * m);
  }

  distributeFees(_order: IPerpetualOrder, _hasOpened: boolean): number {
    return this._distributeFees(
      _order.traderAddr,
      _order.executorAddr,
      _order.brokerAddr,
      _order.fAmount,
      _hasOpened
    );
  }
  private _distributeFees(
    _traderAddr: string,
    _executorAddr: string,
    _brokerAddr: string,
    _fDeltaPositionBC: number,
    _hasOpened: boolean
  ): number {
    // fees
    let fTreasuryFee = 0;
    let fPnLparticipantFee = 0;
    let fReferralRebate = 0;
    let fBrokerFee = 0;

    {
      [fPnLparticipantFee, fTreasuryFee, fReferralRebate, fBrokerFee] =
        this.calculateFees(
          _traderAddr,
          _executorAddr,
          Math.abs(_fDeltaPositionBC),
          _hasOpened,
          _brokerAddr != ""
        );
    }
    let fTotalFee =
      fPnLparticipantFee + fTreasuryFee + fReferralRebate + fBrokerFee;
    this.updateTraderMargin(_traderAddr, -fTotalFee);

    // send fee to broker
    this.transferBrokerFee(_brokerAddr, fBrokerFee);
    // transfer protocol fee and referral rebate
    this.transferProtocolFee(
      _traderAddr,
      _executorAddr,
      fPnLparticipantFee,
      fReferralRebate,
      fTreasuryFee
    );
    return fTotalFee;
  }

  calculateFees(
    _traderAddr: string,
    _executorAddr: string,
    _fDeltaPos: number,
    _hasOpened: boolean,
    _hasBroker: boolean
  ): [number, number, number, number] {
    require(_fDeltaPos >= 0, "absolute trade value required");
    // convert to collateral currency
    _fDeltaPos =
      _fDeltaPos * this.getBaseToCollateralConversionMultiplier(false, false);
    let fTreasuryFee = 0;
    let fPnLparticipantFee = 0;
    let fReferralRebate = 0;
    let fBrokerFee = 0;
    [fTreasuryFee, fPnLparticipantFee, fReferralRebate, fBrokerFee] =
      this.determineFeeInCollateral(
        _fDeltaPos,
        this.params.fReferralRebateCC,
        _traderAddr,
        _executorAddr,
        _hasBroker,
        _hasOpened
      );
    // if the trader opens the position, 'available margin' is the margin balance - initial margin
    // requirement. If the trader closes, 'available margin' is the remaining margin balance
    if (!_hasOpened) {
      const fAvailableMargin = this.getMarginBalance(_traderAddr);
      if (fAvailableMargin <= 0) {
        fPnLparticipantFee = 0;
        fTreasuryFee = 0;
        fReferralRebate = 0;
        fBrokerFee = 0;
      } else if (
        fPnLparticipantFee + fTreasuryFee + fReferralRebate + fBrokerFee >
        fAvailableMargin
      ) {
        // make sure the sum of fees = available margin
        let fRate =
          fAvailableMargin /
          (fPnLparticipantFee + fTreasuryFee + fReferralRebate + fBrokerFee);
        fTreasuryFee = fTreasuryFee * fRate;
        fReferralRebate = fReferralRebate * fRate;
        fBrokerFee = fBrokerFee * fRate;
        fPnLparticipantFee =
          fAvailableMargin - fTreasuryFee - fReferralRebate - fBrokerFee;
      }
    } else {
      //_hasOpened, get initial margin balance and ensure fees smaller
      const fAvailableMargin = this.getAvailableMargin(_traderAddr, true);
      // If the margin of the trader is not enough for fee: If trader open position, the trade will be reverted.
      require(fPnLparticipantFee +
        fTreasuryFee +
        fReferralRebate +
        fBrokerFee <=
        fAvailableMargin, "margin not enough");
    }

    return [fPnLparticipantFee, fTreasuryFee, fReferralRebate, fBrokerFee];
  }

  determineFeeInCollateral(
    _fDeltaPosCC: number,
    _fReferralRebateCC: number,
    _trader: string,
    _executor: string,
    _hasBroker: boolean,
    _hasOpened: boolean
  ): [number, number, number, number] {
    let fTreasuryFee = 0;
    let fPnLparticipantFee = 0;
    let fReferralRebate = 0;
    let fBrokerFee = 0;
    let fTotalFeeRate = 0;
    {
      let protocolFeeTbps = 0;
      const mgn = this.marginAccounts.get(_trader);
      protocolFeeTbps = mgn?.feeTbps ?? 0;
      fBrokerFee = _hasBroker
        ? tbpsToABDK(mgn?.brokerFeeTbps ?? 0) * _fDeltaPosCC
        : 0;
      if (!_hasOpened) {
        // closing position: if holding period not satisfied a penalty is added
        protocolFeeTbps += this.ammPerpLogic.holdingPeriodPenalty(
          this.block.timestamp - (mgn?.iLastOpenTimestamp ?? 0),
          this.LAMBDA_SPREAD_CONVERGENCE
        );
      }
      fTotalFeeRate = tbpsToABDK(
        protocolFeeTbps + (_hasBroker ? mgn?.brokerFeeTbps ?? 0 : 0)
      );
      // these are rates, but we save memory by reusing variables
      [fTreasuryFee, fPnLparticipantFee] =
        this.splitProtocolFee(protocolFeeTbps);
    }
    // broker fee is accounted for, what's left is protocol = treasury + pnl part
    let fTotalFee = _fDeltaPosCC * fTotalFeeRate - fBrokerFee;
    fPnLparticipantFee =
      fTotalFee * (fPnLparticipantFee / (fTreasuryFee + fPnLparticipantFee));
    fTreasuryFee = fTotalFee - fPnLparticipantFee;
    fReferralRebate = _executor != "" ? _fReferralRebateCC : 0;
    return [fTreasuryFee, fPnLparticipantFee, fReferralRebate, fBrokerFee];
  }

  splitProtocolFee(protocolFeeTbps: number): [number, number] {
    let treasuryfee = tbpsToABDK(protocolFeeTbps);
    let pnlPartFee = treasuryfee * this.PNL_PART_FEE_RATIO;
    treasuryfee = treasuryfee - pnlPartFee;
    return [treasuryfee, pnlPartFee];
  }

  transferProtocolFee(
    _traderAddr: string,
    _executorAddr: string,
    _fPnLparticipantFee: number,
    _fReferralRebate: number,
    _fTreasuryFee: number
  ) {
    require(_fPnLparticipantFee >= 0, "PnL participant should earn fee");
    require(_fReferralRebate >= 0, "executor should earn fee");
    const _liqPool = this.poolStorage;
    //update PnL participant balance, AMM Cash balance, default fund balance
    if (_liqPool.fPnLparticipantsCashCC != 0) {
      _liqPool.fPnLparticipantsCashCC =
        _liqPool.fPnLparticipantsCashCC + _fPnLparticipantFee;
    } else {
      // currently no pnl participant funds, hence add the fee to the AMM fee
      _fTreasuryFee = _fTreasuryFee + _fPnLparticipantFee;
    }

    // contribution to DF
    _liqPool.fDefaultFundCashCC = _liqPool.fDefaultFundCashCC + _fTreasuryFee;

    // executor gets margin token
    this.transferFromVaultToUser(_executorAddr, _fReferralRebate);
  }

  transferBrokerFee(_brokerAddr: string, _fBrokerFeeCC: number) {
    if (_fBrokerFeeCC == 0) {
      return;
    }
    this.transferFromVaultToUser(_brokerAddr, _fBrokerFeeCC);
  }

  executeTrade(
    _traderAddr: string,
    _fTraderPos: number,
    _fTradeAmount: number,
    _fPrice: number,
    _isClose: boolean
  ): number {
    const [fDeltaCashCC, fDeltaLockedValue] = this.getTradeDeltas(
      _traderAddr,
      _fTraderPos,
      _fTradeAmount,
      _fPrice,
      _isClose
    );
    // execute trade: update margin, position, and open interest:
    this.updateMargin(
      this.address,
      -_fTradeAmount,
      fDeltaCashCC,
      fDeltaLockedValue
    );
    this.updateMargin(
      _traderAddr,
      _fTradeAmount,
      fDeltaCashCC,
      fDeltaLockedValue
    );
    if (!_isClose) {
      // update the average position size for AMM Pool and Default Fund target size.
      // We only account for 'opening trades'
      this.updateAverageTradeExposures(_fTradeAmount + _fTraderPos);
    }
    return fDeltaCashCC;
  }

  updateAverageTradeExposures(_fTraderPos: number) {
    let fCurrentObs = 0;
    // (neg) AMM exposure (aggregated trader exposure)
    {
      fCurrentObs = -this.marginAccounts.get(this.address)!.fPositionBC;
      const iIndex = fCurrentObs > 0 ? 1 : 0;
      const fCurrentEMA = this.fCurrentAMMExposureEMA[iIndex];
      const fLambda =
        Math.abs(fCurrentObs) > Math.abs(fCurrentEMA)
          ? this.params.fDFLambda[1]
          : this.params.fDFLambda[0];
      const fMinEMA = this.params.fMinimalAMMExposureEMA;
      if (Math.abs(fCurrentObs) < fMinEMA) {
        fCurrentObs = iIndex == 0 ? -fMinEMA : fMinEMA;
      }
      this.fCurrentAMMExposureEMA[iIndex] = ema(
        fCurrentEMA,
        fCurrentObs,
        fLambda
      );
    }

    // trader exposure
    {
      fCurrentObs = Math.abs(_fTraderPos);
      const fCurrentEMA = this.fCurrentTraderExposureEMA;
      const fLambda =
        fCurrentObs > fCurrentEMA
          ? this.params.fDFLambda[1]
          : this.params.fDFLambda[0];
      const fMinEMA = this.params.fMinimalTraderExposureEMA;
      if (fCurrentObs < fMinEMA) {
        fCurrentObs = fMinEMA;
      }
      this.fCurrentTraderExposureEMA = ema(fCurrentEMA, fCurrentObs, fLambda);
    }
  }

  updateMargin(
    _traderAddr: string,
    _fDeltaPosition: number,
    _fDeltaCashCC: any,
    _fDeltaLockedInValueQC: any
  ) {
    let account: MarginAccount;
    if (this.marginAccounts.has(_traderAddr)) {
      account = this.marginAccounts.get(_traderAddr)!;
    } else {
      account = {
        fCashCC: 0,
        fLockedInValueQC: 0,
        fPositionBC: 0,
        fUnitAccumulatedFundingStart: 0,
        feeTbps: 0,
        brokerFeeTbps: 0,
        iLastOpenTimestamp: 0,
      };
    }
    let fOldPosition = account.fPositionBC;
    let fFundingPayment = 0;
    if (fOldPosition != 0) {
      fFundingPayment =
        (this.fUnitAccumulatedFunding - account.fUnitAccumulatedFundingStart) *
        fOldPosition;
    }
    //position
    account.fPositionBC = account.fPositionBC + _fDeltaPosition;
    //cash
    {
      let fNewCashCC = account.fCashCC + _fDeltaCashCC - fFundingPayment;
      if (_traderAddr != this.address && fNewCashCC < 0) {
        /* if liquidation happens too late, the trader cash becomes negative (margin used up).
                In this case, we cannot add the full amount to the AMM margin and leave the
                trader margin negative (trader will never pay). Hence we subtract the amount
                the trader cannot pay from the AMM margin (it is added previously to the AMM margin).
                */
        let fAmountOwed = -fNewCashCC;
        fNewCashCC = 0;
        const accountAMM = this.marginAccounts.get(this.address)!;
        accountAMM.fCashCC = accountAMM.fCashCC - fAmountOwed;
      }
      account.fCashCC = fNewCashCC;
    }
    // update funding start for potential next funding payment
    account.fUnitAccumulatedFundingStart = this.fUnitAccumulatedFunding;
    //locked-in value in quote currency
    account.fLockedInValueQC =
      account.fLockedInValueQC + _fDeltaLockedInValueQC;

    // adjust open interest
    {
      let fDeltaOpenInterest = 0;
      if (fOldPosition > 0) {
        fDeltaOpenInterest = -fOldPosition;
      }
      if (account.fPositionBC > 0) {
        fDeltaOpenInterest = fDeltaOpenInterest + account.fPositionBC;
      }
      this.fOpenInterest = this.fOpenInterest + fDeltaOpenInterest;
    }
  }

  getTradeDeltas(
    _traderAddr: string,
    _fTraderPos: number,
    _fTradeAmount: number,
    _fPrice: number,
    _isClose: boolean
  ): [number, number] {
    // check that market is open
    let fIndexS2 = this.fSettlementS2PriceData;
    let fPremium = _fTradeAmount * (_fPrice - fIndexS2);
    let fDeltaCashCC = -fPremium;
    let fC2Q = this.getCollateralToQuoteConversionMultiplier(false);
    fDeltaCashCC = fDeltaCashCC / fC2Q;
    let fDeltaLockedValue = _fTradeAmount * fIndexS2;
    // if we're opening a position, L <- L + delta position * price, and no change in cash account
    // otherwise, we will have a PnL from closing:
    if (_isClose) {
      require(_fTraderPos != 0, "already closed");
      let fAvgPrice = this.getLockedInValue(_traderAddr);
      require(fAvgPrice != 0, "cannot be closing if no exposure");

      fAvgPrice = Math.abs(fAvgPrice / _fTraderPos);
      // PnL = new price*pos - locked-in-price*pos
      //     = avgprice*delta_pos - new_price*delta_pos
      //     = avgprice*delta_pos - _fDeltaLockedValue
      let fPnL = fAvgPrice * _fTradeAmount - fDeltaLockedValue;
      // The locked-in-value should change proportionally to the amount that is closed:
      // delta LockedIn = delta position * avg price
      // delta LockedIn = delta position * price + PnL
      // Since we have delta LockedIn = delta position * price up to this point,
      // it suffices to add the PnL from above:
      fDeltaLockedValue = fDeltaLockedValue + fPnL;
      // equivalently, L <- L * new position / old position,
      // i.e. if we are selling 10%, then the new locked in value is the 90% remaining
      fDeltaCashCC = fDeltaCashCC + fPnL / fC2Q;
    }
    return [fDeltaCashCC, fDeltaLockedValue];
  }
  getLockedInValue(_traderAddr: string): number {
    if (this.marginAccounts.has(_traderAddr)) {
      return this.marginAccounts.get(_traderAddr)!.fLockedInValueQC;
    } else {
      return 0;
    }
  }

  doOpeningTradeActions(
    _fTraderPos: number,
    _fPrice: number,
    _order: IPerpetualOrder
  ) {
    // pre condition: !isClose, i.e., the trade is (further) opening the position
    this.updateKStar();
    // if trader opens a position the total position amount should be smaller than the max amount,
    // unless the trade decreases the AMM risk.
    const maxTradeDelta = this.getMaxSignedOpenTradeSizeForPos(
      _fTraderPos,
      _order.fAmount > 0
    );
    require(Math.abs(_order.fAmount) <=
      Math.abs(maxTradeDelta), "Trade amt>max amt for trader/AMM");

    if (!this.marginAccounts.has(_order.traderAddr)) {
      this.marginAccounts.set(_order.traderAddr, {
        fCashCC: 0,
        fLockedInValueQC: 0,
        fPositionBC: 0,
        fUnitAccumulatedFundingStart: 0,
        feeTbps: 0,
        brokerFeeTbps: 0,
        iLastOpenTimestamp: 0,
      });
    }
    this.marginAccounts.get(_order.traderAddr)!.iLastOpenTimestamp =
      this.block.timestamp;
  }

  getMaxSignedOpenTradeSizeForPos(
    _fCurrentTraderPos: number,
    _isBuy: boolean
  ): number {
    const fMaxPos = this.getMaxSignedPositionSize(_isBuy);
    // having the maximal (signed) position size, we can determine the maximal trade amount
    let maxSignedTradeAmount = fMaxPos - _fCurrentTraderPos;
    if (
      (_isBuy && maxSignedTradeAmount < 0) ||
      (!_isBuy && maxSignedTradeAmount > 0)
    ) {
      maxSignedTradeAmount = 0;
    } else {
      // we allow for up to 1 k-star, even if this means the max position is exceeded
      const fkStar = this.fkStar;
      if (maxSignedTradeAmount > 0 && fkStar > maxSignedTradeAmount) {
        maxSignedTradeAmount = fkStar;
      } else if (maxSignedTradeAmount < 0 && fkStar < maxSignedTradeAmount) {
        maxSignedTradeAmount = fkStar;
      }
    }
    return maxSignedTradeAmount;
  }

  getMaxSignedPositionSize(isLong: boolean): number {
    let fPosSize = this.fCurrentTraderExposureEMA;
    require(fPosSize > 0, "fCurrentTraderExposureEMA>0");
    let scale = this.params.fMaximalTradeSizeBumpUp;
    const fkStar = this.fkStar;
    if ((isLong && fkStar < 0) || (!isLong && fkStar > 0)) {
      // trade in adverse direction: what is the maximal position?
      const liqPool = this.poolStorage;
      let fTotalDefaultFunds =
        liqPool.fDefaultFundCashCC + liqPool.fBrokerFundCashCC;
      let fPnLParticipantCash =
        this.getCollateralTokenAmountForPricing(liqPool);
      // account for excess LP funds
      if (fPnLParticipantCash > liqPool.fTargetAMMFundSize) {
        fTotalDefaultFunds =
          fTotalDefaultFunds +
          (fPnLParticipantCash - liqPool.fTargetAMMFundSize);
      }
      let fundingRatio = fTotalDefaultFunds / liqPool.fTargetDFSize;

      if (fundingRatio > 1) {
        fundingRatio = 1;
      }
      // if default fund < target: scale = fundingratio * BumpUp
      // if default fund > target: scale = BumpUp
      scale = scale * fundingRatio;
    }
    // maxAbs = emwaTraderK*bumpUp or reduced bumpUp
    // maxAbs = emwaTraderK*bumpUp or reduced bumpUp
    fPosSize = isLong ? fPosSize * scale : -fPosSize * scale;
    return fPosSize;
  }

  doMarginCollateralActions(
    _fTraderPos: number,
    _fPrice: number,
    _order: IPerpetualOrder
  ) {
    // determine and set fee for treasury
    let totalFeeRateTbps: number = this.setExchangeFee(_order);
    // determine and set fee for broker + update the volume EMA relevant for fees
    totalFeeRateTbps =
      totalFeeRateTbps + this.setBrokerFeeAndUpdateVolumeEMA(_order);
    if (
      Math.abs(_fTraderPos + _order.fAmount) < this.params.fLotSizeBC ||
      _order.leverageTDR == 0
    ) {
      // nothing to do, no leverage set or resulting position is zero
      return;
    }
    // determine target leverage
    const isOpen = hasTheSameSign(_order.fAmount, _fTraderPos);
    const isFlip = Math.abs(_order.fAmount) > Math.abs(_fTraderPos) && !isOpen;

    if (!isOpen && !isFlip && !keepPositionLeverageOnClose(_order.flags)) {
      // the order trades towards closing the position,
      // does not flip the position sign, and the
      // order instructions are to not touch the margin collateral
      // (!keepPositionLeverageOnClose)
      return;
    }
    // now the leverage is either position leverage if !isFlip and !isOpen,
    // or trade leverage if isFlip. For position leverage we pass 0.
    const fTargetLeverage =
      isFlip || isOpen ? tdrToABDK(_order.leverageTDR) : 0;
    let fDeposit = this.calcMarginForTargetLeverage(
      _fTraderPos,
      _fPrice,
      _order.fAmount,
      fTargetLeverage,
      _order.traderAddr,
      isOpen
    );
    // correct for fees
    let fTotalFee = 0;
    {
      const fx = this.getBaseToCollateralConversionMultiplier(false, false);
      const fTradeAmountCC = Math.abs(_order.fAmount) * fx;
      fTotalFee = fTradeAmountCC * tbpsToABDK(totalFeeRateTbps);
      if (_order.executorAddr != "") {
        fTotalFee = fTotalFee + this.params.fReferralRebateCC;
      }
    }
    fDeposit = fDeposit + fTotalFee;
    if (fDeposit > 0) {
      // can revert:
      this.depositMarginForOpeningTrade(fDeposit, _order);
    } else if (fDeposit < 0) {
      this.reduceMarginCollateral(_order.traderAddr, Math.abs(fDeposit));
    }
  }

  setBrokerFeeAndUpdateVolumeEMA(_order: IPerpetualOrder): number {
    //   if (_order.brokerAddr == address(0x0) || _order.brokerSignature.length == 0) {
    //     // broker did not sign
    //     marginAccounts[_order.iPerpetualId][_order.traderAddr].brokerFeeTbps = 0;
    //     // update volume EMA (without broker)
    //     _getBrokerFeeLogic().updateVolumeEMAOnNewTrade(
    //         _order.iPerpetualId,
    //         _order.traderAddr,
    //         address(0x0),
    //         _order.fAmount
    //     );
    //     return 0;
    // }
    // // pre-condition: signature was checked in _setExchangeFee. That is, if we are here,
    // // we can be sure that the broker signature is correct.
    // marginAccounts[_order.iPerpetualId][_order.traderAddr].brokerFeeTbps = _order
    //     .brokerFeeTbps;
    // // update volume EMA (with broker)
    // _getBrokerFeeLogic().updateVolumeEMAOnNewTrade(
    //     _order.iPerpetualId,
    //     _order.traderAddr,
    //     _order.brokerAddr,
    //     _order.fAmount
    // );
    // return _order.brokerFeeTbps;
    return 0;
  }

  setExchangeFee(_order: IPerpetualOrder): number {
    const fee = this.determineExchangeFee(_order);
    if (!this.marginAccounts.has(_order.traderAddr)) {
      this.marginAccounts.set(_order.traderAddr, {
        fCashCC: 0,
        fLockedInValueQC: 0,
        fPositionBC: 0,
        fUnitAccumulatedFundingStart: 0,
        feeTbps: 0,
        brokerFeeTbps: 0,
        iLastOpenTimestamp: 0,
      });
    }
    this.marginAccounts.get(_order.traderAddr)!.feeTbps = fee;
    return fee;
  }

  determineExchangeFee(_order: IPerpetualOrder): number {
    //   if (_order.brokerAddr == address(0x0) || _order.brokerSignature.length == 0) {
    //     // no broker or no signature
    //     return _determineExchangeFee(0, _order.traderAddr, address(0x0));
    // }
    // // broker. Check signature
    // uint8 poolId = perpetualPoolIds[_order.iPerpetualId];
    // // the signer of the signature is the broker
    // bytes32 digest = _order._getBrokerDigest(address(this));
    // address signatory = ECDSA.recover(digest, _order.brokerSignature);
    // require(signatory == _order.brokerAddr, "signature mismatch");
    // return _determineExchangeFee(poolId, _order.traderAddr, _order.brokerAddr);
    let fFeeTbps = this.getTraderInducedFee(_order.traderAddr);
    if (_order.brokerAddr != "") {
      const fBrokerInducedFee = this.getBrokerInducedFee(_order.brokerAddr);
      // charge the lower of the two
      if (fBrokerInducedFee < fFeeTbps) {
        fFeeTbps = fBrokerInducedFee;
      }
    }
    return fFeeTbps;
  }

  getBrokerInducedFee(_brokerAddr: any) {
    if (_brokerAddr == "") {
      return 0;
    }
    return this.params.fBrokerFee;
  }

  getTraderInducedFee(_traderAddr: string) {
    // uint16 fFee1 = _getFeeForStake(_traderAddr, traderTiers, traderFeesTbps);
    // uint16 fFee2 = _getFeeForTraderVolume(_poolId, _traderAddr);
    // return fFee1 > fFee2 ? fFee1 : fFee2;
    return this.params.fTradingFee;
  }

  calcMarginForTargetLeverage(
    _fTraderPos: number,
    _fPrice: number,
    _fTradeAmountBC: number,
    _fTargetLeverage: number,
    _traderAddr: string,
    _ignorePosBalance: boolean
  ): number {
    // determine current position leverage
    const fC2Q = this.getCollateralToQuoteConversionMultiplier(false);

    const fS2Mark = this.getPerpetualMarkPrice(false);

    const b0 = _ignorePosBalance ? 0 : this.getMarginBalance(_traderAddr);

    if (_fTargetLeverage == 0) {
      // leverage is to be set to position leverage
      _fTargetLeverage = Math.abs(_fTraderPos) * fS2Mark;
      _fTargetLeverage = _fTargetLeverage / fC2Q / b0;
      // make sure leverage is not higher than initial margin requirement
      const fMaxLvg = 1 / this.params.fInitialMarginRate;
      if (_fTargetLeverage > fMaxLvg) {
        _fTargetLeverage = fMaxLvg;
      }
    }
    // calculate required deposit for new position
    _fTraderPos = _ignorePosBalance ? 0 : _fTraderPos;
    return this.ammPerpLogic.getDepositAmountForLvgPosition(
      _fTraderPos,
      b0,
      _fTradeAmountBC,
      _fTargetLeverage,
      _fPrice,
      fS2Mark,
      fC2Q,
      this.fSettlementS2PriceData
    );
  }

  reduceMarginCollateral(_traderAddr: string, _fAmountToWithdraw: number) {
    if (!this.marginAccounts.has(_traderAddr)) {
      this.marginAccounts.set(_traderAddr, {
        fCashCC: 0,
        fLockedInValueQC: 0,
        fPositionBC: 0,
        fUnitAccumulatedFundingStart: 0,
        feeTbps: 0,
        brokerFeeTbps: 0,
        iLastOpenTimestamp: 0,
      });
    }
    const account = this.marginAccounts.get(_traderAddr)!;
    require(_fAmountToWithdraw > 0, "reduce mgn coll must be positive");
    // amount to withdraw could be larger than margin deposit in case the trader has a very positive
    // P&L
    account.fCashCC = account.fCashCC - _fAmountToWithdraw;
    this.transferFromVaultToUser(_traderAddr, _fAmountToWithdraw);
  }

  depositMarginForOpeningTrade(
    _fDepositRequired: number,
    _order: IPerpetualOrder
  ) {
    if (_order.traderAddr != this.address) {
      this.activeAccounts.add(_order.traderAddr);
    }
    // check if allowance set and enough cash available
    // reverts
    this.isDepositAllowed(_order.traderAddr, _fDepositRequired);
    this.updateTraderMargin(_order.traderAddr, _fDepositRequired);
    this.transferFromUserToVault(_order.traderAddr, _fDepositRequired);
    return true;
  }

  isDepositAllowed(_traderAddr: string, _fDepositRequired: number) {
    /**
     * IERC20Upgradeable marginToken = IERC20Upgradeable(_pool.marginTokenAddress);
        uint256 amountWei = _fAmount.toUDecN(_pool.marginTokenDecimals);
        uint256 balance = marginToken.balanceOf(_userAddr);
        require(balance >= amountWei, "balance not enough");
        uint256 allowance = marginToken.allowance(_userAddr, address(this));
        require(allowance >= amountWei, "allowance not enough");
     */
    return true;
  }

  preTrade(_order: IPerpetualOrder): [number, number] {
    // round the trade amount to the next lot size
    let _fAmount = roundToLot(_order.fAmount, this.params.fLotSizeBC);
    require(Math.abs(_fAmount) > 0, "trade amount too small");

    const fTraderPos =
      this.marginAccounts.get(_order.traderAddr)?.fPositionBC ?? 0;
    // don't leave dust. If the resulting position is smaller than minimal size,
    // we close the position (if closing) or revert (if opening)
    let closePos = fTraderPos != 0 && !hasTheSameSign(fTraderPos, _fAmount);
    if (
      Math.abs(fTraderPos + _fAmount) <
      this.MIN_NUM_LOTS_PER_POSITION * this.params.fLotSizeBC
    ) {
      if (closePos) {
        // the position size is adjusted to a full close.
        _fAmount = -fTraderPos;
      } else {
        // cannot open a position below minimal size
        require(false, "position too small");
      }
    } else {
      // closing but resulting position is not small enough to be a full close
      closePos = false;
    }
    // handle close only flag or dust
    if (closePos || isCloseOnly(_order.flags)) {
      _fAmount = shrinkToMaxPositionToClose(fTraderPos, _fAmount);
      require(_fAmount != 0, "no amount to close");
    }
    // query price from AMM
    const fPrice = this.queryPriceFromAMM(_fAmount);
    if (!validatePrice(_fAmount >= 0, fPrice, _order.fLimitPrice)) {
      if (isMarketOrder(_order.flags)) {
        return [0, _fAmount];
      } else {
        require(false, "price exceeds limit");
      }
    }
    return [fPrice, _fAmount];
  }

  queryPriceFromAMM(_fAmount: number) {
    const [ammState, marketState] = this.prepareAMMAndMarketData(false);
    return this.queryPriceGivenAMMAndMarketData(
      _fAmount,
      ammState,
      marketState
    );
  }

  queryPriceGivenAMMAndMarketData(
    _fTradeAmount: number,
    _ammState: AMMVariables,
    _marketState: MarketVariables
  ) {
    require(this.fCurrentTraderExposureEMA > 0, "pos size EMA is non-positive");

    // funding status
    let fCurrSpread = 0;
    let fIncentiveSpread = 0;
    if (_fTradeAmount != 0) {
      fCurrSpread = this.calculateVolatilitySpread();
      fIncentiveSpread = tbpsToABDK(this.params.incentiveSpreadTbps);
    }
    return this.ammPerpLogic.calculatePerpetualPrice(
      _ammState,
      _marketState,
      _fTradeAmount,
      fCurrSpread,
      fIncentiveSpread
    );
  }

  calculateVolatilitySpread(): number {
    const numSecSinceJump = this.block.timestamp - this.iLastPriceJumpTimestamp;

    return this.ammPerpLogic.volatilitySpread(
      this.jumpSpreadTbps,
      this.params.minimalSpreadTbps,
      numSecSinceJump,
      this.LAMBDA_SPREAD_CONVERGENCE
    );
  }

  /**
   *
   * @param _order
   */
  rebateExecutor(_order: IPerpetualOrder) {
    this.transferFromUserToVault(
      _order.traderAddr,
      this.params.fReferralRebateCC
    );
    this.transferFromVaultToUser(
      _order.executorAddr,
      this.params.fReferralRebateCC
    );
  }

  /**
   *
   * @param revertIfClosed
   */
  updateFundingAndPricesBefore(revertIfClosed: boolean) {
    this.checkOracleStatus(revertIfClosed);
    this.accumulateFundingInPerp();
  }

  /**
   *
   * @returns
   */
  accumulateFundingInPerp() {
    let iTimeElapsed = this.iLastFundingTime;
    if (
      this.block.timestamp <= iTimeElapsed ||
      this.state != PerpetualState.NORMAL
    ) {
      // already updated or not running
      return;
    }
    // block.timestamp > iLastFundingTime, so safe:
    iTimeElapsed = this.block.timestamp - iTimeElapsed;

    // determine payment in collateral currency for 1 unit of base currency \
    // (e.g. USD payment for 1 BTC for BTCUSD)
    let fInterestPaymentLong =
      (iTimeElapsed * this.fCurrentFundingRate) / this.FUNDING_INTERVAL_SEC;
    // fInterestPaymentLong will be applied to 'base currency 1' (multiply with position size)
    // Finally, we convert this payment from base currency into collateral currency
    const fConversion = this.getBaseToCollateralConversionMultiplier(
      false,
      false
    );
    fInterestPaymentLong = fInterestPaymentLong * fConversion;
    this.fUnitAccumulatedFunding =
      this.fUnitAccumulatedFunding + fInterestPaymentLong;
  }

  checkOracleStatus(revertIfClosed: boolean) {
    throw new Error("Method not implemented.");
  }

  getSafeOraclePriceS3(): number {
    throw new Error("Method not implemented.");
  }

  getSafeOraclePriceS2(): number {
    throw new Error("Method not implemented.");
  }

  /**
   *
   */
  updateFundingAndPricesAfter() {
    this.updateFundingRatesInPerp();
  }

  /**
   *
   * @returns
   */
  updateFundingRatesInPerp() {
    if (
      this.iLastFundingTime >= this.block.timestamp ||
      this.state != PerpetualState.NORMAL
    ) {
      // invalid time or not running
      return;
    }
    this.updateFundingRate();
    //update iLastFundingTime (we need it in _accumulateFundingInPerp and _updateFundingRatesInPerp)
    this.iLastFundingTime = this.block.timestamp;
  }

  /**
   *
   */
  updateFundingRate() {
    let fFundingRate = 0;
    let fBase = 0;
    {
      const fFundingRateClamp = this.params.fFundingRateClamp;
      let fPremiumRate = this.getMarkPremiumRateEMA();
      // clamp the rate
      const K2 = -this.marginAccounts.get(this.address)!.fPositionBC;
      if (fPremiumRate > fFundingRateClamp) {
        // r > 0 applies only if also K2 > 0
        fPremiumRate = K2 > 0 ? fPremiumRate : fFundingRateClamp;
        fFundingRate = fPremiumRate - fFundingRateClamp;
      } else if (fPremiumRate < -fFundingRateClamp) {
        // r < 0 applies only if also K2 < 0
        fPremiumRate = K2 < 0 ? fPremiumRate : -fFundingRateClamp;
        fFundingRate = fPremiumRate + fFundingRateClamp;
      }
      fBase = K2 >= 0 ? this.BASE_RATE : -this.BASE_RATE;
    }

    fFundingRate = fFundingRate + fBase;
    this.fCurrentFundingRate = fFundingRate;
  }

  /**
   *
   * @returns
   */
  getMarkPremiumRateEMA() {
    return this.currentMarkPremiumRate.fPrice;
  }

  /**
   *
   * @returns
   */
  rebalance() {
    if (this.state !== PerpetualState.NORMAL) {
      return;
    }
    this.equalizeAMMMargin();
    this.updateAMMTargetFundSize();
    this.updateMarkPrice();
    this.updateKStar();
  }

  /**
   *
   */
  equalizeAMMMargin() {
    const [fMarginBalance, fInitialBalance] = this.getRebalanceMargin();
    if (fMarginBalance > fInitialBalance) {
      // from margin to pool
      this.transferFromAMMMarginToPool(fMarginBalance - fInitialBalance);
    } else {
      // from pool to margin
      // It's possible that there are not enough funds to draw from
      // in this case not the full margin will be replenished
      // (and emergency state is raised)
      this.transferFromPoolToAMMMargin(
        fInitialBalance - fMarginBalance,
        fMarginBalance
      );
    }
  }

  /**
   *
   */
  updateMarkPrice() {
    const fCurrentPremiumRate = this.calcInsurancePremium();
    this.updatePremiumMarkPrice(fCurrentPremiumRate);
    this.premiumRatesEMA = ema(
      this.premiumRatesEMA,
      fCurrentPremiumRate,
      this.params.fMarkPriceEMALambda
    );
  }

  /**
   *
   * @returns
   */
  calcInsurancePremium() {
    const [ammState, marketState] = this.prepareAMMAndMarketData(false);
    let px_premium = this.ammPerpLogic.calculatePerpetualPrice(
      ammState,
      marketState,
      0,
      0,
      0
    );
    px_premium =
      (px_premium - marketState.fIndexPriceS2) / marketState.fIndexPriceS2;
    return px_premium;
  }

  /**
   *
   * @param _bUseOracle
   * @returns
   */
  prepareAMMAndMarketData(
    _bUseOracle: boolean
  ): [AMMVariables, MarketVariables] {
    // prepare data

    let marketState: MarketVariables = {
      fIndexPriceS2: _bUseOracle
        ? this.getSafeOraclePriceS2()
        : this.fSettlementS2PriceData,
      fIndexPriceS3: _bUseOracle
        ? this.getSafeOraclePriceS3()
        : this.fSettlementS3PriceData,
      fSigma2: this.params.fSigma2,
      fSigma3: this.params.fSigma3,
      fRho23: this.params.fRho23,
    };

    const AMMMarginAcc = this.marginAccounts.get(this.address)!;

    let fPricingPnLCashCC = this.getPerpetualAllocatedFunds();
    // add cash from AMM margin account
    fPricingPnLCashCC = fPricingPnLCashCC + AMMMarginAcc.fCashCC;

    // get current locked-in value
    let ammState: AMMVariables = {
      fLockedValue1: -AMMMarginAcc.fLockedInValueQC,
      fAMM_K2: -AMMMarginAcc.fPositionBC,
      fPoolM1: 0, // M1 in quote currency
      fPoolM2: 0, // M2 in base currency
      fPoolM3: 0, // M3 in quanto currency
      fCurrentTraderExposureEMA: this.fCurrentTraderExposureEMA,
    };

    const ccy = this.params.eCollateralCurrency;
    if (ccy == CollateralCurrency.BASE) {
      ammState.fPoolM2 = fPricingPnLCashCC;
    } else if (ccy == CollateralCurrency.QUANTO) {
      ammState.fPoolM3 = fPricingPnLCashCC;
    } else {
      ammState.fPoolM1 = fPricingPnLCashCC;
    }
    return [ammState, marketState];
  }

  /**
   *
   * @param _fCurrentPremiumRate
   */
  updatePremiumMarkPrice(_fCurrentPremiumRate: number) {
    const iCurrentTimeSec = this.block.timestamp;
    if (
      this.currentMarkPremiumRate.time != iCurrentTimeSec &&
      this.state == PerpetualState.NORMAL
    ) {
      // no update of mark-premium rate if we are in emergency state (we want the mark-premium to be frozen)
      // update mark-price if we are in a new block
      // now set the mark price to the last block EMA
      this.currentMarkPremiumRate.time = iCurrentTimeSec;
      // assign last EMA of previous block
      this.currentMarkPremiumRate.fPrice = this.premiumRatesEMA;
    }
  }

  /**
   *
   */
  updateKStar() {
    const ccy = this.params.eCollateralCurrency;
    const AMMMarginAcc = this.marginAccounts.get(this.address)!;
    const K2 = -AMMMarginAcc.fPositionBC;
    //M1/M2/M3  = LP cash allocted + margin cash
    const fM = this.getPerpetualAllocatedFunds() + AMMMarginAcc.fCashCC;

    if (ccy == CollateralCurrency.BASE) {
      this.fkStar = fM - K2;
    } else if (ccy == CollateralCurrency.QUOTE) {
      this.fkStar = -K2;
    } else {
      const fB2C = this.getBaseToCollateralConversionMultiplier(false, false); // s2 / s3
      const nominator =
        this.params.fRho23 * this.params.fSigma2 * this.params.fSigma3;
      const denom = this.params.fSigma2 * this.params.fSigma2;
      this.fkStar = (nominator / denom / fB2C) * fM - K2;
    }
  }

  /**
   *
   */
  updateAMMTargetFundSize() {
    const liquidityPool = this.poolStorage;
    const fOldTarget = this.fTargetAMMFundSize;
    this.fTargetAMMFundSize = this.getUpdatedTargetAMMFundSize(
      this.params.eCollateralCurrency
    );
    // update total target sizes in pool data
    liquidityPool.fTargetAMMFundSize =
      liquidityPool.fTargetAMMFundSize - fOldTarget + this.fTargetAMMFundSize;
  }

  /**
   *
   * @param _ccy
   * @returns
   */
  getUpdatedTargetAMMFundSize(_ccy: CollateralCurrency): number {
    // loop through perpetuals of this pool and update the
    // pool size
    const mv: MarketVariables = {
      fIndexPriceS2: this.fSettlementS2PriceData,
      fIndexPriceS3: this.fSettlementS3PriceData,
      fSigma2: this.params.fSigma2,
      fSigma3: this.params.fSigma3,
      fRho23: this.params.fRho23,
    };
    let fMStar = 0;
    let fK = -this.marginAccounts.get(this.address)!.fPositionBC;
    let fLockedIn = -this.marginAccounts.get(this.address)!.fLockedInValueQC;
    // adjust current K and L for EMA trade size:
    // kStarSide = kStar > 0 ? 1 : -1;
    if (this.fkStar < 0) {
      fK = fK + this.fCurrentTraderExposureEMA;
      fLockedIn = fLockedIn + this.fCurrentTraderExposureEMA * mv.fIndexPriceS2;
    } else {
      fK = fK - this.fCurrentTraderExposureEMA;
      fLockedIn = fLockedIn - this.fCurrentTraderExposureEMA * mv.fIndexPriceS2;
    }

    if (_ccy == CollateralCurrency.BASE) {
      // get target collateral for current AMM exposure
      if (fK == 0 || fLockedIn == 0) {
        fMStar = this.params.fAMMMinSizeCC;
      } else {
        fMStar = this.ammPerpLogic.getTargetCollateralM2(
          fK,
          fLockedIn,
          mv,
          this.params.fAMMTargetDD
        );
      }
    } else if (_ccy == CollateralCurrency.QUANTO) {
      if (fK == 0) {
        fMStar = this.params.fAMMMinSizeCC;
      } else {
        // get target collateral for current AMM exposure
        fMStar = this.ammPerpLogic.getTargetCollateralM3(
          fK,
          fLockedIn,
          mv,
          this.params.fAMMTargetDD
        );
      }
    } else {
      if (fK == 0) {
        fMStar = this.params.fAMMMinSizeCC;
      } else {
        // get target collateral for conservative negative AMM exposure
        fMStar = this.ammPerpLogic.getTargetCollateralM1(
          fK,
          fLockedIn,
          mv,
          this.params.fAMMTargetDD
        );
      }
    }
    // M = pool + margin, target is for pool funds only:
    fMStar = fMStar - this.marginAccounts.get(this.address)!.fCashCC;
    if (fMStar < this.params.fAMMMinSizeCC) {
      fMStar = this.params.fAMMMinSizeCC;
    }
    // EMA: new M* = L x old M* + (1 - L) x spot M*, same speed as DF target (slow)
    fMStar = ema(this.fTargetAMMFundSize, fMStar, this.params.fDFLambda[0]);
    return fMStar;
  }

  /**
   *
   * @returns
   */
  getRebalanceMargin() {
    const fInitialMargin = this.getInitialMargin(this.address);
    const fMarginBalance = this.getMarginBalance(this.address);
    return [fMarginBalance, fInitialMargin];
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  getMarginBalance(_traderAddr: string) {
    const atMark =
      _traderAddr != this.address || this.state != PerpetualState.NORMAL;
    // base to collateral currency conversion
    const fConversionB2C = this.getBaseToCollateralConversionMultiplier(
      atMark,
      false
    );
    // quote to collateral currency conversion
    const fConversionC2Q = this.getCollateralToQuoteConversionMultiplier(false);
    const fLockedInValueCC =
      (this.marginAccounts.get(_traderAddr)?.fLockedInValueQC ?? 0) /
      fConversionC2Q;
    const fCashCC = this.getAvailableCash(_traderAddr);
    let fMargin =
      (this.marginAccounts.get(_traderAddr)?.fPositionBC ?? 0) *
        fConversionB2C +
      fCashCC;
    fMargin = fMargin - fLockedInValueCC;
    return fMargin;
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  getAvailableCash(_traderAddr: string): number {
    const account = this.marginAccounts.get(_traderAddr);
    const fCashCC = account?.fCashCC ?? 0;
    // unit-funding is in collateral currency
    const fFundingUnitPayment =
      this.fUnitAccumulatedFunding -
      (account?.fUnitAccumulatedFundingStart ?? 0);
    return fCashCC - (account?.fPositionBC ?? 0) * fFundingUnitPayment;
  }

  /**
   *
   * @param _bUseOracle
   * @returns
   */
  getCollateralToQuoteConversionMultiplier(_bUseOracle: boolean): number {
    const ccy = this.params.eCollateralCurrency;
    if (ccy == CollateralCurrency.BASE) {
      return _bUseOracle && this.state == PerpetualState.NORMAL
        ? this.getSafeOraclePriceS2()
        : this.fSettlementS2PriceData;
    }
    if (ccy == CollateralCurrency.QUANTO) {
      return _bUseOracle && this.state == PerpetualState.NORMAL
        ? this.getSafeOraclePriceS3()
        : this.fSettlementS3PriceData;
    } else {
      return 1;
    }
  }

  /**
   *
   * @param _traderAddr
   * @returns
   */
  getInitialMargin(_traderAddr: string) {
    const isMark = _traderAddr != this.address;
    // base to collateral currency conversion
    const fConversionB2C = this.getBaseToCollateralConversionMultiplier(
      isMark,
      false
    );
    const pos = this.marginAccounts.get(_traderAddr)?.fPositionBC ?? 0;
    const m = this.params.fInitialMarginRate;
    return Math.abs(pos * fConversionB2C * m);
  }

  /**
   *
   * @param fAmount
   * @returns
   */
  transferFromAMMMarginToPool(fAmount: number) {
    if (fAmount == 0) {
      return;
    }
    require(fAmount > 0, "transferFromAMMMgnToPool >0");
    const pool = this.poolStorage;
    // update margin of AMM
    this.updateTraderMargin(this.address, -fAmount);

    // split amount ensures PnL part and DF split profits according to their relative sizes
    const [fPnLparticipantAmount, fDFAmount] = this.splitAmount(
      pool,
      fAmount,
      false
    );
    this.increasePoolCash(pool, fPnLparticipantAmount);
    pool.fDefaultFundCashCC = pool.fDefaultFundCashCC + fDFAmount;
  }

  /**
   *
   * @param _liquidityPool
   * @param _fAmount
   */
  increasePoolCash(_liquidityPool: LiquidityPoolData, _fAmount: number) {
    require(_fAmount >= 0, "inc neg pool cash");
    _liquidityPool.fPnLparticipantsCashCC =
      _liquidityPool.fPnLparticipantsCashCC + _fAmount;
  }

  /**
   *
   * @param _liquidityPool
   * @param _fAmount
   * @param _isWithdrawn
   * @returns
   */
  splitAmount(
    _liquidityPool: LiquidityPoolData,
    _fAmount: number,
    _isWithdrawn: boolean
  ) {
    if (_fAmount == 0) {
      return [0, 0];
    }
    let [fAmountPnLparticipants, fAmountDF] = [0, 0];

    {
      // will divide this by fAvailCash below
      let fWeightPnLparticipants =
        this.getCollateralTokenAmountForPricing(_liquidityPool);
      const fAvailCash =
        fWeightPnLparticipants + _liquidityPool.fDefaultFundCashCC;
      require(_fAmount > 0, ">0 amount expected");
      require(!_isWithdrawn || fAvailCash >= _fAmount, "pre-cond not met");
      fWeightPnLparticipants = fWeightPnLparticipants / fAvailCash;
      const fCeilPnLShare = _liquidityPool.fCeilPnLShare;
      // ceiling for PnL participant share of PnL
      if (fWeightPnLparticipants > fCeilPnLShare) {
        fWeightPnLparticipants = fCeilPnLShare;
      }

      fAmountPnLparticipants = fWeightPnLparticipants * _fAmount;
      fAmountDF = _fAmount - fAmountPnLparticipants;
    }

    // ensure we have have non-negative funds when withdrawing
    // re-distribute otherwise
    if (_isWithdrawn) {
      // pre-condition: _fAmount<available PnLparticipantCash+dfcash
      // because of CEIL_PNL_SHARE we might allocate too much to DF
      // fix this here
      let fSpillover = _liquidityPool.fDefaultFundCashCC - fAmountDF;
      if (fSpillover < 0) {
        fSpillover = -fSpillover;
        fAmountDF = fAmountDF - fSpillover;
        fAmountPnLparticipants = fAmountPnLparticipants + fSpillover;
      }
    }

    return [fAmountPnLparticipants, fAmountDF];
  }

  /**
   *
   * @param _liquidityPool
   * @returns
   */
  getCollateralTokenAmountForPricing(_liquidityPool: LiquidityPoolData) {
    if (this.poolStorage.totalSupplyShareToken == 0) {
      return 0;
    }
    const pnlPartCash = this.poolStorage.fPnLparticipantsCashCC;
    const shareProportion =
      this.getShareTokenAmountForPricing() /
      this.poolStorage.totalSupplyShareToken;
    return shareProportion * pnlPartCash;
  }

  /**
   *
   * @param _traderAddr
   * @param _fDeltaCash
   * @returns
   */
  updateTraderMargin(_traderAddr: string, _fDeltaCash: number) {
    if (_fDeltaCash == 0) {
      return;
    }
    if (this.marginAccounts.has(_traderAddr)) {
      const account = this.marginAccounts.get(_traderAddr)!;
      account.fCashCC = account.fCashCC + _fDeltaCash;
    } else {
      this.marginAccounts.set(_traderAddr, {
        fCashCC: _fDeltaCash,
        fLockedInValueQC: 0,
        fPositionBC: 0,
        fUnitAccumulatedFundingStart: 0,
        feeTbps: 0,
        brokerFeeTbps: 0,
        iLastOpenTimestamp: 0,
      });
    }
  }

  /**
   *
   * @param _fAmount
   * @param _fMarginBalance
   * @returns
   */
  transferFromPoolToAMMMargin(_fAmount: number, _fMarginBalance: number) {
    // transfer from pool to AMM: amount >= 0
    if (_fAmount == 0) {
      return 0;
    }
    require(_fAmount > 0, "transferFromPoolToAMM>0");
    // perpetual state cannot be normal with 0 cash
    const pool = this.poolStorage;
    const fPnLPartFunds = this.getCollateralTokenAmountForPricing(pool);
    require(pool.fDefaultFundCashCC > 0 ||
      fPnLPartFunds > 0 ||
      this.state != PerpetualState.NORMAL, "state abnormal: 0 DF Cash");
    // we first withdraw from the broker fund
    const fBrokerAmount = this.withdrawFromBrokerPool(_fAmount);
    let fFeasibleMargin = fBrokerAmount;
    if (fBrokerAmount < _fAmount) {
      // now we aim to withdraw _fAmount - fBrokerAmount from the liquidity pools
      // fDFAmount, fLPAmount will give us the amount that can be withdrawn
      const [fDFAmount, fLPAmount] =
        this.getFeasibleTransferFromPoolToAMMMargin(
          _fAmount - fBrokerAmount,
          _fMarginBalance,
          fPnLPartFunds,
          pool
        );
      fFeasibleMargin = fFeasibleMargin + fLPAmount + fDFAmount;
    }
    // update margin
    this.updateTraderMargin(this.address, fFeasibleMargin);
    return fFeasibleMargin;
  }

  /**
   *
   * @param _fAmount
   * @param _fMarginBalance
   * @param _fPnLPartFunds
   * @param _pool
   * @returns
   */
  getFeasibleTransferFromPoolToAMMMargin(
    _fAmount: number,
    _fMarginBalance: number,
    _fPnLPartFunds: number,
    _pool: LiquidityPoolData
  ) {
    // perp funds coming from the liquidity pool
    const fPoolFunds = this.getPerpetualAllocatedFunds();
    let [fLPAmount, fDFAmount] = [0, 0];
    if (fPoolFunds + _fMarginBalance > 0) {
      // the AMM has a positive margin balance when accounting for all allocated funds
      // -> funds are transferred to the margin, capped at:
      // 1) available amount, and  2) no more than to keep AMM at initial margin
      [fLPAmount, fDFAmount] = this.splitAmount(
        _pool,
        fPoolFunds > _fAmount ? _fAmount : fPoolFunds,
        true
      );
    } else {
      // AMM has lost all its allocated funds: emergency state
      // 1) all LP funds are used, 2) DF covers what if left
      fLPAmount = fPoolFunds;
      fDFAmount = -(fPoolFunds + _fMarginBalance);
      if (_fPnLPartFunds > _pool.fTargetAMMFundSize) {
        // there are some LP funds allocated to the DF -> split what DF pays
        const fDFWeight =
          _pool.fDefaultFundCashCC /
          (_pool.fDefaultFundCashCC +
            _fPnLPartFunds -
            _pool.fTargetAMMFundSize);
        fLPAmount = (fLPAmount + fDFAmount) * (1 - fDFWeight);
        fDFAmount = fDFAmount * fDFWeight;
      }
      this.setEmergencyState();
    }
    // ensure DF is not depleted: PnL sharing cap may cause DF to overc-contribute
    //-> PnL participants cover the rest
    if (fDFAmount > _pool.fDefaultFundCashCC) {
      fLPAmount = fLPAmount + (fDFAmount - _pool.fDefaultFundCashCC);
      fDFAmount = _pool.fDefaultFundCashCC;
    }
    // ensure LPs can cover total: otherwise stop the pool
    if (fLPAmount >= _fPnLPartFunds) {
      // liquidity pool is depleted
      fLPAmount = _fPnLPartFunds;
      this.setLiqPoolEmergencyState();
    }
    this.decreaseDefaultFundCash(fDFAmount);
    this.decreasePoolCash(fLPAmount);
    // this function returns (fDFAmount, fLPAmount);
    return [fDFAmount, fLPAmount];
  }

  /**
   *
   */
  setEmergencyState() {
    this.state = PerpetualState.EMERGENCY;
  }

  /**
   *
   * @returns
   */
  getPerpetualAllocatedFunds() {
    if (this.fTargetAMMFundSize <= 0) {
      return 0;
    }
    const pool = this.poolStorage;
    let fPricingCash = this.getCollateralTokenAmountForPricing(pool);
    if (fPricingCash <= 0) {
      return 0;
    }
    let fFunds = 0;
    if (fPricingCash > pool.fTargetAMMFundSize) {
      fFunds = this.fTargetAMMFundSize;
    } else {
      fFunds =
        fPricingCash * (this.fTargetAMMFundSize / pool.fTargetAMMFundSize);
    }
    return fFunds;
  }

  /**
   *
   * @param _isMarkPriceRequest
   * @param _bUseOracle
   * @returns
   */
  getBaseToCollateralConversionMultiplier(
    _isMarkPriceRequest: boolean,
    _bUseOracle: boolean
  ) {
    const ccy = this.params.eCollateralCurrency;
    /*
        Quote: Pos * markprice --> quote currency
        Base: Pos * markprice / indexprice; E.g., 0.1 BTC * 36500 / 36000
        Quanto: Pos * markprice / index3price. E.g., 0.1 BTC * 36500 / 2000 = 1.83 ETH
        where markprice is replaced by indexprice if _isMarkPriceRequest=FALSE
        */
    let fPx2: number;
    let fPxIndex2: number;
    if (!_bUseOracle || this.state != PerpetualState.NORMAL) {
      fPxIndex2 = this.fSettlementS2PriceData;
      require(fPxIndex2 > 0, "settl px S2 not set");
    } else {
      fPxIndex2 = this.getSafeOraclePriceS2();
    }

    if (_isMarkPriceRequest) {
      fPx2 = this.getPerpetualMarkPrice(_bUseOracle);
    } else {
      fPx2 = fPxIndex2;
    }

    if (ccy == CollateralCurrency.BASE) {
      // equals ONE if _isMarkPriceRequest=FALSE
      return fPx2 / fPxIndex2;
    }
    if (ccy == CollateralCurrency.QUANTO) {
      // Example: 0.5 contracts of ETHUSD paid in BTC
      //  the rate is ETHUSD * 1/BTCUSD
      //  BTCUSD = 31000 => 0.5/31000 = 0.00003225806452 BTC
      return _bUseOracle && this.state == PerpetualState.NORMAL
        ? fPx2 / this.getSafeOraclePriceS3()
        : fPx2 / this.fSettlementS3PriceData;
    } else {
      // Example: 0.5 contracts of ETHUSD paid in USD
      //  the rate is ETHUSD
      //  ETHUSD = 2000 => 0.5 * 2000 = 1000
      require(ccy == CollateralCurrency.QUOTE, "unknown state");
      return fPx2;
    }
  }
}
