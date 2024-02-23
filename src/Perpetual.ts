import { LiquidityPool } from "./LiquidityPool";
import {
  CollateralCurrency,
  IPerpetualOrder,
  LiquidityPoolData,
  MarginAccount,
  PerpetualParams,
  PerpetualState,
} from "./types";
import {
  AMMVariables,
  MarketVariables,
  ema,
  require,
  validateStopPrice,
} from "./utils";

export class Perpetual extends LiquidityPool {
  paused: boolean = false;
  address: string;
  marginAccounts: Map<string, MarginAccount>;
  fkStar: number;
  state: PerpetualState;

  premiumRatesEMA: number;
  iTradeDelaySec: number = 5;
  fSettlementS2PriceData: number;
  fSettlementS3PriceData: number;
  fCurrentTraderExposureEMA: number;
  currentMarkPremiumRate: any;
  fTargetAMMFundSize: number;
  fUnitAccumulatedFunding: number;

  params: PerpetualParams;
  iLastFundingTime: any;
  fCurrentFundingRate: any;
  FUNDING_INTERVAL_SEC: any;

  constructor(poolData: LiquidityPoolData, params: PerpetualParams) {
    super(poolData);
    this.params = params;
  }

  tradeViaOrderBook(_order: IPerpetualOrder, _isApprovedExecutor: boolean) {
    require(!this.paused, "paused");
    this.updateFundingAndPricesBefore(true);

    require(this.block.timestamp >=
      _order.submittedTimestamp + this.iTradeDelaySec ||
      _isApprovedExecutor, "delay required");

    if (_order.flags.isStopOrder()) {
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

  getPerpetualMarkPrice(_bUseOracle: boolean) {
    const fPremiumRate = this.currentMarkPremiumRate.fPrice;
    const markPrice =
      _bUseOracle && this.state == PerpetualState.NORMAL
        ? this.getSafeOraclePriceS2() * (1 + fPremiumRate)
        : this.fSettlementS2PriceData * (1 + fPremiumRate);
    return markPrice;
  }

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
    return true;
  }

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

  updateFundingAndPricesBefore(revertIfClosed: boolean) {
    this.checkOracleStatus(revertIfClosed);
    this.accumulateFundingInPerp();
  }

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

  updateFundingAndPricesAfter() {
    return;
  }

  rebalance() {
    if (this.state !== PerpetualState.NORMAL) {
      return;
    }
    this.equalizeAMMMargin();
    this.updateAMMTargetFundSize();
    this.updateMarkPrice();
    this.updateKStar();
  }

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

  updateMarkPrice() {
    const fCurrentPremiumRate = this.calcInsurancePremium();
    this.updatePremiumMarkPrice(fCurrentPremiumRate);
    this.premiumRatesEMA = ema(
      this.premiumRatesEMA,
      fCurrentPremiumRate,
      this.params.fMarkPriceEMALambda
    );
  }

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
  getSafeOraclePriceS3(): number {
    throw new Error("Method not implemented.");
  }
  getSafeOraclePriceS2(): number {
    throw new Error("Method not implemented.");
  }

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

  getRebalanceMargin() {
    const fInitialMargin = this.getInitialMargin(this.address);
    const fMarginBalance = this.getMarginBalance(this.address);
    return [fMarginBalance, fInitialMargin];
  }

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

  getAvailableCash(_traderAddr: string): number {
    const account = this.marginAccounts.get(_traderAddr);
    const fCashCC = account?.fCashCC ?? 0;
    // unit-funding is in collateral currency
    const fFundingUnitPayment =
      this.fUnitAccumulatedFunding -
      (account?.fUnitAccumulatedFundingStart ?? 0);
    return fCashCC - (account?.fPositionBC ?? 0) * fFundingUnitPayment;
  }

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
    pool.fDefaultFundCashCC = pool.fDefaultFundCashCC.add(fDFAmount);
  }

  increasePoolCash(_liquidityPool: LiquidityPoolData, _fAmount: number) {
    require(_fAmount >= 0, "inc neg pool cash");
    _liquidityPool.fPnLparticipantsCashCC =
      _liquidityPool.fPnLparticipantsCashCC + _fAmount;
  }

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
      });
    }
  }

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
          (_pool.fDefaultFundCashCC.add(_fPnLPartFunds) -
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

  setEmergencyState() {
    this.state = PerpetualState.EMERGENCY;
  }

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
