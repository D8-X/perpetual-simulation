import { LiquidityPool, LiquidityPoolData } from "./LiquidityPool";
import { ema, require } from "./utils";

export enum CollateralCurrency {
  BASE = 0,
  QUANTO,
  QUOTE,
}

export enum PerpetualState {
  NORMAL = 0,
  INITIALIZING,
  INVALID,
  EMERGENCY,
}

export interface MarginAccount {
  fCashCC: number;
  fPositionBC: number;
}

export interface IPerpetualOrder {
         leverageTDR; // 12.43x leverage is represented by 1243 (two-digit integer representation); 0 if deposit and trade separate
         brokerFeeTbps; // broker can set their own fee
         iPerpetualId; // global id for perpetual
         traderAddr; // address of trader
         executionTimestamp; // normally set to current timestamp; order will not be executed prior to this timestamp.
         brokerAddr; // address of the broker or zero
         submittedTimestamp;
         flags; // order flags
         iDeadline; //deadline for price (seconds timestamp)
         executorAddr; // address of the executor set by contract
         fAmount; // amount in base currency to be traded
         fLimitPrice; // limit price
         fTriggerPrice; //trigger price. Non-zero for stop orders.
         brokerSignature; //signature of broker (or 0)
}


export class Perpetual extends LiquidityPool {
  paused: boolean = false;
  address: string;
  marginAccounts: Map<string, MarginAccount>;
  eCollateralCurrency: CollateralCurrency;
  fkStar: number;
  fRho23: number;
  fSigma2: number;
  fSigma3: number;
  state: PerpetualState;
  
  premiumRatesEMA: any;
  fMarkPriceEMALambda: any;
  iTradeDelaySec: number = 5;
    fReferralRebateCC: number;

  constructor(storage: LiquidityPoolData) {
    super(storage);
  }

  tradeViaOrderBook(_order: IPerpetualOrder, _isApprovedExecutor: boolean) {
    require(!this.paused, "paused");
    this.updateFundingAndPricesBefore(true);
    
    require(
        this.block.timestamp >= _order.submittedTimestamp + this.iTradeDelaySec || _isApprovedExecutor,
        "delay required"
    );

    if (_order.flags.isStopOrder()) {
        // reverts if trigger not met
        this.validateStopPrice(
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
        if (!this.trade(_order, digest)) {
            // executedOrCancelledOrders[digest] = MASK_ORDER_CANCELLED;
            this.rebateExecutor(_order);
            this.updateFundingAndPricesAfter();
            return false;
        }
        this.updateFundingAndPricesAfter();
        return true;
    
  }
    trade(_order: IPerpetualOrder, digest: any) {
        throw new Error("Method not implemented.");
    }
    rebateExecutor(_order: IPerpetualOrder) {
        this.transferFromUserToVault(_order.traderAddr, this.fReferralRebateCC);
        this.transferFromVaultToUser(_order.executorAddr, this.fReferralRebateCC);
    }
    

  updateFundingAndPricesBefore(revertIfClosed: boolean) {
    return;
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
      this.fMarkPriceEMALambda
    );
  }

  calcInsurancePremium() {
    // prepare data

    // this method is called when rebalancing, which occurs in three cases:
    // 1) trading and liquidations (settlement px is up to date and markets are open)
    // 2) detecting an oracle route is terminated (settlement px is used)
    // 3) margin withdrawals (settlement px is up to date and markets are open)
    // 4) adding and removing liquidity (checkOracleStatus is called without reverting,
    //    so settlement is updated if possible)
    const [ammState, marketState] = this.prepareAMMAndMarketData(false);

    // mid price has no minimal spread
    // mid-price parameter obtained using amount k=0
    let px_premium = calculatePerpetualPrice(ammState, marketState, 0, 0, 0);
    px_premium =
      (px_premium - marketState.fIndexPriceS2) / marketState.fIndexPriceS2;
    return px_premium;
  }
  prepareAMMAndMarketData(_bUseOracle: boolean) {
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
      // emit UpdateMarkPrice(
      //     _perpetual.id,
      //     _fCurrentPremiumRate,
      //     _perpetual.currentMarkPremiumRate.fPrice,
      //     _getSafeOraclePriceS2(_perpetual)
      // );
    }
  }

  updateKStar() {
    const ccy = this.eCollateralCurrency;
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
      const nominator = this.fRho23 * this.fSigma2 * this.fSigma3;
      const denom = this.fSigma2 * this.fSigma2;
      this.fkStar = (nominator / denom / fB2C) * fM - K2;
    }
  }
  updateAMMTargetFundSize() {
    return;
  }

  getRebalanceMargin() {
    const fInitialMargin = this.getInitialMargin(this.address);
    const fMarginBalance = this.getMarginBalance(this.address);
    return [fMarginBalance, fInitialMargin];
  }

  getMarginBalance(address: string) {
    return 0;
  }

  getInitialMargin(address: string) {
    return 0;
  }

  transferFromAMMMarginToPool(fAmount: number) {
    if (fAmount == 0) {
      return;
    }
    require(fAmount > 0, "transferFromAMMMgnToPool >0");
    const pool = this.storage;
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
    // emit UpdateDefaultFundCash(pool.id, fDFAmount, pool.fDefaultFundCashCC);
  }

  increasePoolCash(_liquidityPool: LiquidityPoolData, _fAmount: number) {
    require(_fAmount >= 0, "inc neg pool cash");
    _liquidityPool.fPnLparticipantsCashCC =
      _liquidityPool.fPnLparticipantsCashCC + _fAmount;
    // emit UpdateParticipationFundCash(
    //     _liquidityPool.id,
    //     _fAmount,
    //     _liquidityPool.fPnLparticipantsCashCC
    // );
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
    return 0;
  }
  updateTraderMargin(address: string, arg1: number) {
    throw new Error("Method not implemented.");
  }
  transferFromPoolToAMMMargin(_fAmount: number, _fMarginBalance: number) {
    // transfer from pool to AMM: amount >= 0
    if (_fAmount == 0) {
        return 0;
    }
    require(_fAmount > 0, "transferFromPoolToAMM>0");
    // perpetual state cannot be normal with 0 cash
    const pool = this.storage;
    const fPnLPartFunds = this.getCollateralTokenAmountForPricing(pool);
    require(
        pool.fDefaultFundCashCC > 0 ||
            fPnLPartFunds > 0 ||
            this..state != PerpetualState.NORMAL,
        "state abnormal: 0 DF Cash"
    );
    // we first withdraw from the broker fund
    const fBrokerAmount = this.withdrawFromBrokerPool(_fAmount);
    let fFeasibleMargin = fBrokerAmount;
    if (fBrokerAmount < _fAmount) {
        // now we aim to withdraw _fAmount - fBrokerAmount from the liquidity pools
        // fDFAmount, fLPAmount will give us the amount that can be withdrawn
        const [fDFAmount, fLPAmount] = this.getFeasibleTransferFromPoolToAMMMargin(
            _fAmount-(fBrokerAmount),
            _fMarginBalance,
            fPnLPartFunds,
            pool
        );
        fFeasibleMargin = fFeasibleMargin+(fLPAmount)+(fDFAmount);
    }
    // update margin
    this.updateTraderMargin(this.address, fFeasibleMargin);
    return fFeasibleMargin;
  }

    getFeasibleTransferFromPoolToAMMMargin(_fAmount: number, _fMarginBalance: number, _fPnLPartFunds: number, _pool: LiquidityPoolData) {
        // perp funds coming from the liquidity pool
        const fPoolFunds = this.getPerpetualAllocatedFunds();
        let [fLPAmount, fDFAmount] = [0, 0];
        if (fPoolFunds+(_fMarginBalance) > 0) {
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
            fDFAmount = -(fPoolFunds+_fMarginBalance);
            if (_fPnLPartFunds > _pool.fTargetAMMFundSize) {
                // there are some LP funds allocated to the DF -> split what DF pays
                const fDFWeight = _pool.fDefaultFundCashCC/(
                    _pool.fDefaultFundCashCC.add(_fPnLPartFunds)-(_pool.fTargetAMMFundSize)
                );
                fLPAmount = (fLPAmount+fDFAmount) * (1 - (fDFWeight));
                fDFAmount = fDFAmount*(fDFWeight);
            }
            this.setEmergencyState();
        }
        // ensure DF is not depleted: PnL sharing cap may cause DF to overc-contribute
        //-> PnL participants cover the rest
        if (fDFAmount > _pool.fDefaultFundCashCC) {
            fLPAmount = fLPAmount+(fDFAmount-(_pool.fDefaultFundCashCC));
            fDFAmount = _pool.fDefaultFundCashCC;
        }
        // ensure LPs can cover total: otherwise stop the pool
        if (fLPAmount >= _fPnLPartFunds) {
            // liquidity pool is depleted
            fLPAmount = _fPnLPartFunds;
            this.setLiqPoolEmergencyState(_pool);
        }
        this.decreaseDefaultFundCash(fDFAmount);
        this.decreasePoolCash(fLPAmount);
        // this function returns (fDFAmount, fLPAmount);
    }
    
    
    setEmergencyState() {
        this.state = PerpetualState.EMERGENCY;
    }

  getPerpetualAllocatedFunds() {
    return 0;
  }

  getBaseToCollateralConversionMultiplier(
    isMarkPriceRequest: boolean,
    bUseOracle: boolean
  ) {
    return 1;
  }
}

