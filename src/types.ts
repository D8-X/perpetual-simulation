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

export interface PerpetualParams {
  S3QuoteCCY: string;
  S3BaseCCY: string;
  S2QuoteCCY: string;
  S2BaseCCY: string;
  fDFCoverNRate: number;
  minimalSpreadTbps: number;
  incentiveSpreadTbps: number;
  fMaximalTradeSizeBumpUp: number;
  fMinimalTraderExposureEMA: number;
  fMinimalAMMExposureEMA: number;
  fInitialMarginRate: number;
  fMaintenanceMarginRate: number;
  fLotSizeBC: number;
  fSigma2: number;
  fSigma3: number;
  fRho23: number;
  fMarkPriceEMALambda: number;
  fReferralRebateCC: number;
  fAMMMinSizeCC: number;
  fAMMTargetDD: number;
  fDFLambda: [number, number];
  eCollateralCurrency: CollateralCurrency;
  fFundingRateClamp: number;
  fStressReturnS2: [number, number];
  fStressReturnS3: [number, number];
  // this is simulation-specific (not on-chain)
  fTradingFee: number;
  fBrokerFee: number;
}

export interface LiquidityPoolParams {
  iTargetPoolSizeUpdateTime: number;
  fBrokerCollateralLotSize: number;
  shareTokenAddress: string;
  marginTokenAddress: string;
  marginTokenDecimals: number;
  fCeilPnLShare: number;
}

export interface LiquidityPoolData extends LiquidityPoolParams {
  isRunning: boolean; // state
  iPerpetualCount: number; // state
  id: number; // parameter: index, starts from 1
  // fCeilPnLShare: number; // parameter: cap on the share of PnL allocated to liquidity providers
  // marginTokenDecimals: number; // parameter: decimals of margin token, inferred from token contract
  // iTargetPoolSizeUpdateTime: number; //parameter: timestamp in seconds. How often we update the pool's target size
  // marginTokenAddress; //parameter: address of the margin token
  // -----
  prevAnchor: number; // state: keep track of timestamp since last withdrawal was initiated
  fRedemptionRate: number; // state: used for settlement in case of AMM default
  // shareTokenAddress; // parameter
  // -----
  fPnLparticipantsCashCC: number; // state: addLiquidity/withdrawLiquidity + profit/loss - rebalance
  fTargetAMMFundSize: number; // state: target liquidity for all perpetuals in pool (sum)
  // -----
  fDefaultFundCashCC: number; // state: profit/loss
  fTargetDFSize: number; // state: target default fund size for all perpetuals in pool
  // -----
  // fBrokerCollateralLotSize: number; // param:how much collateral do brokers deposit when providing "1 lot" (not trading lot)
  prevTokenAmount: number; // state
  // -----
  nextTokenAmount: number; // state
  totalSupplyShareToken: number; // state
  // -----
  fBrokerFundCashCC: number; // state: amount of cash in broker fund
}

export interface Block {
  number: number;
  timestamp: number;
}

export interface MarginAccount {
  fLockedInValueQC: number;
  fCashCC: number;
  fPositionBC: number;
  fUnitAccumulatedFundingStart: number;
  feeTbps: number;
  brokerFeeTbps: number;
  iLastOpenTimestamp: number;
}

export interface PerpetualOrder {
  leverageTDR: number; // 12.43x leverage is represented by 1243 (two-digit integer representation); 0 if deposit and trade separate
  brokerFeeTbps: number; // broker can set their own fee
  // iPerpetualId: number; // global id for perpetual
  traderAddr: string; // address of trader
  executionTimestamp: number; // normally set to current timestamp; order will not be executed prior to this timestamp.
  brokerAddr: string; // address of the broker or zero
  submittedTimestamp: number;
  flags: bigint; // order flags
  iDeadline: number; //deadline for price (seconds timestamp)
  executorAddr: string; // address of the executor set by contract
  fAmount: number; // amount in base currency to be traded
  fLimitPrice: number; // limit price
  fTriggerPrice: number; //trigger price. Non-zero for stop orders.
  brokerSignature: string; //signature of broker (or 0)
}

export interface ClientOrder {
  // uint24 iPerpetualId; // unique id of the perpetual
  fLimitPrice: number; // order will not execute if realized price is above (buy) or below (sell) this price
  leverageTDR: number; // leverage, set to 0 if deposit margin and trade separate; format: two-digit integer (e.g., 12.34 -> 1234)
  executionTimestamp: number; // the order will not be executed before this timestamp, allows TWAP orders
  flags: bigint; // Order-flags are specified in OrderFlags.sol
  iDeadline: number; // order will not be executed after this deadline
  brokerAddr: string; // can be empty, address of the broker
  fTriggerPrice: number; // trigger price for stop-orders|0. Order can be executed if the mark price is below this price (sell order) or above (buy)
  fAmount: number; // signed amount of base-currency. Will be rounded to lot size
  // parentChildDigest1: string; // see notice in LimitOrderBook.sol
  traderAddr: string; // address of the trader
  // parentChildDigest2: string; // see notice in LimitOrderBook.sol
  brokerFeeTbps: number; // broker fee in tenth of a basis point
  brokerSignature: string; // signature, can be empty if no brokerAddr provided
  // address callbackTarget; // address of contract implementing callback function
  //address executorAddr; <- will be set by LimitOrderBook
  //uint64 submittedBlock <- will be set by LimitOrderBook
}

export interface AMMVariables {
  // all variables are
  // signed 64.64-bit fixed point number
  fLockedValue1: number; // L1 in quote currency
  fPoolM1: number; // M1 in quote currency
  fPoolM2: number; // M2 in base currency
  fPoolM3: number; // M3 in quanto currency
  fAMM_K2: number; // AMM exposure (positive if trader long)
  fCurrentTraderExposureEMA: number; // current average unsigned trader exposure
}

export interface MarketVariables {
  fIndexPriceS2: number; // base index
  fIndexPriceS3: number; // quanto index
  fSigma2: number; // standard dev of base currency
  fSigma3: number; // standard dev of quanto currency
  fRho23: number; // correlation base/quanto currency
}
