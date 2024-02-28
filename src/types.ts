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
  // this is simulation-specific (not on-chain)
  fTradingFee: number;
  fBrokerFee: number;
}

export interface LiquidityPoolData {
  isRunning; // state
  iPerpetualCount: number; // state
  id; // parameter: index, starts from 1
  fCeilPnLShare: number; // parameter: cap on the share of PnL allocated to liquidity providers
  marginTokenDecimals: number; // parameter: decimals of margin token, inferred from token contract
  iTargetPoolSizeUpdateTime: number; //parameter: timestamp in seconds. How often we update the pool's target size
  marginTokenAddress; //parameter: address of the margin token
  // -----
  prevAnchor: number; // state: keep track of timestamp since last withdrawal was initiated
  fRedemptionRate: number; // state: used for settlement in case of AMM default
  shareTokenAddress; // parameter
  // -----
  fPnLparticipantsCashCC: number; // state: addLiquidity/withdrawLiquidity + profit/loss - rebalance
  fTargetAMMFundSize: number; // state: target liquidity for all perpetuals in pool (sum)
  // -----
  fDefaultFundCashCC: number; // state: profit/loss
  fTargetDFSize: number; // state: target default fund size for all perpetuals in pool
  // -----
  fBrokerCollateralLotSize: number; // param:how much collateral do brokers deposit when providing "1 lot" (not trading lot)
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

export interface IPerpetualOrder {
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
