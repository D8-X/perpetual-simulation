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
  fRho23: number;
  fSigma2: number;
  fSigma3: number;
  fMarkPriceEMALambda: number;
  fReferralRebateCC: number;
  fInitialMarginRate: number;
  fAMMMinSizeCC: number;
  fAMMTargetDD: number;
  fDFLambda: [number, number];
  eCollateralCurrency: CollateralCurrency;
}

export interface LiquidityPoolData {
  isRunning; // state
  iPerpetualCount; // state
  id; // parameter: index, starts from 1
  fCeilPnLShare; // parameter: cap on the share of PnL allocated to liquidity providers
  marginTokenDecimals; // parameter: decimals of margin token, inferred from token contract
  iTargetPoolSizeUpdateTime; //parameter: timestamp in seconds. How often we update the pool's target size
  marginTokenAddress; //parameter: address of the margin token
  // -----
  prevAnchor; // state: keep track of timestamp since last withdrawal was initiated
  fRedemptionRate; // state: used for settlement in case of AMM default
  shareTokenAddress; // parameter
  // -----
  fPnLparticipantsCashCC; // state: addLiquidity/withdrawLiquidity + profit/loss - rebalance
  fTargetAMMFundSize; // state: target liquidity for all perpetuals in pool (sum)
  // -----
  fDefaultFundCashCC; // state: profit/loss
  fTargetDFSize; // state: target default fund size for all perpetuals in pool
  // -----
  fBrokerCollateralLotSize; // param:how much collateral do brokers deposit when providing "1 lot" (not trading lot)
  prevTokenAmount; // state
  // -----
  nextTokenAmount; // state
  totalSupplyShareToken; // state
  // -----
  fBrokerFundCashCC; // state: amount of cash in broker fund
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
