import { require } from "./utils";

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

export class LiquidityPool {
  storage: LiquidityPoolData;
  block: Block;
  constructor(storage: LiquidityPoolData) {
    this.storage = storage;
  }

  withdrawFromBrokerPool(_fAmount: number) {
    // pre-condition: require(_fAmount > 0, "withdraw amount must>0");
    const fBrokerPoolCC = this.storage.fBrokerFundCashCC;
    if (fBrokerPoolCC == 0) {
      return 0;
    }
    const withdraw = _fAmount > fBrokerPoolCC ? fBrokerPoolCC : _fAmount;
    this.storage.fBrokerFundCashCC = fBrokerPoolCC - withdraw;
    return withdraw;
  }

  decreaseDefaultFundCash(_fAmount: number) {
    require(_fAmount >= 0, "dec neg pool cash");
    this.storage.fDefaultFundCashCC =
      this.storage.fDefaultFundCashCC - _fAmount;
    require(this.storage.fDefaultFundCashCC >= 0, "DF cash cannot be <0");
  }

  decreasePoolCash(_fAmount: number) {
    require(_fAmount >= 0, "dec neg pool cash");
    this.storage.fPnLparticipantsCashCC =
      this.storage.fPnLparticipantsCashCC - _fAmount;
  }

  transferFromVaultToUser(executorAddr: any, fReferralRebateCC: number) {
    throw new Error("Method not implemented.");
  }
  transferFromUserToVault(traderAddr: any, fReferralRebateCC: number) {
    throw new Error("Method not implemented.");
  }
}
