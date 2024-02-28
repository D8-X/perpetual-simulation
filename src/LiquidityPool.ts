import { AMMPerpLogic } from "./AMMPerpLogic";
import { Perpetual } from "./Perpetual";
import { LiquidityPoolData, PerpetualParams, PerpetualState } from "./types";
import { require } from "./utils";

export interface Block {
  number: number;
  timestamp: number;
}

export class LiquidityPool {
  poolStorage: LiquidityPoolData;
  block: Block;
  perpetuals: Perpetual[];
  ammPerpLogic: AMMPerpLogic;

  constructor(
    block: Block,
    ammPerpLogic: AMMPerpLogic,
    storage: LiquidityPoolData
  ) {
    this.poolStorage = storage;
    this.ammPerpLogic = ammPerpLogic;
    this.block = block;
  }

  createPerpetual(perpParams: PerpetualParams) {
    this.poolStorage.iPerpetualCount++;
    const perpetual = new Perpetual(
      this.block,
      this.ammPerpLogic,
      this.poolStorage,
      perpParams
    );
    this.perpetuals.push(perpetual);
    return perpetual;
  }

  activatePerpetual(perpIndex: number) {
    require(perpIndex >= 0 &&
      perpIndex < this.poolStorage.iPerpetualCount, "perp index out of bounds");
    const perpetual = this.perpetuals[perpIndex];
    require(perpetual.state ==
      PerpetualState.INVALID, "state should be INVALID");
    // if pool already running: invalid -> normal and we need to add pool cash
    if (this.poolStorage.isRunning) {
      // set starting value for AMM and DF target Pool sizes
      const fMinTarget = perpetual.params.fAMMMinSizeCC;
      perpetual.fTargetAMMFundSize = fMinTarget;
      perpetual.fTargetDFSize = fMinTarget;
      this.poolStorage.fTargetDFSize =
        this.poolStorage.fTargetDFSize + fMinTarget;
      this.poolStorage.fTargetAMMFundSize =
        this.poolStorage.fTargetAMMFundSize + fMinTarget;
      // ready to set normal state
      perpetual.setNormalState();
    } else {
      perpetual.state = PerpetualState.INITIALIZING;
    }
  }

  getOraclePrice(baseQuote: [string, string]): number {
    throw new Error("Method not implemented.");
  }

  setLiqPoolEmergencyState() {
    for (const perpetual of this.perpetuals) {
      perpetual.setEmergencyState();
    }
  }

  getShareTokenAmountForPricing(): number {
    throw new Error("Method not implemented.");
  }

  withdrawFromBrokerPool(_fAmount: number) {
    // pre-condition: require(_fAmount > 0, "withdraw amount must>0");
    const fBrokerPoolCC = this.poolStorage.fBrokerFundCashCC;
    if (fBrokerPoolCC == 0) {
      return 0;
    }
    const withdraw = _fAmount > fBrokerPoolCC ? fBrokerPoolCC : _fAmount;
    this.poolStorage.fBrokerFundCashCC = fBrokerPoolCC - withdraw;
    return withdraw;
  }

  decreaseDefaultFundCash(_fAmount: number) {
    require(_fAmount >= 0, "dec neg pool cash");
    this.poolStorage.fDefaultFundCashCC =
      this.poolStorage.fDefaultFundCashCC - _fAmount;
    require(this.poolStorage.fDefaultFundCashCC >= 0, "DF cash cannot be <0");
  }

  decreasePoolCash(_fAmount: number) {
    require(_fAmount >= 0, "dec neg pool cash");
    this.poolStorage.fPnLparticipantsCashCC =
      this.poolStorage.fPnLparticipantsCashCC - _fAmount;
  }

  transferFromVaultToUser(executorAddr: any, fReferralRebateCC: number) {
    throw new Error("Method not implemented.");
  }
  transferFromUserToVault(traderAddr: any, fReferralRebateCC: number) {
    throw new Error("Method not implemented.");
  }
}
