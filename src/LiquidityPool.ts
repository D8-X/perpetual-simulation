import { AMMPerpLogic } from "./AMMPerpLogic";
import { Perpetual } from "./Perpetual";
import { LiquidityPoolData } from "./types";
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

  constructor(storage: LiquidityPoolData) {
    this.poolStorage = storage;
    this.ammPerpLogic = new AMMPerpLogic();
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
