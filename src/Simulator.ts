import { AMMPerpLogic } from "./AMMPerpLogic";
import { LiquidityPool } from "./LiquidityPool";
import { OrderBook } from "./OrderBook";
import {
  Block,
  LiquidityPoolData,
  LiquidityPoolParams,
  PerpetualParams,
  PerpetualState,
} from "./types";
import { loadData, require } from "./utils";

export class Simulator {
  poolCount: number = 0;
  pools: Map<number, LiquidityPool> = new Map();
  orderBooks: Map<number, OrderBook[]> = new Map();
  ammPerpLogic: AMMPerpLogic;
  block: Block;
  data: unknown;

  constructor() {
    // load "time" data?
    this.data = loadData();
    this.ammPerpLogic = new AMMPerpLogic();
    this.block = {
      number: 0,
      timestamp: 0,
    };
  }

  executeOrders(poolId: number) {
    const orderBooks = this.orderBooks.get(poolId)!;
    for (const orderBook of orderBooks) {
      const orderIds = Object.keys(orderBook.orders);
      for (const orderId of orderIds) {
        orderBook.executeOrder(+orderId);
      }
    }
  }

  createLiquidityPool(poolParams: LiquidityPoolParams) {
    this.poolCount++;
    const poolData: LiquidityPoolData = {
      isRunning: false,
      iPerpetualCount: 0,
      id: this.poolCount,
      prevAnchor: 0,
      fRedemptionRate: 0,
      fPnLparticipantsCashCC: 0,
      fTargetAMMFundSize: 0,
      fDefaultFundCashCC: 0,
      fTargetDFSize: 0,
      prevTokenAmount: 0,
      nextTokenAmount: 0,
      totalSupplyShareToken: 0,
      fBrokerFundCashCC: 0,
      ...poolParams,
    };
    this.pools.set(
      poolData.id,
      new LiquidityPool(this.block, this.ammPerpLogic, poolData)
    );
  }

  createPerpetual(poolId: number, perpParams: PerpetualParams) {
    const pool = this.pools.get(poolId);
    require(!!pool, "check pool id");
    const perpetual = pool!.createPerpetual(perpParams);
    if (this.orderBooks.has(poolId)) {
      this.orderBooks.get(poolId)?.push(new OrderBook(perpetual));
    } else {
      this.orderBooks.set(poolId, [new OrderBook(perpetual)]);
    }
  }

  runLiquidityPool(poolId: number) {
    require(poolId > 0 && poolId <= this.poolCount, "pool index out of range");
    const liquidityPool = this.pools.get(poolId)!;
    require(!liquidityPool.poolStorage.isRunning, "pool already running");
    const length = liquidityPool.poolStorage.iPerpetualCount;
    require(length > 0, "need perpetuals to run");
    let fMinTargetSum = 0;
    for (const perpetual of liquidityPool.perpetuals) {
      let fMinTarget = perpetual.params.fAMMMinSizeCC;
      if (perpetual.state == PerpetualState.INITIALIZING) {
        perpetual.fTargetAMMFundSize = fMinTarget;
        perpetual.fTargetDFSize = fMinTarget;
        fMinTargetSum = fMinTargetSum + fMinTarget;
        perpetual.setNormalState();
      }
    }
    liquidityPool.poolStorage.fTargetDFSize = fMinTargetSum;
    liquidityPool.poolStorage.fTargetAMMFundSize = fMinTargetSum;
    liquidityPool.poolStorage.isRunning = true;
  }
}
