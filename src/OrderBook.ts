import { Perpetual } from "./Perpetual";
import { ClientOrder, PerpetualOrder } from "./types";
import { isStopOrder, require } from "./utils";

export class OrderBook {
  orderCount: number = 0;
  orders: Map<number, PerpetualOrder> = new Map();
  activeDigests: Set<string> = new Set();
  perpetual: Perpetual;

  constructor(perpetual: Perpetual) {
    this.perpetual = perpetual;
  }

  postOrder(_order: ClientOrder) {
    require(_order.traderAddr != "", "invalid-trader");
    require(_order.fAmount != 0, "invalid amount");
    require(_order.iDeadline >
      this.perpetual.block.timestamp, "invalid-deadline");
    // executionTimestamp prior to now+7 days
    require(_order.executionTimestamp < _order.iDeadline &&
      _order.executionTimestamp <
        this.perpetual.block.timestamp + 604800, "invalid exec ts");
    if (isStopOrder(_order.flags)) {
      require(_order.fTriggerPrice > 0, "invalid trigger price");
    }
    // copy client-order into a more lean perp-order
    const perpOrder: PerpetualOrder = this.clientOrderToPerpOrder(_order);
    // set order submission time
    perpOrder.submittedTimestamp = this.perpetual.block.timestamp;

    // _checkBrokerSignature(perpOrder);
    const digest = this.getDigest(perpOrder);
    require(!this.activeDigests.has(digest), "order already exists");
    const orderId = this.orderCount++;
    this.orders.set(orderId, perpOrder);
    return orderId;
  }

  getDigest(_order: PerpetualOrder) {
    return [
      _order.brokerFeeTbps, // trader needs to sign for the broker fee
      _order.traderAddr,
      _order.brokerAddr, // trader needs to sign for broker
      _order.fAmount,
      _order.fLimitPrice,
      _order.fTriggerPrice,
      _order.iDeadline,
      _order.flags,
      _order.leverageTDR,
      _order.submittedTimestamp,
    ]
      .map((x) => x.toString())
      .join("#");
  }

  executeOrder(orderId: number) {
    const order = this.orders.get(orderId);
    require(!!order, "order does not exist");
    const digest = this.getDigest(order!);
    order!.executorAddr = "0xexecutor";
    this.perpetual.tradeViaOrderBook(order!, true);
    this.activeDigests.delete(digest);
  }

  clientOrderToPerpOrder(_order: ClientOrder): PerpetualOrder {
    return {
      flags: _order.flags,
      // iPerpetualId : _order.iPerpetualId;
      brokerFeeTbps: _order.brokerFeeTbps,
      traderAddr: _order.traderAddr,
      brokerAddr: _order.brokerAddr,
      brokerSignature: _order.brokerSignature,
      fAmount: _order.fAmount,
      fLimitPrice: _order.fLimitPrice,
      fTriggerPrice: _order.fTriggerPrice,
      leverageTDR: _order.leverageTDR,
      iDeadline: _order.iDeadline,
      executionTimestamp: _order.executionTimestamp,
      submittedTimestamp: 0,
      executorAddr: "",
    };
  }
}
