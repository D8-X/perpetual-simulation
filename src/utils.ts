import { AMMVariables, MarketVariables } from "./types";

const MASK_CLOSE_ONLY = BigInt(0x80000000);
const MASK_MARKET_ORDER = BigInt(0x40000000);
const MASK_STOP_ORDER = BigInt(0x20000000);
const MASK_FILL_OR_KILL = BigInt(0x10000000);
const MASK_KEEP_POS_LEVERAGE = BigInt(0x08000000);
const MASK_LIMIT_ORDER = BigInt(0x04000000);

export function require(condition: boolean, error: string) {
  if (!condition) {
    throw new Error(error);
  }
}

export function roundToLot(amount: number, lotSize: number) {
  return Math.round(amount / lotSize + 1e-15) * lotSize;
}

export function ema(_fEMA: number, _fCurrentObs: number, _fLambda: number) {
  require(_fLambda > 0, "EMALambda must be gt 0");
  require(_fLambda < 1, "EMALambda must be st 1");
  // result must be between the two values _fCurrentObs and _fEMA, so no overflow
  return _fEMA * _fLambda + (1 - _fLambda) * _fCurrentObs;
}

export function emaWithTimeJumps(
  _mean: number,
  _newObs: number,
  _fLambda: number,
  _deltaTime: number
) {
  _fLambda = _fLambda ** _deltaTime;
  let result = tbpsToABDK(_mean) * (1 - _fLambda);
  result = result + _fLambda * tbpsToABDK(_newObs);
  return result;
}

export function isStopOrder(flags: bigint) {
  return (flags & MASK_STOP_ORDER) > 0;
}

export function isCloseOnly(flags: bigint) {
  return (flags & MASK_CLOSE_ONLY) > 0;
}

export function isMarketOrder(flags: bigint) {
  return (flags & MASK_MARKET_ORDER) > 0;
}

export function keepPositionLeverageOnClose(flags: bigint) {
  return flags & MASK_KEEP_POS_LEVERAGE;
}

export function validateStopPrice(
  _isLong: boolean,
  _fMarkPrice: number,
  _fTriggerPrice: number
) {
  {
    if (_fTriggerPrice == 0) {
      return;
    }
    // if stop order, mark price must meet trigger price condition
    const isTriggerSatisfied = _isLong
      ? _fMarkPrice >= _fTriggerPrice
      : _fMarkPrice <= _fTriggerPrice;
    require(isTriggerSatisfied, "trigger cond not met");
  }
}

export function hasTheSameSign(_fX: number, _fY: number) {
  if (_fX == 0 || _fY == 0) {
    return true;
  }
  return (_fX ^ _fY) >> 127 == 0;
}

export function hasOpenedPosition(_fNewPos: number, fDeltaPos: number) {
  if (_fNewPos == 0) {
    return false;
  }
  return hasTheSameSign(_fNewPos, fDeltaPos);
}

export function tbpsToABDK(x: number): number {
  return x / 1e5;
}

export function ABDKToTbps(x: number): number {
  return Math.floor(x * 1e5);
}

export function tdrToABDK(x: number): number {
  return x / 1000;
}

export function shrinkToMaxPositionToClose(
  _fPosition: number,
  _fAmount: number
) {
  require(_fPosition != 0, "trader has no position to close");
  require(!hasTheSameSign(_fPosition, _fAmount), "trade is close only");
  return Math.abs(_fAmount) > Math.abs(_fPosition)
    ? Math.abs(_fPosition)
    : _fAmount;
}

export function validatePrice(
  _isLong: boolean,
  _fPrice: number,
  _fPriceLimit: number
) {
  require(_fPrice > 0, "price must be positive");
  return _isLong ? _fPrice <= _fPriceLimit : _fPrice >= _fPriceLimit;
}
