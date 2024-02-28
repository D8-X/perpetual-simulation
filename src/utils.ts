export function require(condition: boolean, error: string) {
  if (!condition) {
    throw new Error(error);
  }
}

export function ema(_fEMA: number, _fCurrentObs: number, _fLambda: number) {
  require(_fLambda > 0, "EMALambda must be gt 0");
  require(_fLambda < 1, "EMALambda must be st 1");
  // result must be between the two values _fCurrentObs and _fEMA, so no overflow
  return _fEMA * _fLambda + (1 - _fLambda) * _fCurrentObs;
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

export class AMMPerpLogic {
  getTargetCollateralM3(
    fK: number,
    fLockedIn: number,
    mv: MarketVariables,
    fAMMTargetDD: any
  ): number {
    throw new Error("Method not implemented.");
  }
  getTargetCollateralM1(
    fK: number,
    fLockedIn: number,
    mv: MarketVariables,
    fAMMTargetDD: any
  ): number {
    throw new Error("Method not implemented.");
  }
  getTargetCollateralM2(
    fK: number,
    fLockedIn: number,
    mv: MarketVariables,
    fAMMTargetDD: any
  ): number {
    throw new Error("Method not implemented.");
  }
  calculateRiskNeutralPD(
    _ammVars: AMMVariables,
    _mktVars: MarketVariables,
    _fTradeAmount,
    _withCDF
  ) {
    const dL = _fTradeAmount.mul(_mktVars.fIndexPriceS2);
    const dK = _fTradeAmount;
    _ammVars.fLockedValue1 = _ammVars.fLockedValue1 + dL;
    _ammVars.fAMM_K2 = _ammVars.fAMM_K2 + dK;
    // -L1 - k*s2 - M1
    const fNumerator = -_ammVars.fLockedValue1 - _ammVars.fPoolM1;
    // s2*(M2-k2-K2) if no quanto, else M3 * s3
    let fDenominator =
      _ammVars.fPoolM3 == 0
        ? (_ammVars.fPoolM2 - _ammVars.fAMM_K2) * _mktVars.fIndexPriceS2
        : _ammVars.fPoolM3 * _mktVars.fIndexPriceS3;
    // handle edge sign cases first
    let fThresh = 0;
    if (_ammVars.fPoolM3 == 0) {
      if (fNumerator < 0) {
        if (fDenominator >= 0) {
          // P( den * exp(x) < 0) = 0
          return [0, -20];
        } else {
          // // num < 0 and den < 0, and P(exp(x) > infty) = 0
          // int256 result = (fNumerator) << 64) / fDenominator;
          // if (result > MAX_64x64) {
          //     return (int128(0), TWENTY_64x64.neg());
          // }
          // fThresh = int128(result);
          fThresh = fNumerator / fDenominator;
        }
      } else if (fNumerator > 0) {
        if (fDenominator <= 0) {
          // P( exp(x) >= 0) = 1
          return [1, 20];
        } else {
          // num > 0 and den > 0, and P(exp(x) < infty) = 1
          // int256 result = (int256(fNumerator) << 64) / fDenominator;
          // if (result > MAX_64x64) {
          //     return (int128(ONE_64x64), TWENTY_64x64);
          // }
          // fThresh = int128(result);
          fThresh = fNumerator / fDenominator;
        }
      } else {
        return fDenominator >= 0 ? [0, -20] : [1, 20];
      }
    } else {
      // denom is O(M3 * S3), div should not overflow
      fThresh = fNumerator / fDenominator;
    }
    // if we're here fDenominator !=0 and fThresh did not overflow
    // sign tells us whether we consider norm.cdf(f(threshold)) or 1-norm.cdf(f(threshold))
    // we recycle fDenominator to store the sign since it's no longer used
    fDenominator = fDenominator < 0 ? -1 : 1;
    const dd =
      _ammVars.fPoolM3 == 0
        ? this.calculateRiskNeutralDDNoQuanto(
            _mktVars.fSigma2,
            fDenominator,
            fThresh
          )
        : this.calculateRiskNeutralDDWithQuanto(
            _ammVars,
            _mktVars,
            fDenominator,
            fThresh
          );

    let q;
    if (_withCDF) {
      q = this.normalCDF(dd);
    }
    return [q, dd];
  }
  calculateRiskNeutralDDNoQuanto(
    fSigma2: number,
    fDenominator: number,
    fThresh: number
  ): number {
    throw new Error("Method not implemented.");
  }
  calculateRiskNeutralDDWithQuanto(
    _ammVars: AMMVariables,
    _mktVars: MarketVariables,
    fDenominator: number,
    fThresh: number
  ): number {
    throw new Error("Method not implemented.");
  }

  normalCDF(x: number) {
    const t = 1 / (1 + 0.2315419 * Math.abs(x));
    const d = 0.3989423 * Math.exp((-x * x) / 2);
    let prob =
      d *
      t *
      (0.3193815 +
        t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if (x > 0) prob = 1 - prob;
    return prob;
  }

  calculatePerpetualPrice(
    _ammVars: AMMVariables,
    _mktVars: MarketVariables,
    _fTradeAmount: number,
    _fHBidAskSpread: number,
    _fIncentiveSpread: number
  ) {
    // add minimal spread in quote currency
    _fHBidAskSpread = _fTradeAmount > 0 ? _fHBidAskSpread : -_fHBidAskSpread;
    if (_fTradeAmount == 0) {
      _fHBidAskSpread = 0;
    }
    // get risk-neutral default probability (always >0)
    {
      let [fQ, dd, fkStar] = [0, 0, _ammVars.fPoolM2 - _ammVars.fAMM_K2];
      [fQ, dd] = this.calculateRiskNeutralPD(
        _ammVars,
        _mktVars,
        _fTradeAmount,
        true
      );
      if (_ammVars.fPoolM3 != 0) {
        // amend K* (see whitepaper)
        const nominator =
          _mktVars.fRho23 * (_mktVars.fSigma2 * _mktVars.fSigma3);
        const denom = _mktVars.fSigma2 * _mktVars.fSigma2;
        let h = (nominator / denom) * _ammVars.fPoolM3;
        h = (h * _mktVars.fIndexPriceS3) / _mktVars.fIndexPriceS2;
        fkStar = fkStar + h;
      }
      // decide on sign of premium
      if (_fTradeAmount < fkStar) {
        fQ = -fQ;
      }
      // no rebate if exposure increases
      if (_fTradeAmount > 0 && _ammVars.fAMM_K2 > 0) {
        fQ = fQ > 0 ? fQ : 0;
      } else if (_fTradeAmount < 0 && _ammVars.fAMM_K2 < 0) {
        fQ = fQ < 0 ? fQ : 0;
      }
      // handle discontinuity at zero
      if (
        _fTradeAmount == 0 &&
        ((fQ < 0 && _ammVars.fAMM_K2 > 0) || (fQ > 0 && _ammVars.fAMM_K2 < 0))
      ) {
        fQ = fQ / 2;
      }
      _fHBidAskSpread = _fHBidAskSpread + fQ;
    }
    // get additional slippage
    if (_fTradeAmount != 0) {
      _fIncentiveSpread =
        _fIncentiveSpread *
        this.calculateBoundedSlippage(_ammVars, _fTradeAmount);
      _fHBidAskSpread = _fHBidAskSpread + _fIncentiveSpread;
    }
    // s2*(1 + sign(qp-q)*q + sign(k)*minSpread)
    return _mktVars.fIndexPriceS2 * (1 + _fHBidAskSpread);
  }
  calculateBoundedSlippage(_ammVars: AMMVariables, _fTradeAmount: number) {
    const fTradeSizeEMA = _ammVars.fCurrentTraderExposureEMA;
    let fSlippageSize = 1;
    if (Math.abs(_fTradeAmount) < fTradeSizeEMA) {
      fSlippageSize = fSlippageSize - Math.abs(_fTradeAmount) / fTradeSizeEMA;
      fSlippageSize = 1 - fSlippageSize * fSlippageSize;
    }
    return _fTradeAmount > 0 ? fSlippageSize : -fSlippageSize;
  }
}
