import { AMMVariables, MarketVariables } from "./types";
import { ABDKToTbps, emaWithTimeJumps, tbpsToABDK, require } from "./utils";

export class AMMPerpLogic {
  volatilitySpread(
    _jumpTbps: number,
    _minimalSpreadTbps: number,
    _numSecSinceJump: number,
    _fLambda: number
  ): number {
    // v0 = (1-L**numBlocksSinceJump) * minimalSpread
    // v1 = jump*L**numBlocksSinceJump
    // return v0+v1
    if (_numSecSinceJump > 30) {
      return tbpsToABDK(_minimalSpreadTbps);
    }
    return emaWithTimeJumps(
      _minimalSpreadTbps,
      _jumpTbps,
      _fLambda,
      _numSecSinceJump
    );
  }

  getDepositAmountForLvgPosition(
    _fPosition0: number,
    _fBalance0: number,
    _fTradeAmount: number,
    _fTargetLeverage: number,
    _fPrice: number,
    _fS2Mark: number,
    _fS3: number,
    _fS2: number
  ): number {
    // calculation has to be aligned with _getAvailableMargin and _executeTrade
    // calculation
    // otherwise the calculated deposit might not be enough to declare
    // the margin to be enough
    // aligned with get available margin balance
    let fPremiumCash = _fTradeAmount * (_fPrice - _fS2);
    let fDeltaLockedValue = _fTradeAmount * _fS2;
    let fPnL = _fTradeAmount * _fS2Mark;
    // we replace _fTradeAmount * price/S3 by
    // fDeltaLockedValue + fPremiumCash to be in line with
    // _executeTrade
    fPnL = fPnL - fDeltaLockedValue - fPremiumCash;
    let fLvgFrac = Math.abs(_fPosition0 + _fTradeAmount);
    fLvgFrac = (fLvgFrac * _fS2Mark) / _fTargetLeverage;
    fPnL = (fPnL - fLvgFrac) / _fS3;
    _fBalance0 = _fBalance0 + fPnL;
    return -_fBalance0;
  }

  holdingPeriodPenalty(
    _secondsSinceLastOpen: number,
    _fLambda: number
  ): number {
    if (_secondsSinceLastOpen > 32) {
      return 0;
    }
    return ABDKToTbps(emaWithTimeJumps(0, 80, _fLambda, _secondsSinceLastOpen));
  }

  getTargetCollateralM3(
    _fK2: number,
    _fL1: number,
    _mktVars: MarketVariables,
    _fTargetDD: any
  ): number {
    // we solve the quadratic equation A x^2 + Bx + C = 0
    // B = 2 * [X + Y * target_dd^2 * (exp(rho*sigma2*sigma3) - 1) ]
    // C = X^2  - Y^2 * target_dd^2 * (exp(sigma2^2) - 1)
    // where:
    // X = L1 / S3 - Y and Y = K2 * S2 / S3
    // we re-use L1 for X and K2 for Y to save memory since they don't enter the equations otherwise
    _fK2 = (_fK2 * _mktVars.fIndexPriceS2) / _mktVars.fIndexPriceS3; // Y
    _fL1 = _fL1 / _mktVars.fIndexPriceS3 - _fK2; // X
    // we only need the square of the target DD
    _fTargetDD = _fTargetDD * _fTargetDD;
    // and we only need B/2
    let fHalfB =
      _fL1 +
      _fK2 *
        (_fTargetDD *
          (_mktVars.fRho23 * (_mktVars.fSigma2 * _mktVars.fSigma3)));
    let fC =
      _fL1 * _fL1 -
      _fK2 * _fK2 * _fTargetDD * (_mktVars.fSigma2 * _mktVars.fSigma2);
    // A = 1 - (exp(sigma3^2) - 1) * target_dd^2
    let fA = 1 - _mktVars.fSigma3 * _mktVars.fSigma3 * _fTargetDD;
    // we re-use C to store the discriminant: D = (B/2)^2 - A * C
    fC = fHalfB * fHalfB - fA * fC;
    if (fC < 0) {
      // no solutions -> AMM is in profit, probability is smaller than target regardless of capital
      return 0;
    }
    // we want the larger of (-B/2 + sqrt((B/2)^2-A*C)) / A and (-B/2 - sqrt((B/2)^2-A*C)) / A
    // so it depends on the sign of A, or, equivalently, the sign of sqrt(...)/A
    fC = Math.sqrt(fC) / fA;
    fHalfB = fHalfB / fA;
    return fC > 0 ? fC - fHalfB : -fC - fHalfB;
  }

  getTargetCollateralM1(
    _fK2: number,
    _fL1: number,
    _mktVars: MarketVariables,
    _fTargetDD: any
  ): number {
    let fMu2 = -0.5 * _mktVars.fSigma2 * _mktVars.fSigma2;
    let ddScaled =
      _fK2 < 0 ? _mktVars.fSigma2 * _fTargetDD : -_mktVars.fSigma2 * _fTargetDD;
    let A1 = Math.exp(fMu2 + ddScaled);
    return _fK2 * _mktVars.fIndexPriceS2 * A1 - _fL1;
  }

  getTargetCollateralM2(
    _fK2: number,
    _fL1: number,
    _mktVars: MarketVariables,
    _fTargetDD: any
  ): number {
    let fMu2 = -0.5 * _mktVars.fSigma2 * _mktVars.fSigma2;
    let ddScaled =
      _fL1 < 0 ? _mktVars.fSigma2 * _fTargetDD : -_mktVars.fSigma2 * _fTargetDD;
    let A1 = Math.exp(fMu2 + ddScaled) * _mktVars.fIndexPriceS2;
    return _fK2 - _fL1 / A1;
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
    _fSign: number,
    _fThresh: number
  ): number {
    require(_fThresh > 0, "argument to log must be >0");
    let _fLogTresh = Math.log(_fThresh);
    let fSigma2_2 = fSigma2 * fSigma2;
    let fMean = -fSigma2_2 / 2;
    let fDistanceToDefault = (_fLogTresh - fMean) / fSigma2;
    // because 1-Phi(x) = Phi(-x) we change the sign if _fSign<0
    // now we would like to get the normal cdf of that beast
    if (_fSign < 0) {
      fDistanceToDefault = -fDistanceToDefault;
    }
    return fDistanceToDefault;
  }

  calculateRiskNeutralDDWithQuanto(
    _ammVars: AMMVariables,
    _mktVars: MarketVariables,
    _fSign: number,
    _fThresh: number
  ): number {
    require(_fSign > 0, "no sign in quanto case");
    // 1) Calculate C3
    let fC3 =
      (_mktVars.fIndexPriceS2 * (_ammVars.fPoolM2 - _ammVars.fAMM_K2)) /
      (_ammVars.fPoolM3 * _mktVars.fIndexPriceS3);
    let fC3_2 = fC3 * fC3;

    // 2) Calculate Variance
    let fSigmaZ = this.calculateStandardDeviationQuanto(_mktVars, fC3, fC3_2);

    // 3) Calculate mean
    let fMean = fC3 + 1;
    // 4) Distance to default
    let fDistanceToDefault = (_fThresh - fMean) / fSigmaZ;
    return fDistanceToDefault;
  }

  calculateStandardDeviationQuanto(
    _mktVars: MarketVariables,
    _fC3: number,
    _fC3_2: number
  ) {
    // fVarA = (exp(sigma2^2) - 1)
    let fVarA = _mktVars.fSigma2 * _mktVars.fSigma2;

    // fVarB = 2*(exp(sigma2*sigma3*rho) - 1)
    let fVarB = _mktVars.fSigma2 * _mktVars.fSigma3 * _mktVars.fRho23 * 2;

    // fVarC = exp(sigma3^2) - 1
    let fVarC = _mktVars.fSigma3 * _mktVars.fSigma3;

    // sigmaZ = fVarA*C^2 + fVarB*C + fVarC
    let fSigmaZ = Math.sqrt(fVarA * _fC3_2 + fVarB * _fC3 + fVarC);
    return fSigmaZ;
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
