#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plot

from trader import CollateralCurrency

class SingularPricingError(Exception):
    """Raised then the pricing formulas become singular (e.g. Q = 1 for some trading amounts)"""

def get_variance_Z_withC(r, sig2, sig3, rho, C3):
    return np.exp(2*r)*(
        (np.exp(sig3**2)-1)*C3**2 + (np.exp(sig2**2)-1) +
        2*(np.exp(sig2*sig3*rho)-1)*C3
    )

def get_variance_Z(r, sig2, sig3, rho, M3, s2, s3, M2, K2):
    C3=M3*s3/(M2*s2-K2*s2)
    return np.exp(2*r)*(
        (np.exp(sig3**2)-1)*C3**2 + (np.exp(sig2**2)-1) +
        2*(np.exp(sig2*sig3*rho)-1)*C3
    )

def prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    assert(M3 > 0)
    # Cases: 
    # a) M2 - K2 >= 0
    #    i) L1 + M1 >= 0 --> 0
    #   ii) L1 + M1 < 0 --> log-normal
    # b) M2 - K2 < 0 --> always normal 
    #    i) 0 < s2*(K2 - M2) < s3 * M3 --> normal as is
    #   ii) 0 < s3*M3 < s2*(K2 - M2) --> normal but different?

    # a) i)
    if L1 + M1 >= 0 and M2 - K2 >= 0:
        return 0, -100
    
    A = s2 * (M2 - K2)
    B = s3 * M3
    er = np.exp(r)
    esig_2 = np.exp(sig2 * sig2) - 1
    esig_3 = np.exp(sig3 * sig3) - 1
    erho_23 = np.exp(rho * sig2 * sig3) - 1
    
    # a) ii) A >= 0, B >= 0 and max(A, B) > 0
    if False and A >= 0: # kill the log-normal approx for now
        esig_sq = (A * A * esig_2 + B * B * esig_3 + 2 * A * B * erho_23) / (A + B) / (A + B)
        sig_Z = np.log(1 + esig_sq)
        mu_Z = r + np.log(A + B) - 0.5 * sig_Z 
        sig_Z = np.sqrt(sig_Z)
        assert(L1 + M1 < 0)
        dd = (np.log(-L1 - M1) - mu_Z) / sig_Z
    # b) i)  B >= |A| > 0
    elif True or A + B >= 0:
        # assert(A < 0)
        mu_Z = er * (1 + A / B)
        sig_Z = er * np.sqrt(esig_3 + (A / B) * (A / B) * esig_2 + 2 * A / B * erho_23)
        dd = -((L1 + M1) / B + mu_Z) / sig_Z
    # b) ii) |A| > B > 0
    else:
        assert(A < 0)
        mu_Z = er * (1 + B / A)
        sig_Z = er * np.sqrt(esig_2 + (B / A) * (B / A) * esig_3 + 2 * B / A * erho_23)
        dd = ((L1 + M1) / A + mu_Z) / sig_Z
    qobs = norm.cdf(dd)
    return qobs, dd

def prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    assert(M3==0)# "no quanto allowed"
    if M2-K2>=0 and -L1-M1<=0:
        return 0, -100
    if M2-K2<=0 and -L1-M1>0:
        return 1, 100
    
    sigY = sig2
    muY = r-0.5*sig2**2
    denom = s2*(M2-K2)
    Qplus_score = np.log((-L1-M1)/denom) - muY
    dd = Qplus_score/sigY
    
    if M2-K2<0:
        dd = -dd
    Qplus = norm.cdf(dd)
    return Qplus, dd

def get_Kstar(s2, s3, M1, M2, M3, K2, L1, sig2, sig3, rho23, r):
    assert(r == 0) # don't have a formula for r!=0
    # option 1 (from quanto by matching moments)
    k_star = M2 - K2 + (s3 / s2) * (np.exp(rho23 * sig2 * sig3) - 1) / (np.exp(sig2 * sig2) - 1) * M3
    # option 2 (from PD def by finding the zero interval, even if end-points are flipped!)
    #k_star = 0.5 * (-(L1 + M1) / s2 + M2 - K2)
    return k_star
    
def penalization_function(k_ratio):
    if k_ratio < -1:
        return -1
    if k_ratio < 0:
        return (1 - np.abs(k_ratio))**2 - 1
    if k_ratio < 1:
        return 1 - (1 - np.abs(k_ratio))**2
    return 1

def calculate_insurance(M1, M2, M3, K2, L1, s20, s30, r, sig2, sig3, rho23):
    assert(M3 == 0)
    
    if K2 <= M2 and M1 + L1 >= 0:
        return 0
    elif (K2 >= M2) and (M1 + L1 <= 0):
        return (K2 - M2) * np.exp(r) * s20 - (M1 + L1)
    
    Theta = np.log((M1 + L1) / ((K2 - M2) * s20))
    d1 = (Theta - r + 0.5 * sig2**2) / sig2
    d2 = (Theta - r - 0.5 * sig2**2) / sig2

    if (K2 > M2) and (M1 + L1 > 0):
        return (K2 - M2) * np.exp(r) * s20 * (1 - norm.cdf(d2)) - (M1 + L1) * (1 - norm.cdf(d1))
    else:
        return (K2 - M2) * np.exp(r) * s20 * norm.cdf(d2) - (M1 + L1) * norm.cdf(d1)


def calculate_perp_priceV4(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread=0.0001, incentiveSpread=0.0005, k_bar=1):
    # current champion method
    dL = k*s2
    k_star = get_Kstar(s2, s3, M1, M2, M3, K2, L1, sig2, sig3, rho, r)
    if M3==0:
        q, dd = prob_def_no_quanto(K2+k, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    else:
        q, dd = prob_def_quanto(K2+k, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    incentive = incentiveSpread * penalization_function(k / k_bar)
    prem = np.sign(k - k_star) * q
    if K2 > 0 and k > 0:
        prem = np.max((prem, 0))
    elif K2 < 0 and k < 0:
        prem = np.min((prem, 0))
    px = s2 * (1 + prem + np.sign(k) * minSpread + incentive)
    return px


def calculate_perp_priceV5(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread=0.0001, incentiveSpread=0.0005, k_bar=1):
    if M3 > 0:
        return calculate_perp_priceV4(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incentiveSpread, k_bar)
    I0 = calculate_insurance(M1, M2, M3, K2, L1, s2, s3, r, sig2, sig3, rho)
    Ik = calculate_insurance(M1, M2, M3, K2 + k, L1 + k * s2, s2, s3, r, sig2, sig3, rho)
    prem = 0 if I0 == Ik else (Ik - I0) / (k * s2)
    incentive = incentiveSpread * penalization_function(k / k_bar)
    px = s2 * (1 + prem + np.sign(k) * minSpread + incentive)
    return px

def calculate_perp_priceV3(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread):
    # Calculate perp-price based on difference in Q - OBSOLETE
    # Problematic for large Q. Dynamic difficult to handle/understand.

    # handle discontinuity
    k_approx = k
    if np.abs(k)<0.00001:
        # keep k flat for unrealistically small quantities
        k_approx = 0.00001 if k>0 else -0.00001
    
    dL = k_approx*s2
    if M3==0:
        q0, dd = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        q, dd = prob_def_no_quanto(K2+k_approx, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)       
    else:
        assert(0)

    v = s2*(q-q0)/k_approx
    minSpread = minSpread
    #incentive = (np.abs(kStar - k)-np.abs(kStar))/k*incentive_spread
    return (s2 +  v) + minSpread*k

def calculate_perp_priceV2(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread):

    L = L1+k*s2
    if M3==0:
        q, dd = prob_def_no_quanto(K2+k, L, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q, dd = prob_def_quanto(K2+k, L, s2, s3, sig2, sig3, rho, r, M1, M2, M3)

    u = -L1/s2 - M1/s2
    v = K2 - M2
    kStar = (u-v)/2
    sgnm = np.sign(k - kStar)
    minSpread = 0
    return s2*(1 + sgnm*q*(1-minSpread) + np.sign(k)*minSpread)

def calculate_perp_price(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, dir, isApprox):
    """[summary]

    Args:
        K2 ([type]): [description]
        L1 ([type]): [description]
        s2 ([type]): [description]
        s3 ([type]): [description]
        sig2 ([type]): [description]
        sig3 ([type]): [description]
        rho ([type]): [description]
        r ([type]): [description]
        M1 ([type]): [description]
        M2 ([type]): [description]
        M3 ([type]): [description]
        minSpread ([type]): [description]
        dir ([type]): trade direction -1 short, 1 long, 0 for mid-price
        isApprox (bool): [description]

    Returns:
        [type]: [description]
    """
    if M3==0:
        q, dd = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q, dd = prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)

    if isApprox:
        # use approximated cdf
        q = bad_cdf_approximation(dd)

    return s2*(1+np.sign(K2-M2)*q)+dir*minSpread
    #return s2*(1+np.sign(K2)*q)+dir*minSpread

def calculate_McB_price(K2, k2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    dL = k2*s2
    if M3==0:
        q, _ = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_no_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q, _ = prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    alpha = s2
    return s2+alpha*np.sign(k2+K2)*(qplus-q)

def bad_cdf_approximation(dd):
    # this function provides an approximation for
    # the normal cdf
    # https://mathoverflow.net/questions/19404/approximation-of-a-normal-distribution-function
    r = 1/(1+np.exp(-1.65451*dd))
    return r

def get_target_collateral_M1(_fK2, _fS2, _fL1, _fSigma2, _fTargetDD):
    fMu2 = -0.5*_fSigma2**2
    if _fK2<0:
        fMstar = _fK2 * _fS2 * np.exp(fMu2 + _fSigma2*_fTargetDD) - _fL1
    else:
        fMstar = _fK2 * _fS2 * np.exp(fMu2 - _fSigma2*_fTargetDD) - _fL1
    
    # check
    #pd, _ = prob_def_no_quanto(_fK2, _fL1, _fS2, 0, _fSigma2, 0, 0, 0, fMstar, 0, 0)
    #print("pd=", pd)
    #print("dd=", norm.ppf(pd), ", target was dd=", _fTargetDD)
    return fMstar

def get_target_collateral_M2(_fK2, _fS2, _fL1, _fSigma2, _fTargetDD):
    fMu2 = -0.5*_fSigma2**2
    if _fL1<0:
        fMstar = _fK2  - _fL1/np.exp(fMu2 + _fSigma2*_fTargetDD)/_fS2
    else:
        fMstar = _fK2  - _fL1/np.exp(fMu2 - _fSigma2*_fTargetDD)/_fS2
    assert(fMstar >= _fK2 - _fL1/_fS2)
    #fMstar = np.max((fMstar, _fK2 - _fL1/_fS2))
    # check
    #pd, _ = prob_def_no_quanto(_fK2, _fL1, _fS2, 0, _fSigma2, 0, 0, 0, 0, fMstar, 0)
    #print("pd=", pd)
    #print("dd=", norm.ppf(pd), ", target was dd=", _fTargetDD)
    return fMstar

def get_target_collateral_M3(K2, s2, s3, L1, sig2, sig3, rho, r, _fTargetDD):
    # calculate AMM fund size for target default probability q
    # returns both solutions of the quadratic equation, the 
    # max of the two is the correct one
    # Phi(dd) = q
    # Phi^-1(1 - q) = -dd
    M1, M2 = 0, 0

    if M2 - K2 >= 0 and L1 + M1 >= 0:
        return 0
    
    tau = _fTargetDD ** 2
    a = 1 - (np.exp(sig3 ** 2) - 1) * tau
    b0 = np.exp(-r) * (L1 + M1) / s3 + (s2 / s3) * (M2 - K2)
    c0 = (s2 / s3) * (M2 - K2)
    b = 2 * (b0 - c0 * (np.exp(rho * sig2 * sig3) - 1) * tau)
    c = b0 ** 2 - c0 ** 2 * (np.exp(sig2 ** 2) - 1) * tau
    delta = b * b - 4 * a * c
    if delta < 0:
        print("Quadratic for M3 has no real solutions")
        return 0
    root_delta = np.sqrt(delta)
    Mstar = np.max(((-b + root_delta) / (2 * a), (-b - root_delta) / (2 * a)))
    if Mstar < 0:
        Mstar = 1e-12
    #     print("Quadratic for M3 has only negative roots")
    # check
    # params: K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3
    pd, _ = prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, Mstar)
    if np.floor(10000 * pd) > np.ceil(10000 * norm.cdf(_fTargetDD)):
        print("DANGER!!!!!")
        print("------------")
        print(f"pd={100*pd:.2f}% (dd={norm.ppf(pd):.2f}), target was pd={100*norm.cdf(_fTargetDD):.2f}% (dd={_fTargetDD:.2f})")
        print(f"M3*={Mstar:.2f}, L1={L1:.2f}, K2={K2:.2f}, s2={s2:.2f}, s3={s3:.2f}, M3* + (L1-s2*K2)/s3={Mstar + (L1-s2*K2)/s3:.2f}")
        if M2 >= K2 and L1 + M1 < 0:
            print("LOGNORMAL REGION")
    return Mstar

def get_target_collateral_M3_fromPD(q, K2, s2, s3, L1, sig2, sig3, rho, r):
    # calculate AMM fund size for target default probability q
    # returns both solutions of the quadratic equation, the 
    # max of the two is the correct one
    kappa = L1/s2/K2
    a = np.exp(sig3**2)-1
    b = 2*(np.exp(sig3*sig2*rho)-1)
    c = np.exp(sig2**2)-1
    qinv2 = norm.ppf(q)**2
    v= -s3/s2/K2
    a0 = (a*qinv2-1)*v**2
    b0 = (b*qinv2-2+2*kappa*np.exp(-r))*v
    c0 = c*qinv2 - kappa**2*np.exp(-2*r)+2*kappa*np.exp(-r)-1
    Mstar1 = (-b0 + np.sqrt(b0**2-4*a0*c0))/(2*a0)
    Mstar2 = (-b0 - np.sqrt(b0**2-4*a0*c0))/(2*a0)

    # test correct solutions? - so this must be zero:
    # print(a0*Mstar1**2 + b0*Mstar1 +c0)
    # print(a0*Mstar2**2 + b0*Mstar2 +c0)
    return Mstar1, Mstar2


def get_DF_target_size(K2pair, k2trader, r2pair, r3pair, n,
                            s2, s3, currency_idx, leverage):
    """Calculate the target size for the default fund

    Args:
        K2pair ([type]): [description]
        k2trader ([type]): [description]
        r2pair ([type]): [description]
        r3pair ([type]): [description]
        n ([type]): [description]
        s2 ([type]): [description]
        s3 ([type]): [description]
        currency_idx ([int]): 1 for M1 (quote), 
            2 for M2 (base), 3 for M3 (quanto) 

    Returns:
        [float]: target size
    """
    K2pair = np.abs(K2pair) #/ leverage
    k2abs = np.abs(k2trader)  / leverage
    loss_down = (K2pair[0] + n * k2abs) * (1-np.exp(r2pair[0]))
    loss_up = (K2pair[1] + n * k2abs) * (np.exp(r2pair[1])-1) 
    if currency_idx==CollateralCurrency.QUOTE:
        return s2*np.max((loss_down, loss_up))
    elif currency_idx==CollateralCurrency.BASE:
        return np.max((loss_down/np.exp(r2pair[0]), loss_up/np.exp(r2pair[1])))
    elif currency_idx==CollateralCurrency.QUANTO:
        return (s2/s3)*np.max((loss_down/np.exp(r3pair[0]), loss_up/np.exp(r3pair[1])))
    


def test_default_probability():
    # benchmark for test of default probability in AMMPerp.tests.ts

    # setting
    K2=0.4
    L1=0.4*36000
    s2=38000
    s3=2000
    sig2=0.05
    sig3=0.07
    rho = 0.5
    M1 = 10
    M2 = 0.06
    M3 = 0.04
    r = 0
    k2=0

    q1, dd1=prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, 0)
    q2, dd2=prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    print("q1=",q1)
    print("dd1     =",norm.ppf(q1))
    print("dd1_orig=",dd1)
    print("q2=",q2)
    print("dd2     =",norm.ppf(q2))
    print("dd2_orig=",dd2)
    std_z = np.sqrt(get_variance_Z(r, sig2, sig3, rho, M3, s2, s3, M2, K2))
    print("std_z=",std_z)

    C3=M3*s3/(M2*s2-K2*s2)
    std_z = np.sqrt(get_variance_Z_withC(r, sig2, sig3, rho, C3))
    print("std_z=",std_z)

    print("C3=",C3)
    print("C3^2=",C3**2)
    print("varB1=", np.exp(rho*sig2*sig3))
    print("varB=", 2*(np.exp(rho*sig2*sig3)-1))

    print("PD5: approximate PD1 no quanto")
    pd1_approx = bad_cdf_approximation(dd1)
    print("PD = ", pd1_approx)

    print("PD5: approximate PD2 quanto")
    pd1_approx = bad_cdf_approximation(dd2)
    print("PD = ", pd1_approx)

def test_pricing():
    # benchmark for test of default probability in AMMPerp.tests.ts

    # setting
    K2=0.4
    L1=0.4*36000
    s2=38000
    s3=2000
    sig2=0.05
    sig3=0.07
    rho = 0.5
    M1 = 10
    M2 = 0.06
    M3 = 0.04
    r = 0
    k2=0
    isApprox = True # set to FALSE once correct CDF is available
    minSpread = 0.00
    dir = 1
    p1 = calculate_perp_price(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, dir, isApprox)
    print("p1=", p1)
    print("index =", s2)
    print("---")
    K2_alt=-K2
    L1_alt=-L1
    minSpread = 0.001
    p2 = calculate_perp_price(K2_alt, L1_alt, s2, s3, sig2, sig3, rho, r, M1, M2, 0, minSpread, dir, isApprox)
    print("p2=", p2)
    print("---")
    minSpread = 0.05
    p3 = calculate_perp_price(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, dir, isApprox)
    print("p3=", p3)
    
def mc_default_prob(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):

    assert(M3==0)
    N = 2
    n = 1e6
    num_defaults = 0
    mu = r-sig2**2/2
    for j in range(N):
        r2 = np.random.normal(mu, sig2, int(n))
        num_defaults += sum(np.exp(r2)*s2*(M2-K2)<-L1-M1)
    pd = num_defaults/(n*N)    
    return pd




def test_target_collateral():
    #  benchmark for test of target collateral in AMMPerp.tests.ts
    K2 = 1
    S2 = 36000
    S3 = 2000
    L1 = -36000
    sigma2 = 0.05
    sigma3 = 0.07
    rho = 0.5

    target_dd = norm.ppf(0.0015) # 15 bps
    # -2.9677379253417833
    print("target dd = ", target_dd)
    M1 = get_target_collateral_M1(K2, S2, L1, sigma2, target_dd)
    print("M1 = ", M1)

    M2 = get_target_collateral_M2(K2, S2, L1, sigma2, target_dd)
    print("M2 = ", M2)

    M3 = get_target_collateral_M3(K2, S2, S3, L1, sigma2, sigma3, rho, 0, target_dd)
    print("M3 = ", M3)


def test_insurance_fund_size():
    K2pair = np.array([-0.7, 0.8])
    k2_trader = np.array([-0.11, 0.15])
    fCoverN = 2
    r2pair = np.array([-0.30, 0.20])
    r3pair = np.array([-0.32, 0.18])
    s2 = 2000
    s3 = 31000
    lev = 20
    for currency_idx in range(3):
        i_star = get_DF_target_size(K2pair, k2_trader, r2pair, r3pair, fCoverN,\
                            s2, s3, currency_idx+1, lev)
        print("istar for M",currency_idx+1,": ", i_star)
        
def test_pd_monte_carlo():
    K2=2
    L1=2*46000
    M2=2
    M1=0
    M3=0
    sig2 =0.08
    sig3 = 0
    rho = 0
    r = 0
    s2 = 46000
    s3 = 0

    k_vec = np.arange(-8, 8, 0.25)
    pd_mc = np.zeros(k_vec.shape)
    pd_th = np.zeros(k_vec.shape)
    dd_th = np.zeros(k_vec.shape)
    idx = 0
    for k in k_vec:
        print(str(idx/k_vec.shape[0]*100)+"%")
        pd_mc[idx] = mc_default_prob(K2+k, L1+s2*k, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        print('mc  : {:.17f}%'.format(pd_mc[idx]*100))
        pd_th[idx],dd_th[idx] = prob_def_no_quanto(K2+k, L1+s2*k, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        print('th  : {:.17f}%'.format(pd_th[idx]*100))
        print('diff: {:.17f}%'.format(100*(pd_mc[idx]-pd_th[idx])))
        
        idx += 1
    
    fig, axs = plot.subplots(2)
    axs[0].plot(k_vec, 100*pd_mc, 'r:x', label='pd monte carlo')
    axs[0].plot(k_vec, 100*pd_th, 'k-o', label='pd theoretical')
    fig.suptitle("M2 = "+str(np.round(M2,2))+"BTC, L1="+str(L1)+"$, K2="+str(K2)+"BTC")
    axs[0].set(xlabel="Trade amount k2", ylabel="digital insurance, %")
    axs[0].grid(linestyle='--', linewidth=1)
    axs[0].legend()

    axs[1].plot(k_vec, dd_th, 'k-o', label='distance to default')
    axs[1].set(xlabel="Trade amount k2", ylabel="dd")
    axs[1].grid(linestyle='--', linewidth=1)
    axs[1].legend()
    plot.show()

def max_position_size(emwaTraderK):
    bumpUp = 20000#0.25
    return emwaTraderK*(1+bumpUp)

def max_trade_size(poscurrent, kStar, emwaTraderK):
    mx = max_position_size(emwaTraderK)
    max_trade_long = np.max((0, mx - poscurrent))
    max_trade_short = np.min((0, -mx - poscurrent))
    return (np.min((kStar, max_trade_short)), np.max((kStar, max_trade_long)))

def test_maxtradesize():
    print(max_trade_size(0.1, kStar=-0.2, emwaTraderK=0.1))
    print(max_trade_size(0.1, kStar=-0.2, emwaTraderK=0.1))
    print(max_trade_size(0.1, kStar= 0.2, emwaTraderK=0.3))
    print(max_trade_size(0.1, kStar=-0.2, emwaTraderK=0.3))
    print(max_trade_size(0.1, kStar= 0.2, emwaTraderK=0.3))
    print(max_trade_size(0.1, kStar= 0.2, emwaTraderK=0.1))
    

    # todo: only update emwaTrader if trade<maxAbs, otherwise
    #       risk can be increased in other direction
    #       hence if tradeAmount.abs()>max use max in EWMA

def test_case():
    """
    Assess specific AMM configuration
    """
    M2=0.25
    #cash_cc	0.267947331
    L1= 1025.990581
    K = 0.3091526707
    s2 = 15448.34
    sig2 = 0.05
    M1, M3, r, sig3, rho, s3 = 0,0,0,0,0,0
    minSpread = 0.0001
    isApprox=False
    posvec = np.arange(-1,1,0.01)
    pricevec = np.zeros(posvec.shape)
    priceMcBvec = np.zeros(posvec.shape)
    ddvec = np.zeros(posvec.shape)
    indvec = np.zeros(posvec.shape)
    for j in range(posvec.shape[0]):
        K2 = posvec[j]+K
        L = L1+posvec[j]*s2
        dir = np.sign(posvec[j])
        pricevec[j] = calculate_perp_price(K2, L, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, dir, isApprox)
        priceMcBvec[j] = calculate_McB_price(K, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        p,ddvec[j] = prob_def_no_quanto(K2, L, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        ddvec[j]=np.exp(ddvec[j]*sig2+(r-0.5*sig2**2))
        indvec[j]=np.sign(M2-K2)
    fig, axs = plot.subplots(1, 1)
    axs.plot(posvec+K, pricevec, 'r:', label="s2*(1+np.sign(K2+k-M2)*q)+sign(k)*minSpread")
    axs.set_xlabel("K+k")
    axs.set_ylabel("Price")
    axs.plot(posvec+K, priceMcBvec, 'b--', label = "McB: s2+alpha*np.sign(k2+K2)*(qplus-q), alpha:=s2")
    axs.plot(posvec+K, posvec*0+s2, 'k:', label = "s2")
    axs.legend()
    #axs[1].plot(posvec+K, 100*ddvec, 'b-')
    #axs[1].set_xlabel("K+k")
    #axs[1].set_ylabel("Price McB")
    #axs[1].set_title("s2+alpha*np.sign(k2+K2)*(qplus-q)")
    #axs[1].set_ylabel("NormInv(Q(K+k))")
    #axs[2].plot(posvec+K, 100*indvec, 'b-')
    #axs[2].set_xlabel("K+k")
    #axs[2].set_ylabel("sign(M2-K2-k2)")
    
    plot.show()

def test_DF_target_size():
    s2 = 18_000
    s3 = 1
    
    K2pair = [-1000*20/s2, 1000*20/s2]
    k2trader = 1000 * 20 / s2
    r2pair = [-0.15, 0.1]
    r3pair = [0, 0]
    n = 5
    lev = 20
    
    currency_idx = CollateralCurrency.QUOTE
    size = get_DF_target_size(K2pair, k2trader, r2pair, r3pair, n,
                            s2, s3, currency_idx, lev)
    print(f"size = {size}")

if __name__ == "__main__":
    #test_default_probability()
    #test_target_collateral()
    #test_pricing()
    #test_insurance_fund_size()
    #test_pd_monte_carlo()
    #test_case()
    # test_maxtradesize()
    test_DF_target_size()