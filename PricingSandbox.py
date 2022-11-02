#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plot


def prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    # insurance premium for given level m of quanto fund (M3:=m)
    C3=M3*s3/(M2*s2-K2*s2)
    sigz = np.sqrt(get_variance_Z_withC(r, sig2, sig3, rho, C3))
    muz = np.exp(r)*(1+C3)
    dd = ((-L1-M1)/(s2*(M2-K2))-muz)/sigz
   
    if M2-K2<0:
        dd = -dd
    qobs = norm.cdf(dd)
    return qobs, dd


def get_variance_Z_withC(r, sig2, sig3, rho, C3):
    return np.exp(2*r)*(
        (np.exp(sig3**2)-1)*C3**2 + (np.exp(sig2**2)-1) +
        2*(np.exp(sig2*sig3*rho)-1)*C3
    )

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

def calculate_perp_priceOLD(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread):
    # was calculate_perp_price
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

def calculate_perp_price(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incentiveSpread):
    dL = k*s2
    if M3==0:
        qp, dd = prob_def_no_quanto(K2+k+0.0001, L1+dL+0.0001*s2, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        q, dd = prob_def_no_quanto(K2+k, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q0, dd = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        q, dd = prob_def_quanto(K2+k, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)

    u = -L1/s2 - M1/s2
    v = K2 - M2
    kStar = (u-v)/2
    #sgnm = np.sign(k - kStar)
    sgnm = 1 if qp > q else -1
    #Keq = K2-kStar
    #scale = np.abs( np.exp(5*sig2-sig2**2/2)*(s2*M2-s2*Keq)+L1+M1 ) / ( s2*(np.exp(5*sig2 - sig2**2/2)-1) )
    scale = 1
    incentive = incentiveSpread/scale * (k-kStar)
    return s2*(1 + sgnm*q + np.sign(k)*minSpread + incentive)

def calculate_perp_priceDeltaQ(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread):
    # handle discontinuity
    k_approx = k
    if np.abs(k)<0.00001:
        # keep k flat for unrealistically small quantities
        k_approx = np.sign(k)*0.001

    dL = k_approx*s2
    if M3==0:
        q0, dd = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3) 
        q, dd = prob_def_no_quanto(K2+k_approx, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)       
    else:
        assert(0)

    v = s2*(q-q0)/k_approx
    minSpread = minSpread
    #incentive = (np.abs(kStar - k)-np.abs(kStar))/k*incentive_spread
    return (s2 +  v) + np.sign(k)*minSpread

    #q0 : global min
    #q-q0 > 0
    # long pays s2*(1+abs(p))
    # short pays s2*(1-abs(p))

def calculate_perp_QDiff(K2, k, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    dL = k*s2
    assert(M3==0)
    q0, dd = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3) 
    qK, dd = prob_def_no_quanto(K2+k, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    return qK-q0
    
def calculate_McB_price(K2, k2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    dL = k2*s2
    if M3==0:
        q, _ = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_no_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q, _ = prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    kStar = (-L1/s2 - M1/s2 - K2+M2)/2
    delta = 0.0001
    ins = (qplus-q) + delta * (np.abs(kStar-k2) - np.abs(k2))
    beta = 1
    return s2 * (1+beta*ins/k2)*k2
    #return s2+alpha*(qplus-q)

def calculate_contribution_price(K2, k2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3):
    dL = k2*s2
    if M3==0:
        q, _ = prob_def_no_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_no_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)    
    else:
        q, _ = prob_def_quanto(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qplus, _ = prob_def_quanto(K2+k2, L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    kStar = (-L1/s2 - M1/s2 - K2+M2)/2
    minspread=0.0001    
    return s2*(1+np.sign(k2 - kStar)*(qplus-q)/k2)



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
    minSpread = 0.01

    u = -L1/s2 - M1/s2
    v = K - M2
    kStar = (u-v)/2
    print("Kstar2 = ", kStar)
    
    #K = K+kStar
    #L1 = L1 +kStar*17815.6464651
    posvec = np.sort(np.concatenate((np.arange(-0.15,0.1,0.001), [kStar])))
    pricevec = np.zeros(posvec.shape)
    priceMcBvec = np.zeros(posvec.shape)
    priceContribVec = np.zeros(posvec.shape)
    ddvec = np.zeros(posvec.shape)
    indvec = np.zeros(posvec.shape)
    spread = 0.0010
    incntSpread = 0.015 # 1.5 percent for 1 BTC trade
    for j in range(posvec.shape[0]):
        K2 = posvec[j]+K
        L = L1+posvec[j]*s2
        dir = np.sign(posvec[j])
        pricevec[j] = calculate_perp_price(K, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, spread, incntSpread)
        p,ddvec[j] = prob_def_no_quanto(K2, L, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        ddvec[j]=np.exp(ddvec[j]*sig2+(r-0.5*sig2**2))
        indvec[j]=np.sign(M2-K2)
        
        
    fig, axs = plot.subplots(2, 2)

    idx = np.where(posvec==kStar)
    pxStar = calculate_perp_price(K, posvec[idx], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread)
    print("pxStar =", pxStar)

    u = -L1/s2 - M1/s2
    v = K - M2
    kStar = (v-u)/2
    print("Kstar = ", kStar)
    pxMid = calculate_perp_price(K, 0, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread)
    #axs[0,0].plot(posvec+K, pricevec, 'r:', label="Perp Price")
    #axs[0,0].plot(K, pxMid, 'rx', label="Mid Price")
    axs[0,0].set_xlabel("K2+k2")
    axs[0,0].set_ylabel("Price")
    axs[0,0].axvline(x=K-kStar, color='g', linestyle='-', label="K-Kstar")
    axs[0,0].axvline(x=K, color='c', linestyle='-', label="K")
    axs[0,0].plot(posvec+K, pricevec, 'b--', label = "Perp Price Incremental")
    axs[0,0].plot(posvec+K, posvec*0+s2, 'k:', label = "s2")
    
    axs[0,0].legend()
    axs[0,1].plot(posvec+K, 100*ddvec, 'b-')
    axs[0,1].set_xlabel("K2+k2")
    axs[0,1].set_ylabel("DD [=argument to Phi(.)]")
    axs[0,1].axvline(x=K, color='k', linestyle='--', label="K2")
    axs[0,1].axvline(x=K-kStar, color='g', linestyle='-', label="K-Kstar")
    axs[0,1].legend()
    axs[1,1].plot(posvec+K, indvec, 'b-', label="K2-k2-M2")
    axs[1,1].set_xlabel("K2+k2")
    axs[1,1].set_ylabel("Price")
    axs[1,1].axvline(x=K, color='k', linestyle='--', label="K2")
    axs[1,1].axvline(x=K-kStar, color='g', linestyle='-', label="K-Kstar")
    axs[1,1].legend()
    
    plot.show()


def plot_range(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread, kFrom=-0.15, kTo=0.15):
    kStar = getKstar(K2, L1, s2, M1, M2)
    rng = np.arange(kFrom,kTo,0.001)
    #rng = np.array([kFrom, kTo])
    posvec = np.sort(np.concatenate((rng, [kStar])))
    priceVec1 = np.zeros(posvec.shape)
    priceVec2 = np.zeros(posvec.shape)
    qVec = np.zeros(posvec.shape)
    for j in range(posvec.shape[0]):
        priceVec1[j] = calculate_perp_price(K2, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread)
        priceVec2[j] = calculate_perp_price(K2, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, 0)
        qVec[j] = calculate_perp_QDiff(K2, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        #priceVec[j] = calculate_perp_price(K2, posvec[j], L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread)

    return posvec, priceVec1, priceVec2, qVec


def getKstar(K, L1, s2, M1, M2):
    #calculate Kstar
    u = -L1/s2 - M1/s2
    v = K - M2
    return (u-v)/2

def q_symmetry():
    M2=0.25
    #cash_cc	0.267947331
    L1= 1025.990581
    K = 0.3091526707
    s2 = 15448.34
    sig2 = 0.05
    M1, M3, r, sig3, rho, s3 = 0,0,0,0,0,0
    r = (sig2**2)*0.5
    if M1/s2 + M2 <= 0:
        print("DEFAULT")
    kStar = getKstar(K, L1, s2, M1, M2)
    PD0 = prob_def_no_quanto(K, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    print("Kstar=",kStar)
    print("PD0=",PD0)
    posvec = np.sort(np.concatenate((np.arange(-0.15,0.15,0.001), [kStar])))
    qVec = np.zeros(posvec.shape)
    for j in range(posvec.shape[0]):
        dL = posvec[j]*s2
        q0, dd = prob_def_no_quanto(K+posvec[j], L1+dL, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        qVec[j] = q0
    fig, axs = plot.subplots()
    axs.plot(posvec+K, qVec, 'b-', label="q(K+k)")
    idx = np.where(posvec==kStar)
    axs.plot(K+kStar, qVec[idx], 'rx', label="k*")
    axs.plot(K-M2, qVec[idx], 'k+', label='K2-M2')
    axs.legend()
    plot.show()

def test_dynamics():
    """
    Assess specific AMM configuration
    and trade consecutively
    """
    M2=0.25
    #cash_cc	0.267947331
    L1= 1025.990581
    K = 0.3091526707
    s2 = 15448.34
    sig2 = 0.05
    M1, M3, r, sig3, rho, s3 = 0,0,0,0,0,0
    r = (sig2**2)*0.5
         
    kStar = getKstar(K, L1, s2, M1, M2)
    
    minSpread=0
    incntSpread=0.0100
    for j in range(3):
        kStar = getKstar(K, L1, s2, M1, M2)
        PD0 = prob_def_no_quanto(K, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
        print("Kstar=",kStar)
        print("PD0=",PD0)
        #plot_range(K2, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, kFrom=-0.15, kTo=0.15)
        [x, y1, y2, dQ] = plot_range(K, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread, kFrom=-0.15, kTo=0.05)
        fig1, ax1 = plot.subplots(1,2)
        ax1[0].plot(K+x, y1, label="s2*(1 + (q-q0)/k )+hat")
        ax1[0].plot(K+x, y2, label="s2*(1 + (q-q0)/k )")
        ax1[0].set_title("K="+str(K))
        ax1[0].axvline(x=K+kStar, color='m', linestyle='-', label="K-Kstar")
        ax1[0].axvline(x=K, color='m', linestyle='--', label="K")
        ax1[0].set_xlabel("K+k")
        #ax1[0].set_ylim(-s2*1.5, s2*4)
        ax1[0].axhline(y=s2, color='k', linestyle='--', label="S2")
        ax1[0].legend()
        ax1[1].plot(K+x, dQ, label="Q(k)-Q(0)")
        ax1[1].set_xlabel("K+k")
        ax1[1].legend()
        plot.show()
        px = calculate_perp_price(K, kStar, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread, incntSpread)
        
        #px = calculate_perp_price(K, kStar, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3, minSpread)
        K = K+kStar
        L1= L1 + kStar*(px)
        
        kStar = getKstar(K, L1, s2, M1, M2)
    
    PD0 = prob_def_no_quanto(K, L1, s2, s3, sig2, sig3, rho, r, M1, M2, M3)
    print("Kstar=",kStar)
    print("PD0=",PD0)
        


if __name__ == "__main__":
    #test_default_probability()
    #test_target_collateral()
    #test_pricing()
    #test_insurance_fund_size()
    #test_pd_monte_carlo()
    #test_case()
    test_dynamics()
    #q_symmetry()