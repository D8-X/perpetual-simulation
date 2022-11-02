#/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Plot pricing curve in debug state
# --> breakpoint in perpetual.trade -> then execute this script in debug console
#
#

import matplotlib.pyplot as plot
par=[0.05, 0.08]
for j in range(len(par)):
    self.params['sig2']=par[j]

    amounts = np.arange(-10,10, 0.2)
    price = np.zeros(amounts.shape[0])
    print("pool cash = ", self.amm_pool_cash_cc)
    L = np.round(-self.amm_trader.locked_in_qc, 2)
    print("locked-in value $= ", L)
    K = np.round(-self.amm_trader.position_bc)
    for k in range(amounts.shape[0]):
        price[k] = (self.get_price(amounts[k])-self.idx_s2[self.current_time])/self.idx_s2[self.current_time]

    if j==1:
        plot.plot(amounts, 100*price, 'b--', label='sig=0.08')
    else:
        plot.plot(amounts, 100*price, 'r:', label='sig=0.05')
        plot.title("M2 = "+str(np.round(self.amm_pool_cash_cc,2))+"BTC, L1="+str(L)+"$, K2="+str(K)+"BTC")
        plot.xlabel("Trade amount k2")
        plot.ylabel("Price deviation from spot, %")
plot.grid(linestyle='--', linewidth=1)
plot.legend()
plot.show()

self.params['sig2']=0.05
