# plot funding rate as a function of premium rate
# used for cheatsheet
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plot


r = np.arange(-0.0008, 0.0008, 0.000001)
delta = 0.0005
b = 0.0001
f = np.zeros(r.shape)
for k in range(r.shape[0]):
    f[k] = np.max((r[k],delta)) + np.min((r[k], -delta)) + np.sign(r[k])*b

fig, ax = plot.subplots()
ax.plot(r*100, f*100, linewidth=3)
plot.xticks([-delta*100, 0, delta*100],fontsize=10)
plot.yticks([-0.01, 0, 0.01],fontsize=10)
ax.grid()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '-∆'
labels[1] = '0'
labels[2] = '∆'
ax.set_xticklabels(labels)
plot.xlabel("Premium Rate, %",fontsize=10)
plot.ylabel("Funding Rate, %",fontsize=10)
plot.savefig("fundingrate.png")
plot.show()