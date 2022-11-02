#%% 
import matplotlib.pyplot as plt
import simulation
from datetime import datetime, date, timezone
import numpy as np
import matplotlib.dates as mdates

from_date = datetime(2020, 11, 1, 0, 0, tzinfo=timezone.utc).timestamp()
to_date = datetime(2021, 12, 31, 1, 0, tzinfo=timezone.utc).timestamp()
#to_date = datetime(2020, 11, 4, 1, 0, tzinfo=timezone.utc).timestamp()
# set reload to True after changing dates:
(idx_px, bitmex_px, time_df) = simulation.init_index_data(from_date, to_date, reload=True)
#%% 

idx = np.arange(1, idx_px.shape[0], 60*24)
fig, ax = plt.subplots()
t = [datetime.strptime(k, '%y-%m-%d %H:%M') for k in time_df[idx]]
ax.plot(t, idx_px[idx])
#ax.xticks(time_df[idx].iloc[np.arange(1, idx.shape[0], 7)], rotation=90)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
# tick on mondays every second week
#loc = mdates.MonthLocator(interval=1)
#loc = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
#locator = mdates.AutoDateLocator()
#ax.format_xdata = mdates.DateFormatter(loc, '%Y-%m-%d')
fig.autofmt_xdate()
ax.grid()
plt.savefig("test.png")
plt.show()

# %%
