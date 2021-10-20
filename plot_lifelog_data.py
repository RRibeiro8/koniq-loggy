import pandas as pd, numpy as np, os
from matplotlib import pyplot as plt
from scipy import stats

ids = pd.read_csv('orignal_lsc21-MOS.csv')
print(ids.query("MOS < 30").shape[0])

print(ids.image_path)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(ids.MOS.values, num_bins)

ax.plot()
ax.set_xlabel('MOS')
ax.set_ylabel('Images number')
ax.set_title(r'MOS distribution in lifelog images')
plt.xlim([0, 100])
plt.ylim([0, 20000])
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()