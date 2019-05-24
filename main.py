import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# some test data
x = np.arange(500)/200
y1 = np.sin(1.8*x)*np.random.uniform(0.5, 1.1, len(x))
y2 = np.cos(1.3*x-2)
y = y1 + y2

# target variable (up or down movement)

threshold = 0.1
shift = 100
up = (y[:-shift] > (1+threshold)*y[shift:])*1  # possibly fix
down = (y[:-shift] < (1-threshold)*y[shift:])*(-1)
target = up+down

plt.plot(up + down)
plt.plot(y)
plt.show()

# combine all into a single array
feat = (np.concatenate([y1[:-shift], y2[:-shift], target],
                       axis=0).reshape((3, len(x)-shift))).transpose()
lsplit = 4
csplit = 2


# create shifted arrays
all = []
for i in range(lsplit):
    all.append(feat[i:len(feat)-lsplit+1+i])

# reshape into the correct form for cnn2d (second reshape) with lstm (first reshape)
r = (np.concatenate(all)).reshape((-1,
                                   lsplit,
                                   feat.shape[1]),
                                  order='F').reshape((-1,
                                                      int(lsplit /
                                                          csplit),
                                                      csplit, feat.shape[1], 1))
design_y = r[:, -1, -1, -1, 0]
