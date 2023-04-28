import matplotlib.pyplot as plt
from pyinform import active_info
import pandas as pd
import numpy as np
import antropy as ant

Path_sorce_0 = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
Path_sorce_1 = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
Path_sorce_2= "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCSD_dataset.csv"
ratings0 = np.array(pd.read_csv(Path_sorce_0, header=0, usecols=[0]).values)
ratings1 = np.array(pd.read_csv(Path_sorce_1, header=0, usecols=[0]).values)
ratings2 = np.array(pd.read_csv(Path_sorce_2, header=0, usecols=[0]).values)

From_0 = 62520
From_1 = 39689
From_2 = 62520
to_=5000
target_0= np.reshape(np.array(ratings0[From_0:(From_0+to_)]),(5000))
target_1= np.reshape(np.array(ratings1[From_1:(From_1+to_)]),(5000))
target_2= np.reshape(np.array(ratings2[From_2:(From_2+to_)]),(5000))


# Permutation entropy
print(ant.perm_entropy(target_0, normalize=True))
print(ant.perm_entropy(target_1, normalize=True))
print(ant.perm_entropy(target_2, normalize=True))




species = ("Open Power", "UCSD",	"UCI")
penguin_means = {
    'LSTM': (0.97337913,	0.966299441,	0.954759029),
    'Qut+linear': (0.828742743,	0.877451721,	0.705109778),
    'My approach': (0.998512202,	0.988089706,	0.983699995),

    'LSTM+att': (0.993009025,	0.959001139,	0.946441012),
    'Qut+RF': (0.975825737,	0.97748179,	0.967389638),
    'Bi-LSTM+att': (0.973665438,	0.955428876,	0.938083561),

}

x = np.arange(len(species))  # the label locations
width = 0.15  # the width of the bars
multiplier = 1

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=4)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R square')
ax.set_title('R square in different datasets')
ax.set_xticks(x + width, species)
ax.legend(loc='lower right', ncols=3)
# ax.set_ylim(0, 250)

plt.show()