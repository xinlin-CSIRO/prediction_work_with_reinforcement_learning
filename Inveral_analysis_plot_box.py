import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Creating dataset
# result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\results_summary\\intervals_analysis_UCI.csv"
# result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\results_summary\\intervals_analysis_European.csv"
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\results_summary\\intervals_analysis_UCSD.csv"
ratings = pd.read_csv(result_location, header=0, usecols=[0,2,3])
dataset = ratings.values
dataset_initial = ratings.values

np.random.seed(10)
data_1 =dataset_initial[:,0]
data_2 = dataset_initial[:,1]
data_3 =dataset_initial[:,2]
# data_4 =dataset_initial[:,3]
# data = [data_1, data_2, data_3,data_4]
data = [data_1, data_2, data_3]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=0)

colors = ['#0000FF', '#00FF00',
          '#c083f2', '#FF00FF']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linewidth=1.5,
                linestyle=":",

                )

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B',
            linewidth=2,
            )

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color='red',
               linewidth=3)

# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker='D',
              color='#e7298a',
              alpha=0.5)

# x-axis labels
ax.set_yticklabels(['Qua_RF', 'Qua_Linear','Our method'])

ax.tick_params(axis="x", labelsize=14)
# ax.tick_params(axis="y", labelsize=12)
# ax.set_yticklabels(labels=ax.get_yticklabels(), Fontsize=8)

# Adding title
# plt.title("Open Power dataset-based")
# plt.title("UCSD dataset-based")
# plt.title("UCI dataset-based")
# Removing top axes and right axes
# ticks
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

# show plot
plt.show()
