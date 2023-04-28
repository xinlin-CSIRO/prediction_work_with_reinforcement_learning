import math
import zlib
import numpy as np
import pandas as pd
##data for forecasting##
Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCSD_dataset.csv"
ratings = pd.read_csv(Path_sorce_, header=0, usecols=[0])

dataset = ratings.values

source=Path_sorce_.split('\\')[-1]
if('UCI' in source):
    From_i = 62520
    data_source='UCI_'
elif('European' in source):
    From_i = 39689
    data_source = 'European_'
elif ('UCSD' in source):
    From_i = 62520
    data_source = 'UCSD_'
to_=From_i+5000

target_dataset= np.reshape(np.array(dataset[From_i:to_]),(5000))
# Assume we have a time series stored as a list or numpy array called 'data'
# Convert the data to a string of bytes using the 'tostring()' method
data_str = target_dataset.tobytes()

# Compress the data using the zlib library
compressed_data = zlib.compress(data_str)

# Calculate the compression ratio
compression_ratio = len(compressed_data) / len(data_str)

# Estimate the Kolmogorov complexity as the negative logarithm of the compression ratio
kolmogorov_complexity = -1 * math.log2(compression_ratio)

print("Estimated Kolmogorov complexity:", kolmogorov_complexity)