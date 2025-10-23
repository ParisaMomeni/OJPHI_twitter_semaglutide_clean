import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

 #Parisa rcparams
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 15  # Base font size
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# temporal charts
categories = [
    "7/2021--9/2021", "9/2021--11/2021", "11/2021--1/2022", "1/2022--3/2022",
    "3/2022--5/2022", "5/2022--7/2022", "7/2022--9/2022", "9/2022--11/2022",
    "11/2022--1/2023", "1/2023--3/2023", "3/2023--5/2023", "5/2023--7/2023",
    "7/2023--9/2023", "9/2023--11/2023", "11/2023--1/2024", "1/2024--3/2024",
    "3/2024--5/2024"
]
n_categories = len(categories)
colors = ['blue', 'orange', 'green', 'red', 'purple']
dashes = [[1, 0], [2, 2], [3, 2], [1, 2], [3, 1]]
markers = ['o', 's', '^', 'D', 'v']

top_data = np.array([
    [234, 158, 208, 307, 321, 439, 366, 576, 1296, 2648, 5029, 6247, 5241, 6029, 6701, 8903, 5839],
    [26, 30, 15, 69, 43, 42, 59, 149, 289, 1020, 2727, 4133, 3914, 4878, 6168, 8562, 7996],
    [99, 66, 130, 125, 194, 260, 274, 435, 1041, 1305, 1761, 1647, 2479, 4104, 2020, 4479, 2305],
    [109, 77, 116, 96, 154, 174, 204, 279, 614, 1165, 2284, 2592, 2400, 3708, 3662, 3599, 2550],
    [56, 56, 59, 111, 105, 118, 113, 162, 308, 1057, 2230, 3314, 2881, 3486, 5174, 5973, 4272]
])
top_labels = ["T0", "T1", "T2", "T4", "T9"]

bottom_data = np.array([
    [78, 70, 45, 67, 111, 155, 143, 301, 1086, 1473, 1508, 1783, 1869, 1616, 2092, 2790, 1761],
    [7, 14, 8, 33, 23, 30, 26, 47, 117, 260, 635, 947, 888, 1193, 1449, 2174, 1637],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 5, 6, 13, 502, 145, 511, 280, 23],
    [10, 4, 4, 12, 8, 16, 14, 34, 49, 175, 304, 468, 323, 428, 375, 512, 339],
    [24, 27, 19, 44, 61, 61, 77, 79, 141, 332, 631, 1034, 1762, 1529, 1127, 1992, 1268]
])
bottom_labels = ["T3", "T5", "T6", "T7", "T8"]

x = np.arange(1, top_data.shape[1] + 1)

# Simulate small variation for confidence interval visualization
np.random.seed(42)
n_samples = 30
top_simulated_data = [np.random.normal(loc=row, scale=np.maximum(row * 0.1, 1), size=(n_samples, len(row)))
                  for row in top_data]
bottom_simulated_data = [np.random.normal(loc=row, scale=np.maximum(row * 0.1, 1), size=(n_samples, len(row)))
                  for row in bottom_data]

# Calculate mean and 95% CI
top_means = [np.mean(d, axis=0) for d in top_simulated_data]
top_stds = [np.std(d, axis=0, ddof=1) for d in top_simulated_data]
top_conf_ints = [1.96 * (s / np.sqrt(n_samples)) for s in top_stds]

bottom_means = [np.mean(d, axis=0) for d in bottom_simulated_data]
bottom_stds = [np.std(d, axis=0, ddof=1) for d in bottom_simulated_data]
bottom_conf_ints = [1.96 * (s / np.sqrt(n_samples)) for s in bottom_stds]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
colors = plt.cm.tab10(np.arange(top_data.shape[0]))


for i in range(top_data.shape[0]):
    ax1.plot(x, top_means[i], marmarker = markers[i], dashes = dashes[i], label=f'{top_labels[i]}', color=colors[i])
    ax1.fill_between(x,
                     top_means[i] - top_conf_ints[i],
                     top_means[i] + top_conf_ints[i],
                     color=colors[i],
                     alpha=0.2)

ax1.set_ylabel("Number of Posts")
ax1.legend(loc='upper left')
ax1.grid(True)

for i in range(bottom_data.shape[0]):
    ax2.plot(x, bottom_means[i], marker = markers[i], dashes = dashes[i], label=f'{bottom_labels[i]}', color=colors[i])
    ax2.fill_between(x,
                     bottom_means[i] - bottom_conf_ints[i],
                     bottom_means[i] + bottom_conf_ints[i],
                     color=colors[i],
                     alpha=0.2)

ax2.set_ylabel("Number of Posts")
ax2.legend(loc='upper left')
ax2.grid(True)

plt.xticks(x, categories, rotation=45, ha='right')
plt.xlabel("Bimonthly Period")
plt.tight_layout()
plt.show()