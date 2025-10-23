import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
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

categories = [
    "7/2021--9/2021", "9/2021--11/2021", "11/2021--1/2022", "1/2022--3/2022",
    "3/2022--5/2022", "5/2022--7/2022", "7/2022--9/2022", "9/2022--11/2022",
    "11/2022--1/2023", "1/2023--3/2023", "3/2023--5/2023", "5/2023--7/2023",
    "7/2023--9/2023", "9/2023--11/2023", "11/2023--1/2024", "1/2024--3/2024",
    "3/2024--5/2024"
]


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
y = bottom_data[2]  # T6

interruption_index = categories.index("7/2023--9/2023")

df = pd.DataFrame({
    "time": np.arange(1, len(y) + 1),
    "value": y,
    "interruption": (np.arange(len(y)) >= interruption_index).astype(int)
})
df["time_after_interruption"] = (df["time"] - df["time"][interruption_index]) * df["interruption"]

X = sm.add_constant(df[["time", "interruption", "time_after_interruption"]])
model = sm.OLS(df["value"], X).fit()

print(model.summary())

b0, b1, b2, b3 = model.params
p_level = model.pvalues["interruption"]
p_slope = model.pvalues["time_after_interruption"]

df["predicted"] = model.predict(X)

pre_df = df[df["interruption"] == 0]
post_df = df[df["interruption"] == 1]

pre_trend = b0 + b1 * pre_df["time"]
post_trend = (b0 + b2) + (b1 + b3) * post_df["time"]

plt.figure(figsize=(12, 6))
plt.plot(categories, y, 'o-', label="Observed", color='blue', alpha=0.7)
plt.plot(pre_df.index, pre_trend, 'g--', linewidth=2, label="Pre-interruption trend")
#plt.plot(post_df.index, post_trend, 'r--', linewidth=2, label="Post-interruption trend")
plt.axvline(x=interruption_index, color='black', linestyle='--', alpha=0.7)
#plt.text(interruption_index + 0.3, max(y)*0.9, "Interruption\n(7/2023–9/2023)", color='black')

plt.title("Interrupted Time Series Analysis — T6 (Death)")
plt.xlabel("Bimonthly Period")
plt.ylabel("Number of Tweets")
plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("ITS Results Summary")
print(f"Immediate level change (β₂) = {b2:.2f}, p = {p_level:.4f}")
print(f"Slope change (β₃) = {b3:.2f}, p = {p_slope:.4f}")