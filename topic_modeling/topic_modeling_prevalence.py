import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd

state_to_region = {
        'Alabama': 'Southeast',
        'Alaska': 'West',
        'Arizona': 'Southwest',
        'Arkansas': 'Southeast',
        'California': 'West',
        'Colorado': 'West',
        'Connecticut': 'Northeast',
        'Delaware': 'Northeast',
        'Florida': 'Southeast',
        'Georgia': 'Southeast',
        'Hawaii': 'West',
        'Idaho': 'West',
        'Illinois': 'Midwest',
        'Indiana': 'Midwest',
        'Iowa': 'Midwest',
        'Kansas': 'Midwest',
        'Kentucky': 'Southeast',
        'Louisiana': 'Southeast',
        'Maine': 'Northeast',
        'Maryland': 'Northeast',
        'Massachusetts': 'Northeast',
        'Michigan': 'Midwest',
        'Minnesota': 'Midwest',
        'Mississippi': 'Southeast',
        'Missouri': 'Midwest',
        'Montana': 'West',
        'Nebraska': 'Midwest',
        'Nevada': 'West',
        'New Hampshire': 'Northeast',
        'New Jersey': 'Northeast',
        'New Mexico': 'Southwest',
        'New York': 'Northeast',
        'North Carolina': 'Southeast',
        'North Dakota': 'Midwest',
        'Ohio': 'Midwest',
        'Oklahoma': 'Southwest',
        'Oregon': 'West',
        'Pennsylvania': 'Northeast',
        'Rhode Island': 'Northeast',
        'South Carolina': 'Southeast',
        'South Dakota': 'Midwest',
        'Tennessee': 'Southeast',
        'Texas': 'Southwest',
        'Utah': 'West',
        'Vermont': 'Northeast',
        'Virginia': 'Southeast',
        'Washington': 'West',
        'West Virginia': 'Southeast',
        'Wisconsin': 'Midwest',
        'Wyoming': 'West',
        'District of Columbia': 'Northeast',
}

rcParams['font.family'] = 'Serif'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

df = pd.read_csv("../../data/Semaglutide_Twitter_Topic_mapping.csv") # Full dataset not included in repository
df['state_to_region'] = df['Region'].map(state_to_region)
df['state_to_region'] = df['state_to_region'].fillna('Unknown')

# Subpopulation charts
categories = ["T0","T1","T2","T3","T4", "T5","T6","T7","T8","T9"]
n_groups = len(categories)
colors = ["blue", "orange", "green", "red", "purple"]
hatches = ["/", "\\", "/", "\\", "/"]
bar_width = 0.25
x = np.arange(n_groups)

### Gender data
gender_n_bars = 3
gender_bar_names = ["Male", "Female", "Unknown"]
gender_bar_data = np.array([
    df[df["Gender"] == "male"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Gender"] == "female"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Gender"] == "unknown"]["Topic_mapping"].value_counts(sort=False).sort_index()
])

gender_n_lines = 3
gender_line_names = ["Male %", "Female %", "Unknown %"]
gender_line_data = np.array([
    gender_bar_data[0] / gender_bar_data[0].sum(),
    gender_bar_data[1] / gender_bar_data[1].sum(),
    gender_bar_data[2] / gender_bar_data[2].sum()
])

### Verification status data
verified_n_bars = 2
verified_bar_names = ["Verified", "Not Verified"]
verified_bar_data = np.array([
    df[df["Twitter Verified"] == True]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Twitter Verified"] == False]["Topic_mapping"].value_counts(sort=False).sort_index()
])

verified_n_lines = 2
verified_line_names = ["Verified %", "Not Verified %"]
verified_line_data = np.array([
    verified_bar_data[0] / verified_bar_data[0].sum(),
    verified_bar_data[1] / verified_bar_data[1].sum()
])

### Account type data
account_type_n_bars = 2
account_type_bar_names = ["Individual", "Organizational"]
account_type_bar_data = np.array([
    df[df["Account Type"] == "individual"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Account Type"] == "organisational"]["Topic_mapping"].value_counts(sort=False).sort_index()
])

account_type_n_lines = 2
account_type_line_names = ["Individual %", "Organizational %"]
account_type_line_data = np.array([
    account_type_bar_data[0] / account_type_bar_data[0].sum(),
    account_type_bar_data[1] / account_type_bar_data[1].sum()
])

### Interests data
interests_n_bars = 5
interests_bar_names = ["Business", "Tech", "Beauty", "Science", "Shopping"]
interests_bar_data = np.array([
    df[df["Interest"].str.contains("Business", na=False)]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Interest"].str.contains("Technology", na=False)]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Interest"].str.contains("Beauty/Health & Fitness", na=False)]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Interest"].str.contains("Science", na=False)]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["Interest"].str.contains("Shopping", na=False)]["Topic_mapping"].value_counts(sort=False).sort_index()
])

interests_n_lines = 5
interests_line_names = ["Business %", "Tech %", "Beauty %", "Science %", "Shopping %"]
interests_line_data = np.array([
    interests_bar_data[0] / interests_bar_data[0].sum(),
    interests_bar_data[1] / interests_bar_data[1].sum(),
    interests_bar_data[2] / interests_bar_data[2].sum(),
    interests_bar_data[3] / interests_bar_data[3].sum(),
    interests_bar_data[4] / interests_bar_data[4].sum()
])


### Region data
region_n_bars = 5
region_bar_names = ["Midwest", "Northeast", "Southeast", "Southwest", "West"]
region_bar_data = np.array([
    df[df["state_to_region"] == "Midwest"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["state_to_region"] == "Northeast"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["state_to_region"] == "Southeast"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["state_to_region"] == "Southwest"]["Topic_mapping"].value_counts(sort=False).sort_index(),
    df[df["state_to_region"] == "West"]["Topic_mapping"].value_counts(sort=False).sort_index(),
])

region_n_lines = 5
region_line_names = ["Midwest %", "Northeast %", "Southeast %", "Southwest %", "West %"]
region_line_data = np.array([
    region_bar_data[0] / region_bar_data[0].sum(),
    region_bar_data[1] / region_bar_data[1].sum(),
    region_bar_data[2] / region_bar_data[2].sum(),
    region_bar_data[3] / region_bar_data[3].sum(),
    region_bar_data[4] / region_bar_data[4].sum()
])

fig, (ax1, ax3, ax5, ax7, ax9) = plt.subplots(5, 1, figsize=(12, 15))

### Gender subplot
for i in range(gender_n_bars):
    ax1.bar(
        x + i * bar_width,
        gender_bar_data[i],
        width=bar_width,
        facecolor="none",        # transparent fill
        edgecolor=colors[i],     # colored border
        hatch=hatches[i],        # hatch pattern
        label=f'{gender_bar_names[i]}'
    )

ax1.set_xlabel("Topic Number")
ax1.set_ylabel("Number of Tweeets", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x + bar_width * (gender_n_bars - 1) / 2)
ax1.set_xticklabels([f'{i}' for i in categories])

ax2 = ax1.twinx()
for i in range(gender_n_lines):
    ax2.plot(
        x + bar_width,   # center of group
        gender_line_data[i],
        marker='o',
        linewidth=2,
        label=f'{gender_line_names[i]}'
    )

ax2.set_ylabel("Percentage of Subpopulation", color='black')
ax2.tick_params(axis='y', labelcolor='black')

# --- Combine legends ---
gender_bars_handles, gender_bars_labels = ax1.get_legend_handles_labels()
gender_lines_handles, gender_lines_labels = ax2.get_legend_handles_labels()
ax1.legend(gender_bars_handles + gender_lines_handles, gender_bars_labels + gender_lines_labels, loc='upper right')
ax1.set_title("Topic Prevalence by Gender")

### Verification status subplot
for i in range(verified_n_bars):
    ax3.bar(
        x + i * bar_width,
        verified_bar_data[i],
        width=bar_width,
        facecolor="none",        # transparent fill
        edgecolor=colors[i],     # colored border
        hatch=hatches[i],        # hatch pattern
        label=f'{verified_bar_names[i]}'
    )

ax3.set_xlabel("Topic Number")
ax3.set_ylabel("Number of Tweeets", color='black')
ax3.tick_params(axis='y', labelcolor='black')
ax3.set_xticks(x + bar_width * (verified_n_bars - 1) / 2)
ax3.set_xticklabels([f'{i}' for i in categories])

# --- Lines on right y-axis ---

ax4 = ax3.twinx()
for i in range(verified_n_lines):
    ax4.plot(
        x + bar_width,   # center of group
        verified_line_data[i],
        marker='o',
        linewidth=2,
        label=f'{verified_line_names[i]}'
    )

ax4.set_ylabel("Percentage of Subpopulation", color='black')
ax4.tick_params(axis='y', labelcolor='black')

# --- Combine legends ---
verified_bars_handles, verified_bars_labels = ax3.get_legend_handles_labels()
verified_lines_handles, verified_lines_labels = ax4.get_legend_handles_labels()
ax3.legend(verified_bars_handles + verified_lines_handles, verified_bars_labels + verified_lines_labels, loc='upper right')
ax3.set_title("Topic Prevalence by Verification Status")

### Account type subplot
for i in range(account_type_n_bars):
    ax5.bar(
        x + i * bar_width,
        account_type_bar_data[i],
        width=bar_width,
        facecolor="none",        # transparent fill
        edgecolor=colors[i],     # colored border
        hatch=hatches[i],        # hatch pattern
        label=f'{account_type_bar_names[i]}'
    )

ax5.set_xlabel("Topic Number")
ax5.set_ylabel("Number of Tweeets", color='black')
ax5.tick_params(axis='y', labelcolor='black')
ax5.set_xticks(x + bar_width * (account_type_n_bars - 1) / 2)
ax5.set_xticklabels([f'{i}' for i in categories])

# --- Lines on right y-axis ---

ax6 = ax5.twinx()
for i in range(account_type_n_lines):
    ax6.plot(
        x + bar_width,   # center of group
        account_type_line_data[i],
        marker='o',
        linewidth=2,
        label=f'{account_type_line_names[i]}'
    )

ax6.set_ylabel("Percentage of Subpopulation", color='black')
ax6.tick_params(axis='y', labelcolor='black')

# --- Combine legends ---
account_type_bars_handles, account_type_bars_labels = ax5.get_legend_handles_labels()
account_type_lines_handles, account_type_lines_labels = ax6.get_legend_handles_labels()
ax5.legend(account_type_bars_handles + account_type_lines_handles, account_type_bars_labels + account_type_lines_labels, loc='upper right')
ax5.set_title("Topic Prevalence by Account Type")

### Interests subplot
interests_group_width = 0.8
interests_bar_width = interests_group_width / interests_n_bars
offsets = np.arange(interests_n_bars) * interests_bar_width - (interests_group_width - interests_bar_width) / 2
for i in range(interests_n_bars):
    ax7.bar(
        #x + i * bar_width,
        x + offsets[i],
        interests_bar_data[i],
        width=interests_bar_width,
        facecolor="none",        # transparent fill
        edgecolor=colors[i],     # colored border
        hatch=hatches[i],        # hatch pattern
        label=f'{interests_bar_names[i]}'
    )

ax7.set_xlabel("Topic Number")
ax7.set_ylabel("Number of Tweeets", color='black')
ax7.tick_params(axis='y', labelcolor='black')
#ax7.set_xticks(x + bar_width * (n_bars - 1) / 2)
ax7.set_xticks(x)
ax7.set_xticklabels([f'{i}' for i in categories])

# --- Lines on right y-axis ---

ax8 = ax7.twinx()
for i in range(interests_n_lines):
    ax8.plot(
        #x + bar_width,
        x,
        interests_line_data[i],
        marker='o',
        linewidth=2,
        label=f'{interests_line_names[i]}'
    )

ax8.set_ylabel("Percentage of Subpopulation", color='black')
ax8.tick_params(axis='y', labelcolor='black')

# --- Combine legends ---
interests_bars_handles, interests_bars_labels = ax7.get_legend_handles_labels()
interests_lines_handles, interests_lines_labels = ax8.get_legend_handles_labels()
#ax7.legend(interests_bars_handles + interests_lines_handles, interests_bars_labels + interests_lines_labels, loc='upper right')
ax7.legend(interests_bars_handles, interests_bars_labels, loc='upper right')
ax7.set_title("Topic Prevalence by Interest")

### Region subplot
region_group_width = 0.8
region_bar_width = region_group_width / region_n_bars
offsets = np.arange(region_n_bars) * region_bar_width - (region_group_width - region_bar_width) / 2
for i in range(region_n_bars):
    ax9.bar(
        #x + i * bar_width,
        x + offsets[i],
        region_bar_data[i],
        width=region_bar_width,
        facecolor="none",        # transparent fill
        edgecolor=colors[i],     # colored border
        hatch=hatches[i],        # hatch pattern
        label=f'{region_bar_names[i]}'
    )

ax9.set_xlabel("Topic Number")
ax9.set_ylabel("Number of Tweeets", color='black')
ax9.tick_params(axis='y', labelcolor='black')
#ax7.set_xticks(x + bar_width * (n_bars - 1) / 2)
ax9.set_xticks(x)
ax9.set_xticklabels([f'{i}' for i in categories])

# --- Lines on right y-axis ---

ax10 = ax9.twinx()
for i in range(region_n_lines):
    ax10.plot(
        #x + bar_width,
        x,
        region_line_data[i],
        marker='o',
        linewidth=2,
        label=f'{region_line_names[i]}'
    )

ax10.set_ylabel("Percentage of Subpopulation", color='black')
ax10.tick_params(axis='y', labelcolor='black')

# --- Combine legends ---
region_bars_handles, region_bars_labels = ax9.get_legend_handles_labels()
region_lines_handles, region_lines_labels = ax10.get_legend_handles_labels()
#ax7.legend(interests_bars_handles + interests_lines_handles, interests_bars_labels + interests_lines_labels, loc='upper right')
ax9.legend(region_bars_handles, region_bars_labels, loc='upper right')
ax9.set_title("Topic Prevalence by Region")

plt.tight_layout()
plt.show()