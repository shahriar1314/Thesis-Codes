import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("dronepath.csv")

# Extract time and coordinates
time = df["time"].values
target_x = df["target_x"].values
target_y = df["target_y"].values
target_z = df["target_z"].values
drone_x = df["drone_x"].values
drone_y = df["drone_y"].values
drone_z = df["drone_z"].values

# Prepare figure
plt.rcParams.update({"font.size": 13})  # Change 14 to your desired size
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Function to set axis limits dynamically
def set_axis_limits(ax, data1, data2):
    min_val = min(data1.min(), data2.min()) - 2
    max_val = max(data1.max(), data2.max()) + 3
    ax.set_ylim(min_val, max_val)

# X vs time
axes[0].plot(time, target_z, label="Target X", linestyle="--", color="#D02828FF")
axes[0].plot(time, drone_z, label="Drone X", color="#023743FF")
set_axis_limits(axes[0], target_z, drone_z)
axes[0].set_ylabel("X Coordinates(m)")
axes[0].legend()

# Z vs time (Unity's Y)
axes[1].plot(time, target_x, label="Target Y", linestyle="--", color="#D02828FF")
axes[1].plot(time, drone_x, label="Drone Y", color="#023743FF")
set_axis_limits(axes[1], target_x, drone_x)
axes[1].set_ylabel("Y Coordinates(m)")
axes[1].legend()

# Y vs time (Unity's Z)
axes[2].plot(time, target_y, label="Target Z", linestyle="--", color="#D02828FF")
axes[2].plot(time, drone_y, label="Drone Z", color="#023743FF")
set_axis_limits(axes[2], target_y, drone_y)
axes[2].set_ylabel("Height (m)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

for ax in axes:
    ax.grid(True)

#FED789FF, #023743FF, #72874EFF, #476F84FF, #A4BED5FF, #453947FF


plt.tight_layout()
fig.savefig("trajectory_xyz_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()
