
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def equal_aspect_3d(ax, X, Y, Z):
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    max_range = max(xr, yr, zr, 1e-9)
    xmid = (xmax + xmin) / 2.0
    ymid = (ymax + ymin) / 2.0
    zmid = (zmax + zmin) / 2.0
    ax.set_xlim(xmid - max_range/2, xmid + max_range/2)
    ax.set_ylim(ymid - max_range/2, ymid + max_range/2)
    ax.set_zlim(zmid - max_range/2, zmid + max_range/2)

def pick_offset(df, mode="fly_to_start_target"):
    if isinstance(mode, (tuple, list, np.ndarray)) and len(mode) == 3:
        return np.array(mode, dtype=float)

    if mode == "fly_to_start_target":
        row = df[df['phase'] == 'FLY_TO_START'].head(1)
        if not row.empty:
            return row[['target_x','target_y','target_z']].values[0]
    if mode == "fly_to_start_drone":
        row = df[df['phase'] == 'FLY_TO_START'].head(1)
        if not row.empty:
            return row[['drone_x','drone_y','drone_z']].values[0]
    if mode == "first_row_target":
        row = df.head(1)
        return row[['target_x','target_y','target_z']].values[0]
    if mode == "first_row_drone":
        row = df.head(1)
        return row[['drone_x','drone_y','drone_z']].values[0]

    return np.array([df[['target_x','drone_x']].min().min(),
                     df[['target_y','drone_y']].min().min(),
                     df[['target_z','drone_z']].min().min()], dtype=float)

def main():
    parser = argparse.ArgumentParser(description="Plot dronetarget vs drone 3D paths from CSV.")
    parser.add_argument("csv_path", help="Path to CSV with columns: time,phase,target_x,target_y,target_z,drone_x,drone_y,drone_z")
    parser.add_argument("--offset-mode", default="fly_to_start_target",
                        help="Offset: fly_to_start_target|fly_to_start_drone|first_row_target|first_row_drone|x,y,z")
    parser.add_argument("--out", default=None, help="Output image path (.png). Default: alongside CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df['phase'] = df['phase'].astype(str).str.strip()

    offset_mode = args.offset_mode
    if isinstance(offset_mode, str) and ',' in offset_mode:
        parts = [float(p) for p in offset_mode.split(',')[:3]]
        offset = np.array(parts, dtype=float)
    else:
        offset = pick_offset(df, offset_mode)

    df['tX'] = df['target_x'] - offset[0]
    df['tY'] = df['target_z'] - offset[2]
    df['tZ'] = df['target_y'] - offset[1] + 8
    df['dX'] = df['drone_x']  - offset[0]
    df['dY'] = df['drone_z']  - offset[2]
    df['dZ'] = df['drone_y']  - offset[1] +8

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df['tX'].values, df['tY'].values, df['tZ'].values, linewidth=1.7, linestyle="--", label="Target path")
    ax.plot(df['dX'].values, df['dY'].values, df['dZ'].values, linewidth=2.0,  label="Drone path")

    ax.scatter(df['tX'].iloc[0], df['tY'].iloc[0], df['tZ'].iloc[0], s=50, marker="^", label="Target start")
    ax.scatter(df['tX'].iloc[-1], df['tY'].iloc[-1], df['tZ'].iloc[-1], s=50, marker="o", label="Target end")
    ax.scatter(df['dX'].iloc[0], df['dY'].iloc[0], df['dZ'].iloc[0], s=50, marker="^", label="Drone start")
    ax.scatter(df['dX'].iloc[-1], df['dY'].iloc[-1], df['dZ'].iloc[-1], s=50, marker="o", label="Drone end")
    

    phase_changes = df['phase'] != df['phase'].shift(1)
    indices = list(df.index[phase_changes])
    for idx in indices:
        px, py, pz = df.loc[idx, ['dX','dY','dZ']].values
        phase_name = df.loc[idx, 'phase']
        ax.scatter(px, py, pz, s=30, marker="s")
        # ax.text(px, py, pz, f" {phase_name}", fontsize=8)

    ax.set_title("Desired vs Actual Flight Path", fontsize=16)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=13, markerscale=1.5, handlelength=2.0, borderpad=1.1)
    equal_aspect_3d(ax, np.r_[df['tX'].values, df['dX'].values],
                       np.r_[df['tY'].values, df['dY'].values],
                       np.r_[df['tZ'].values, df['dZ'].values])
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        root, _ = os.path.splitext(args.csv_path)
        out_path = root + "_3dplot.png"
    #fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
