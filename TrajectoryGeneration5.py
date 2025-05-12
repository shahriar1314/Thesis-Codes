import numpy as np
import matplotlib.pyplot as plt


def generate_trajectory(k,
                        p0,
                        ptd,
                        initial_velocity,
                        kd_alpha=0.5,
                        dist_approach=4.0,
                        exit_angle_deg=30.0,
                        td=5.0,
                        n1=500,
                        n2a=100,
                        n2b=200):
    """
    Generate the full UAV trajectory (perch, forward approach, fly-away) for a given k.
    """
    # Phase 1: Perching
    t1 = np.linspace(0, td, n1)
    tau_0 = np.linalg.norm(ptd - p0) / initial_velocity
    d_t = initial_velocity * tau_0 * (1 - t1 / td) ** (1 / k)
    alpha_0 = np.pi / 4
    alpha_t = alpha_0 * (d_t / d_t[0]) ** (1 / kd_alpha)

    p1 = np.zeros((n1, 3))
    for i in range(n1):
        M1 = (np.sin(alpha_t[0]) - np.sin(alpha_t[i])) / np.sin(alpha_t[0])
        M2 = np.sin(alpha_t[i]) / np.sin(alpha_t[0])
        M3 = np.array([0.0, 0.0, d_t[i] * np.sin(alpha_t[i])])
        p1[i] = M1 * ptd + M2 * p0 + M3

    # Shared horizontal vector
    h = ptd - p0
    h[2] = 0
    h_unit = h / np.linalg.norm(h)

    # Phase 2a: Forward Approach
    t2a = np.linspace(0, dist_approach / initial_velocity, n2a)
    p2a = np.array([ptd + h_unit * initial_velocity * ti for ti in t2a])

    # Phase 2b: Fly‐Away
    phi_exit = np.deg2rad(exit_angle_deg)
    u_exit = np.array([
        np.cos(phi_exit) * h_unit[0],
        np.cos(phi_exit) * h_unit[1],
        np.sin(phi_exit)
    ])
    ptd2 = ptd + h_unit * dist_approach
    t2b = np.linspace(0, 5.0, n2b)
    p2b = np.array([ptd2 + initial_velocity * u_exit * ti for ti in t2b])

    # Concatenate time and positions
    t_full = np.concatenate([t1,
                             td + t2a,
                             td + t2a[-1] + t2b])
    p_full = np.vstack([p1, p2a, p2b])
    return t_full, p_full


if __name__ == "__main__":
    # === Parameters ===
    p0 = np.array([0.0, 0.0, 0.0])
    ptd = np.array([13.0, 14.0, 0.0])
    initial_velocity = 4.5
    k = 0.6
    k_set = np.linspace(0, 1.2, 5)

    # === Figure for velocity & acceleration plots ===
    fig_va, (ax_v, ax_a) = plt.subplots(1, 2, figsize=(14, 6))
    ax_v.set_title("Velocity Profile")
    ax_a.set_title("Acceleration Profile")
    ax_v.set_xlabel("Time [s]")
    ax_a.set_xlabel("Time [s]")
    ax_v.set_ylabel("Speed [m/s]")
    ax_a.set_ylabel("Acceleration [m/s²]")

    # === Figure for 3D trajectory plot ===
    fig = plt.figure(figsize=(10, 7))
    ax_traj = fig.add_subplot(111, projection='3d')

    # === Loop over kd_alpha values ===
    for k in k_set:
        t, p = generate_trajectory(k, p0, ptd, initial_velocity, k)

        # 3D Trajectory
        ax_traj.plot(p[:, 0], p[:, 1], p[:, 2], label=f"k={k:.1f}, kd_alpha={k:.2f}", linewidth=2)

        # Velocity and acceleration
        dt = np.gradient(t)
        velocities = np.gradient(p, axis=0) / dt[:, None]
        accelerations = np.gradient(velocities, axis=0) / dt[:, None]
        speed = np.linalg.norm(velocities, axis=1)
        accel_magnitude = np.linalg.norm(accelerations, axis=1)

        # Add to combined plots
        ax_v.plot(t, speed, label=f"kd_alpha={k:.2f}")
        ax_a.plot(t, accel_magnitude, label=f"kd_alpha={k:.2f}")

    # === Finalize velocity & acceleration plots ===
    ax_v.legend()
    ax_a.legend()
    # Set axis ranges (adjust these values as needed)
    ax_v.set_xlim([0, 15])          # Time range for velocity plot
    ax_v.set_ylim([0, 6])           # Speed range

    ax_a.set_xlim([0, 15])          # Time range for acceleration plot
    ax_a.set_ylim([0, 10])          # Acceleration magnitude range
    fig_va.tight_layout()
    fig_va.show()

    # === Annotate key points in trajectory plot ===
    h = ptd - p0
    h[2] = 0
    h_unit = h / np.linalg.norm(h)
    ptd2 = ptd + h_unit * 4.0

    ax_traj.scatter(*p0, color='red', label='Start (p0)', s=50)
    ax_traj.scatter(*ptd, color='green', label='Perch Point', s=50)
    ax_traj.scatter(*ptd2, color='orange', label='Post-Approach Point', s=50)

    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_zlabel('Z (m)')
    ax_traj.set_title('UAV Perch & Fly-Away Trajectories for Multiple kd_alpha Values')
    ax_traj.legend()
    plt.tight_layout()
    plt.show()
