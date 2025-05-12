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

    h = ptd - p0
    h[2] = 0
    h_unit = h / np.linalg.norm(h)

    t2a = np.linspace(0, dist_approach / initial_velocity, n2a)
    p2a = np.array([ptd + h_unit * initial_velocity * ti for ti in t2a])

    phi_exit = np.deg2rad(exit_angle_deg)
    u_exit = np.array([
        np.cos(phi_exit) * h_unit[0],
        np.cos(phi_exit) * h_unit[1],
        np.sin(phi_exit)
    ])
    ptd2 = ptd + h_unit * dist_approach
    t2b = np.linspace(0, 5.0, n2b)
    p2b = np.array([ptd2 + initial_velocity * u_exit * ti for ti in t2b])

    t_full = np.concatenate([t1,
                             td + t2a,
                             td + t2a[-1] + t2b])
    p_full = np.vstack([p1, p2a, p2b])
    return t_full, p_full


def compute_velocity_acceleration(t, p):
    dt = np.gradient(t)
    velocities = np.gradient(p, axis=0) / dt[:, None]
    accelerations = np.gradient(velocities, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocities, axis=1)
    accel_magnitude = np.linalg.norm(accelerations, axis=1)
    return speed, accel_magnitude


if __name__ == "__main__":
    p0 = np.array([0.0, 0.0, 0.0])
    ptd = np.array([13.0, 14.0, 0.0])
    initial_velocity = 4.5
    k_set = np.linspace(0.1, 1.6, 5)
    kd_alpha = 0.5

    # === Plot 3D Trajectories ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for k in k_set:
        t, p = generate_trajectory(k, p0, ptd, initial_velocity, kd_alpha)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=f"k={k:.2f}", linewidth=2)

    h = ptd - p0
    h[2] = 0
    h_unit = h / np.linalg.norm(h)
    ptd2 = ptd + h_unit * 4.0
    ax.scatter(*p[0], color='red', label='Start (p0)', s=50)
    ax.scatter(*ptd, color='green', label='Perch Point', s=50)
    ax.scatter(*ptd2, color='orange', label='Post-Approach Point', s=50)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UAV Perch & Fly-Away Trajectories')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # === Combined Velocity & Acceleration Plots ===
    fig_v, (ax_v, ax_a) = plt.subplots(1, 2, figsize=(14, 6))

    for k in k_set:
        t, p = generate_trajectory(k, p0, ptd, initial_velocity, kd_alpha)
        speed, accel = compute_velocity_acceleration(t, p)
        ax_v.plot(t, speed, label=f'k={k:.2f}')
        ax_a.plot(t, accel, label=f'k={k:.2f}')

    ax_v.set_xlabel("Time [s]")
    ax_v.set_ylabel("Speed [m/s]")
    ax_v.set_title("Velocity Profile")
    ax_v.legend()
    ax_v.grid(True)

    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("Acceleration [m/s²]")
    ax_a.set_title("Acceleration Profile")
    ax_a.legend()
    ax_a.grid(True)

    # Optional: set fixed axis limits
    ax_v.set_ylim([0, 6])
    ax_a.set_ylim([0, 10])

    plt.tight_layout()
    plt.show()
