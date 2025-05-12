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

    Parameters:
        k (float): distance‐decay shape parameter
        p0 (ndarray): initial position [x, y, z]
        ptd (ndarray): perch target position [x, y, z]
        initial_velocity (float): flight speed (m/s)
        kd_alpha (float): angle‐decay shape parameter
        dist_approach (float): forward approach distance (m)
        exit_angle_deg (float): climb‐out angle (degrees)
        td (float): total perch duration (s)
        n1 (int): sample points for perching
        n2a (int): sample points for approach
        n2b (int): sample points for climb‐out

    Returns:
        t_full (ndarray): concatenated time vector
        p_full (ndarray): concatenated position array [n×3]
    """
    # Phase 1: Perching
    t1 = np.linspace(0, td, n1)
    tau_0 = np.linalg.norm(ptd - p0) / initial_velocity
    d_t = initial_velocity * tau_0 * (1 - t1/td)**(1/k)
    alpha_0 = np.pi / 4
    alpha_t = alpha_0 * (d_t / d_t[0])**(1/kd_alpha)

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
    ks = np.arange(0.1, 6.1, 1)

    # Plot setup
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Generate & plot for each k
    #k_set = np.linspace(0.1, 1.6, 5)
    #kd_alpha = 0.5
    kd_alpha_set = np.linspace(0,2,5)
    k = 0.6

    
    for kd_alpha in kd_alpha_set:
        t, p = generate_trajectory(k, p0, ptd, initial_velocity, kd_alpha)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=f"k={k:.1f}, kd_alpha={kd_alpha:.2f}", linewidth=2)   
        # ax.plot(p[:, 0], p[:, 1], p[:, 2], label=f"k={k:.1f}", linewidth=2)

    # Annotate key points
    h = ptd - p0; h[2] = 0; h_unit = h / np.linalg.norm(h)
    ptd2 = ptd + h_unit * 4.0
    ax.scatter(*p[0],   color='red',   label='Start (p0)',   s=50)
    ax.scatter(*ptd,  color='green', label='Perch Point',   s=50)
    ax.scatter(*ptd2, color='orange',label='Post-Approach Point', s=50)

    # Labels & legend
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UAV Perch & Fly-Away Trajectories for Multiple k Values')
    ax.legend()
    plt.tight_layout()
    plt.show()
