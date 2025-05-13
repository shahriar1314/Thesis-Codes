import numpy as np
import matplotlib.pyplot as plt

def tau_action_gap_t_d(d0, v0, k):
    """
    Compute time-to-go t_d for closing a distance gap under constant tau rate.
    """
    tau0 = -d0 / v0
    t_d = -tau0 / k
    return t_d

def case1_trajectory(p0, p_td, d0, v0, k, num_steps=100):
    """
    Case 1 (straight-line tau-based trajectory).
    """
    t_d = tau_action_gap_t_d(d0, v0, k)
    t = np.linspace(0, t_d, num_steps)
    d = d0 * np.power(1 - t / t_d, 1.0 / k)
    direction = (p_td - p0)
    p = p0 + (1 - (d / d0))[:, None] * direction
    return t, p

if __name__ == "__main__":
    # Initial and target positions
    p0 = np.array([0.0, 0.0, 0.0])
    p_td = np.array([10.0, 5.0, 2.0])
    d0 = np.linalg.norm(p_td - p0)
    v0 = 5.0

    k_values = [0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5]

    # Prepare subplots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')

    for k in k_values:
        t, p = case1_trajectory(p0, p_td, d0, v0, k)
        distance = np.linalg.norm(p_td - p, axis=1)
        vel_vec = np.gradient(p, t, axis=0)
        speed = np.linalg.norm(vel_vec, axis=1)
        acc_vec = np.gradient(vel_vec, t, axis=0)
        acceleration = np.linalg.norm(acc_vec, axis=1)

        label = f'k={k}'

        ax1.plot(t, distance, label=label)
        ax2.plot(t, speed, label=label)
        ax3.plot(t, acceleration, label=label)
        ax4.plot(p[:,0], p[:,1], p[:,2], label=label)

    ax1.set_title('Distance vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance')
    ax1.legend()
    ax1.grid(Trues)

    ax2.set_title('Speed vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Acceleration vs Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('3D Trajectories for Various k')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    ax4.grid(True)

    plt.show()
