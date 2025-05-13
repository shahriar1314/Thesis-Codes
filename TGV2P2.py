import numpy as np
import matplotlib.pyplot as plt

"""
Module: tau_trajectory.py
Bio-inspired UAV perching trajectories based on Tau Theory
Formulas and cases refer to:
  Z. Zhang, P. Xie, O. Ma, "Bio-inspired trajectory generation for UAV perching", 2013. citeturn0file0

Six cases (two scenarios, three strategies each):

Scenario A: Perching from flight state (non-zero initial velocity)
  Case 1: Straight-line trajectory (eqs. 6,13,14) citeturn17file19
  Case 2: Pitch-angle coupling (eqs. 13,15,16) citeturn6file6
  Case 3: Pitch & yaw coupling (eqs. 13,15,17,18) citeturn5file5

Scenario B: Perching from hovering state (zero initial velocity)
  Case 4: Straight-line (intrinsic tau gravity, eqs. 11,12,14,19) citeturn11file11
  Case 5: Pitch-angle (eqs. 12,20,16) citeturn4file4
  Case 6: Pitch & yaw (eqs. 12,20,21,18) citeturn4file4
"""

def tau_action_gap_t_d(d0, v0, k):
    """
    Compute time-to-go t_d for closing a distance gap under constant tau rate:
    tau0 = -d0/v0,  t_d = -tau0/k = d0/(v0*k)
    """
    tau0 = -d0/ v0
    t_d = -tau0/ k
    return t_d


def case1_trajectory(p0, p_td, d0, v0, k, num_steps=100):
    """
    Case 1 (straight-line):
    d(t) = d0 * (1 - t/t_d)**(1/k)  (eq.13) 
    p(t) = p0 + (1 - d(t)/d0) * (p_td - p0)  (eq.14)
    """
    t_d = tau_action_gap_t_d(d0, v0, k)
    t = np.linspace(0, t_d, num_steps)
    d = d0 * np.power(1 - t/t_d, 1.0/k)
    # interpolate position along straight line
    direction = (p_td - p0)
    p = p0 + (1 - (d / d0))[:, None] * direction
    return t, p


def case2_trajectory(p0, p_td, d0, v0, k_d_alpha, alpha0, num_steps=100):
    """
    Case 2 (pitch-angle coupling):
    Use same d(t) as Case 1 with k = k_d_alpha.
    alpha(t) = alpha0 * (1 - t/t_d)**(1/k_d_alpha)  (eq.15)
    Return positions p(t) and pitch angles alpha(t).
    """
    t_d = tau_action_gap_t_d(d0, v0, k_d_alpha)
    t = np.linspace(0, t_d, num_steps)
    d = d0 * np.power(1 - t/t_d, 1.0/ k_d_alpha)
    direction = (p_td - p0)
    p = p0 + (1 - (d / d0))[:, None] * direction
    alpha = alpha0 * np.power(1 - t/t_d, 1.0/ k_d_alpha)
    # alpha = alpha0 * np.power(t/t_d, 1.0/ k_d_alpha)
    return t, p, alpha


def case3_trajectory(p0, p_td, d0, v0, k_d_alpha, alpha0, k_d_beta, beta0, num_steps=100):
    """
    Case 3 (pitch & yaw coupling):
    Same d(t) with k = k_d_alpha = k_d_beta assumed.
    alpha(t) and beta(t) both follow eq.17 with respective k.
    Returns p(t), alpha(t), beta(t).
    """
    # reuse k_d_alpha for distance closing
    t_d = tau_action_gap_t_d(d0, v0, k_d_alpha)
    t = np.linspace(0, t_d, num_steps)
    d = d0 * np.power(1 - t/t_d, 1.0/ k_d_alpha)
    direction = (p_td - p0)
    p = p0 + (1 - (d / d0))[:, None] * direction
    alpha = alpha0 * np.power(1 - t/t_d, 1.0/ k_d_alpha)
    beta = beta0 * np.power(1 - t/t_d, 1.0/ k_d_beta)
    return t, p, alpha, beta

# Intrinsic tau gravity guidance helpers

def case4_trajectory(p0, p_td, d0, k_x_g, num_steps=100):
    """
    Case 4 (hover to target straight-line, intrinsic tau gravity):
    d(t) = d0 * (1 - (t/t_d)**2)**(1/(2*k_x_g))  (from eqs.11,12,19)
    p(t) via eq.14
    """
    # duration t_d must be provided or solved; assume t_d input via k_x_g and free-fall
    # here we assume t_d = initial guess
    # For zero initial speed, t_d chosen as: t_d = np.sqrt(d0*2/k_x_g)
    t_d = np.sqrt(d0 * 2.0 / k_x_g)
    t = np.linspace(0, t_d, num_steps)
    d = d0 * np.power(1 - (t/t_d)**2, 1.0/(2.0 * k_x_g))
    direction = (p_td - p0)
    p = p0 + (1 - (d / d0))[:, None] * direction
    return t, p


def case5_trajectory(p0, p_td, d0, k_alpha_g, alpha0, num_steps=100):
    """
    Case 5 (hover + pitch coupling):
    d(t) as Case 4; pitch alpha(t) = alpha0 * (1 - (t/t_d)**2)**(1/(2*k_alpha_g))  (eq.20)
    """
    t, p = case4_trajectory(p0, p_td, d0, k_alpha_g, num_steps)
    # extract t_d from t array
    t_d = t[-1]
    alpha = alpha0 * np.power(1 - (t/t_d)**2, 1.0/(2.0 * k_alpha_g))
    return t, p, alpha


def case6_trajectory(p0, p_td, d0, k_alpha_g, alpha0, k_beta_g, beta0, num_steps=100):
    """
    Case 6 (hover + pitch & yaw coupling):
    d(t) as Case 4; alpha(t) as Case 5; beta(t) similarly from eq.21
    """
    t, p = case4_trajectory(p0, p_td, d0, k_alpha_g, num_steps)
    t_d = t[-1]
    alpha = alpha0 * np.power(1 - (t/t_d)**2, 1.0/(2.0 * k_alpha_g))
    beta = beta0 * np.power(1 - (t/t_d)**2, 1.0/(2.0 * k_beta_g))
    return t, p, alpha, beta

# # Example usage (uncomment to simulate):
# if __name__ == "__main__":
#     p0 = np.array([5.0, 0.0, 0.0])
#     p_td = np.array([3.0, 4.0, 0.0])
#     d0 = np.linalg.norm(p_td - p0)
#     t, traj = case1_trajectory(p0, p_td, d0, v0=10.0, k=0.4)
#     print(traj)

# --- Example usage for Case 1 + plotting ---
if __name__ == "__main__":
    # Initial and target positions
    p0   = np.array([ 0.0,  0.0, 0.0])
    p_td = np.array([10.0,  5.0, 2.0])
    d0   = np.linalg.norm(p_td - p0)

    # Tau parameters
    v0 = 5.0     # initial closing speed
    k  = 0.5     # tau braking constant

    # Generate trajectory
    t, p = case1_trajectory(p0, p_td, d0, v0, k)

    # Compute distance, speed, acceleration
    distance     = np.linalg.norm(p_td - p, axis=1)
    vel_vec      = np.gradient(p, t, axis=0)
    speed        = np.linalg.norm(vel_vec, axis=1)
    acc_vec      = np.gradient(vel_vec, t, axis=0)
    acceleration = np.linalg.norm(acc_vec, axis=1)

    # Plot Distance vs Time
    plt.figure()
    plt.plot(t, distance)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance to Target')
    plt.title('Case 1: Distance vs Time')

    # Plot Speed vs Time
    plt.figure()
    plt.plot(t, speed)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.title('Case 1: Speed vs Time')

    # Plot Acceleration vs Time
    plt.figure()
    plt.plot(t, acceleration)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Case 1: Acceleration vs Time')

    # Plot 3D Trajectory Path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:,0], p[:,1], p[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Case 1: 3D Trajectory Path')

    plt.show()


# # --- Example usage for Case 2 ---
# if __name__ == "__main__":
#     # Initial and target positions
#     p0 = np.array([ 0.0,  0.0, 0.0])
#     p_td = np.array([10.0,  5.0, 2.0])
#     d0 = np.linalg.norm(p_td - p0)
    
#     # Tau parameters
#     v0          =  5.0               # initial closing speed
#     k_d_alpha   =  0.7             # tau coupling constant
#     alpha0      = np.deg2rad(45)     # initial pitch (radians)
    
#     # Generate trajectory + pitch profile
#     t, p, alpha = case2_trajectory(p0, p_td, d0, v0, k_d_alpha, alpha0)

#     # Compute distance, speed, acceleration
#     distance     = np.linalg.norm(p_td - p, axis=1)
#     vel_vec      = np.gradient(p, t, axis=0)
#     speed        = np.linalg.norm(vel_vec, axis=1)
#     acc_vec      = np.gradient(vel_vec, t, axis=0)
#     acceleration = np.linalg.norm(acc_vec, axis=1)

#     # --- Plotting ---
#     plt.figure()
#     plt.plot(t, distance)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Distance to Target')
#     plt.title('Distance vs Time')

#     plt.figure()
#     plt.plot(t, speed)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Speed')
#     plt.title('Speed vs Time')

#     plt.figure()
#     plt.plot(t, acceleration)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Acceleration')
#     plt.title('Acceleration vs Time')

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(p[:,0], p[:,1], p[:,2])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Trajectory Path')

#     plt.show()






