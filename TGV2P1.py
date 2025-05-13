'''
Implementation of Extended Tau Theory (ICRA 2017)

This module provides:
  - Two-stage trajectory generation for 1D non-zero contact velocity (Sec II.B, Eq.13)
  - Constant-tau-dot deceleration parameter selection under max acceleration (Sec II.B, Eqs.11-12)
  - Tau-coupling strategy for 3D motion (Sec II.C, Eqs.15-20)
  - Simulation utilities for gap, velocity, acceleration, and tau profiles
  - Plotting functions for visualization of 1D and 3D trajectories

References:
  Zhang et al., "Extended Tau Theory for Robot Motion Control", ICRA 2017
'''
import numpy as np
import matplotlib.pyplot as plt

def solve_optimal_k(X0, Xdot0, alpha, max_accel, tol=1e-6, max_iter=50):
    """
    Find the largest k in (0,1) satisfying the deceleration constraint under two regimes:
      Stage1 (k<=0.5): a = (1-k)*Xdot0^2/X0
      Stage2 (k>0.5): a = (1-k)*(Xdot0^2/X0)*(alpha/Xdot0)^{(1-2k)/(1-k)}
    Uses robust bisection on the interval where the root exists.
    Returns k.
    """
    def decel_constraint(k):
        try:
            if k <= 0.5:
                a = (1-k)*(Xdot0**2)/X0
            else:
                expo = (1-2*k)/(1-k)
                log_term = expo * np.log(alpha/Xdot0)
                if log_term > 700:
                    term = np.inf
                elif log_term < -700:
                    term = 0.0
                else:
                    term = np.exp(log_term)
                a = (1-k)*(Xdot0**2)/X0 * term
            return a - max_accel
        except Exception:
            return np.inf

    eps = 1e-6
    f0 = decel_constraint(eps)
    f05 = decel_constraint(0.5)
    f1 = decel_constraint(1.0 - eps)
    # choose bracket
    if f05 <= 0 < f1:
        lo, hi = 0.5, 1.0 - eps
    elif f0 <= 0 < f05:
        lo, hi = eps, 0.5
    else:
        raise ValueError("No feasible k in (0,1) satisfies max_accel")

    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        if decel_constraint(mid) <= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo

def compute_two_stage_trajectory_1d(X0, Xdot0, alpha, max_accel, dt=0.01):
    """
    Compute two-stage 1D trajectory: deceleration with constant tau-dot until velocity=alpha,
    then constant velocity to close distance.
    Returns dict with time, X, Xdot, Xddot, tau_ref.
    """
    k = solve_optimal_k(X0, Xdot0, alpha, max_accel)
    tau0 = X0/Xdot0
    factor = (alpha/Xdot0)**(k/(k-1))
    t1 = (factor - 1)*(X0/(k*Xdot0))
    X_t1 = X0 * factor**(1/k)
    Xdot_t1 = Xdot0 * factor**(1/k - 1)
    t2 = X_t1 / (-alpha)
    tf = t1 + t2
    times = np.arange(0, tf + dt, dt)

    X = np.zeros_like(times)
    Xdot = np.zeros_like(times)
    Xddot = np.zeros_like(times)
    tau_ref = np.zeros_like(times)

    for i, t in enumerate(times):
        if t < t1:
            fac_t = 1 + k*(Xdot0/X0)*t
            X[i] = X0 * fac_t**(1/k)
            Xdot[i] = Xdot0 * fac_t**(1/k - 1)
            Xddot[i] = (Xdot0**2/X0)*(1-k)*fac_t**(1/k - 2)
            tau_ref[i] = k*t + tau0
        else:
            dt2 = t - t1
            X[i] = X_t1 + alpha*dt2
            Xdot[i] = alpha
            Xddot[i] = 0.0
            tau_ref[i] = (X_t1/alpha) + dt2

    return {
        'k': k,
        't1': t1,
        't2': t2,
        'times': times,
        'X': X,
        'Xdot': Xdot,
        'Xddot': Xddot,
        'tau_ref': tau_ref
    }

def compute_tau_coupling_constants(X_t1, Xdot_t1, h):
    """
    Compute coupling constant b to synchronize Y/Z closure with X stage1 duration:
      b = (1-h)*Xdot_t1/X_t1
    """
    return (1 - h) * Xdot_t1 / X_t1

def generate_3d_coupled_trajectory(traj1d, Y0, Ydot0, Z0, Zdot0, h_y, h_z, dt=0.01):
    """
    Extend 1D trajectory to coupled 3D reference using τ-coupling:
      τ_y = τ_x/(b_y*τ_x + h_y), similarly for τ_z.
    Returns time series for X,Y,Z and τs.
    """
    times = traj1d['times']
    tau_x = traj1d['tau_ref']
    idx1 = np.argmin(np.abs(times - traj1d['t1']))
    b_y = compute_tau_coupling_constants(traj1d['X'][idx1], traj1d['Xdot'][idx1], h_y)
    b_z = compute_tau_coupling_constants(traj1d['X'][idx1], traj1d['Xdot'][idx1], h_z)

    Y = np.zeros_like(times)
    Z = np.zeros_like(times)
    tau_y = np.zeros_like(times)
    tau_z = np.zeros_like(times)

    for i, tx in enumerate(tau_x):
        if times[i] <= traj1d['t1']:
            ty = tx / (b_y * tx + h_y)
            tz = tx / (b_z * tx + h_z)
            tau_y[i], tau_z[i] = ty, tz
            Y[i] = Ydot0 * ty
            Z[i] = Zdot0 * tz
        else:
            dt2 = times[i] - traj1d['t1']
            Y[i] = Ydot0 * tau_y[idx1] - Ydot0 * dt2
            Z[i] = Zdot0 * tau_z[idx1] - Zdot0 * dt2
            tau_y[i] = Y[i] / Ydot0
            tau_z[i] = Z[i] / Zdot0

    return {
        'times': times,
        'Y': Y,
        'Z': Z,
        'tau_y': tau_y,
        'tau_z': tau_z,
        'b_y': b_y,
        'b_z': b_z
    }

def plot_1d_trajectory(traj):
    """
    Plot X, Xdot, Xddot, and tau_ref over time for 1D trajectory.
    """
    t = traj['times']
    plt.figure(figsize=(10, 8))
    plt.subplot(4,1,1)
    plt.plot(t, traj['X'])
    plt.ylabel('X')
    plt.subplot(4,1,2)
    plt.plot(t, traj['Xdot'])
    plt.ylabel('Xdot')
    plt.subplot(4,1,3)
    plt.plot(t, traj['Xddot'])
    plt.ylabel('Xddot')
    plt.subplot(4,1,4)
    plt.plot(t, traj['tau_ref'])
    plt.ylabel('tau_ref')
    plt.xlabel('time')
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory(traj1d, traj3d):
    """
    Plot tau_x, tau_y, tau_z and gaps Y,Z over time for 3D coupled trajectory.
    """
    t = traj1d['times']
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, traj1d['tau_ref'], label='tau_x')
    plt.plot(t, traj3d['tau_y'], label='tau_y')
    plt.plot(t, traj3d['tau_z'], label='tau_z')
    plt.ylabel('tau')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t, traj3d['Y'], label='Y')
    plt.plot(t, traj3d['Z'], label='Z')
    plt.ylabel('gap')
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage (for testing and visualization)
if __name__ == '__main__':
    # 1D setup
    X0, Xdot0, alpha, max_accel = 10.0, -3.0, -1.0, 0.7
    traj = compute_two_stage_trajectory_1d(X0, Xdot0, alpha, max_accel)
    plot_1d_trajectory(traj)

    # 3D extension
    Y0, Ydot0, Z0, Zdot0 = 8.0, -2.0, 6.0, -1.7
    h_y, h_z = 0.3, 0.2
    traj3d = generate_3d_coupled_trajectory(traj, Y0, Ydot0, Z0, Zdot0, h_y, h_z)
    plot_3d_trajectory(traj, traj3d)
