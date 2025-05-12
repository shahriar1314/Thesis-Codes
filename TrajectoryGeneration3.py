import numpy as np
import matplotlib.pyplot as plt

# Parameters
V0 = 4.5                # initial velocity [m/s]
td = 5.0                # total duration [s]
p0 = 20                 # initial horizontal position [m]
k_values = [0.3, 0.6, 0.9]  # representative values (0<k<0.5 and 0.5<k<1)
kd_alpha = 0.4          # angle decay parameter (given)
n = 500                 # number of points

# Time vector
t = np.linspace(0, td, n)
dt = t[1] - t[0]

plt.figure(figsize=(15, 10))

for k in k_values:
    # distance decay
    d = p0 * (1 - t/td)**(1/k)

    # angle decay
    alpha0 = np.pi / 4
    alpha = alpha0 * (d / d[0])**(1/kd_alpha)

    # vertical and horizontal positions
    z = d * np.sin(alpha)
    x = d * np.cos(alpha)

    # velocity (first derivative)
    velocity = np.gradient(d, dt)

    # acceleration (second derivative)
    acceleration = np.gradient(velocity, dt)

    # Plotting trajectories
    plt.subplot(2, 2, 1)
    plt.plot(x, z, label=f'k={k}')
    plt.xlabel('Horizontal Position [m]')
    plt.ylabel('Vertical Position [m]')
    plt.title('Trajectory shape')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, velocity, label=f'k={k}')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity vs Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, acceleration, label=f'k={k}')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Acceleration vs Time')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, d, label=f'k={k}')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.title('Distance vs Time')
    plt.legend()

plt.tight_layout()
plt.show()