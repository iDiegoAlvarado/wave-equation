"""
    ------------------------------
    ---- Wave Equation in 1-D ----
    ------------------------------

    The wave equation is a second-order partial differential equation
    describing waves, including traveling and standing waves.

    In this code, the 1-D wave equation (vibrating string) under 
    different initial conditions and constraints is resolved using 
    the so called discrete method: finite differences. 

    Equation:    ü = c² ∇²u


    author: @idiegoalvarado
    repo:   github.com/idiegoalvarado/wave-equation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




"""
    Define physical parameters.
"""

L = 1.0            # Length of the string  [m]
T = 2.0            # Total time            [s]
c = 1.0            # Propagation velocity  [m/s]

# Discretizing space-time
nx, nt = 51, 102
dx, dt = L/(nx-1), T/(nt-1)
x, t   = np.linspace(0, L, nx), np.linspace(0, T, nt)




"""
    Define initial confitions.

    The initial conditions (at t = 0) constrain the evolution of 
    system's motion. In this case we will need to to define two
    initial conditions:

        u (x, t=0) = f(x)
        u'(x, t=0) = v(x)
"""

# u(x, 0) = f(x)
def normal(x, k):
    # Normal vibration mode
    return 1.0 * np.sin(k * np.pi * x / L)

def plucked(x, xi, h=1.0):
    # Plucked string
    if (x < xi):
        return h * x / xi
    else:
        return h * (L - x) / (L - xi)
    
def f(x):
    # Any other initial shape
    return normal(x, 5)* np.exp(-x**2/L)


# u'(x, 0) = v(x)
def v(x):
    # Initial velocity distribution
    d, xi, h = L/5, L/5, 1.0
    v_0 = - (h / 10) / dt
    if (abs(x - xi) < d/2):
        return v_0 * np.cos(np.pi * (x - xi) / d)
    else:
        return 0.0

def v_zero(x):
    # Release from the rest
    return x * 0.0




"""
    Initialize solution matrix.
"""

u = np.zeros((nt,nx))

u_0 = np.vectorize(f)(x)
v_0 = np.vectorize(v_zero)(x)*0

u[0] = u_0
u[1] = u[0] + v_0 * dt


"""
    Solve using the finite differences method.

    Consider u(x,t), a system discretized as follows:
    
        (space): x = [x_1, x_2, x_3, ..., x_N], i -> x_i
        (time):  t = [t_1, t_2, t_3, ..., t_M], n -> t_n

    Using Taylor series we can approximate the discrete form of the
    differential factors composing the PDE. At the end, after making
    a couple of transformation, it comes the following formula:

    u(n+1,i) = 2u(n,i)- u(n-1,i) + r [u(n,i+1) - 2u(n,i) + u(n,i-1)]

    where r = c² (dt/dx)².
"""

r = (c * dt / dx) ** 2

for n in range(1, nt-1):
    for i in range (1, nx-1):
        u[n+1,i] = 2 * u[n,i] - u[n-1,i] + r * (u[n,i+1] - 2 * u[n,i] + u[n,i-1])




"""
    Animate solution.
"""

# Initialize figures
plt.rcParams["figure.figsize"] = [4.50*2, 3.0]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()
fig.suptitle('Vibrating string')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')

min_y = np.min(u) * 2
max_y = np.max(u) * 2

plt.xlim([0, L])
plt.ylim([min_y, max_y])
line, = ax.plot(x, u[0], lw=2)

# Update figure
def animate(n):
    line.set_ydata(u[n])
    return line,

ani = FuncAnimation(fig, animate, frames=nt, interval=20)
plt.show()
