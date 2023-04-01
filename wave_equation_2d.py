"""
    ------------------------------
    ---- Wave Equation in 2-D ----
    ------------------------------

    The wave equation is a second-order partial differential equation
    describing waves, including traveling and standing waves.

    In this code, the 1-2 wave equation (vibrating membrene) under 
    different initial conditions and constraints is resolved using 
    the so called discrete method: finite differences. 

    Equation:    ü = c² ∇²u


    author: @idiegoalvarado
    github.com/iDiegoAlvarado/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




"""
    Define physical parameters.
"""

Lx, Ly = 1.0, 1.0     # Rectangle membrene's sides [m]
T = 2.0               # Total time                 [s]
c = 1.0               # Propagation velocity       [m/s]

# Discretizing space-time
nx, ny, nt = 51, 51, 199 * 1
dx, dy, dt = Lx/(nx-1), Ly/(ny-1), T/(nt-1)

x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny) 
t    = np.linspace(0, T, nt)




"""
    Define initial confitions.

    The initial conditions (at t = 0) constrain the evolution of 
    system's motion. In this case we will need to to define two
    initial conditions:

        u (x, y; t=0) = f(x, y)
        u'(x, y; t=0) = v(x, y)
"""

# u(x, y; 0) = f(x, y)
def normal(x, y, k_x, k_y):
    # Normal vibration mode
    return np.sin(k_x * np.pi * x / Lx) * np.sin(k_y * np.pi * y / Ly)
    
def f(x, y):
    # Any other initial shape
    return normal(x, y, 5, 5) * np.exp(- (x + y) ** 2 / Lx)


# u'(x, 0) = v(x)
def v(x, y, xi_x, xi_y):
    # Initial velocity distribution
    d, h = Lx / 5, 1.0  
    v_0 = (h/5) / dt
    if (abs(x-xi_x) < d/2) and (abs(y-xi_y) < d/2):
        return v_0 * np.cos(np.pi * (x-xi_x) / d) * np.cos(np.pi * (y-xi_y) / d)
    else:
        return 0.0

def v_zero(x, y):
    # Release from the rest
    return x * y * 0.0




"""
    Initialize solution matrix.
"""

u = np.zeros((nt,nx,ny))
X, Y = np.meshgrid(x, y)

u_0 = np.vectorize(normal)(X, Y, 1, 5) * 0 + \
       np.vectorize(normal)(X, Y, 5, 1) * 0

v_0 = np.vectorize(v)(X, Y, Lx/5, Ly/5) + np.vectorize(v)(X, Y, 4*Lx/5, 4*Ly/5) + \
       np.vectorize(v)(X, Y, 4*Lx/5, Ly/5) + np.vectorize(v)(X, Y, Lx/5, 4*Ly/5)

u[0] = u_0
u[1] = u[0] + v_0 * dt


"""
    Solve using the finite differences method.

    Consider u(x,t), a system discretized as follows:
    
        (space): x = [x_1, x, x_3, ..., x_N], i -> x_i
                 y = [x_1, x, x_3, ..., x_N], j -> y_j
        (time):  t = [t_1, t, t_3, ..., t_M], n -> t_n

    Using Taylor series we can approximate the discrete form of the
    differential factors composing the PDE. At the end, after making
    a couple of transformation, it comes the following formula:

    u(n+1,i,j) = 2u(n,i,j) - u(n-1,i,j)
               + r {  [u(n,i+1,j) - 2u(n,i,j) + u(n,i-1,j)] / dx² 
                    + [u(n,i,j+1) - 2u(n,i,j) + u(n,i,j-1)) / dy²) }

    where r = c²dt².
"""

r = (c * dt) ** 2

for n in range(1, nt-1):
    for i in range (1, nx-1):
        for j in range(1, ny-1):           
            u[n+1,i,j] = 2 * u[n,i,j] - u[n-1,i,j] + \
            r * ((u[n,i+1,j] - 2 * u[n,i,j] + u[n,i-1,j]) / dx**2 + \
                   (u[n,i,j+1] - 2 * u[n,i,j] + u[n,i,j-1]) / dy**2)




"""
    Animate solution.
"""

# Initialize figures
plt.rcParams["figure.figsize"]    = [6.0, 5.0]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['animation.embed_limit'] = 2**128

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(-2.1, 2.1)
ax.set_xlim(-0., 1.)
ax.set_ylim(-0., 1.)

ax.set_title('Vibrating membrene')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u(x,y)$')

# Update figure
def animate(n, u, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, u[n], cmap="magma", 
                                vmin=np.min(u)*1., vmax=np.max(u)*1.)

plot = [ax.plot_surface(X, Y, u[0], color='0.75', rstride=1, cstride=1, 
                          vmin=np.min(u)*1., vmax=np.max(u)*1.)]

ani = FuncAnimation(fig, animate, frames=nt, interval=20, 
                    fargs=(u, plot), save_count=20)

plt.show()