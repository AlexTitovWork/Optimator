import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def gradient_descent(max_iterations, threshold, XY_init,
                     obj_func, grad_func, extra_param=[],
                     learning_rate=0.05, momentum=0.8):
    X, Y = XY_init
    w = np.array([X, Y])
    w_history = X, Y
    f_history = obj_func(X, Y, extra_param)
    delta_w = np.zeros(XY_init.shape)
    i = 0
    diff = 1.0e9

    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w[0], w[1], extra_param)
        w = w + delta_w
        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w[0], w[1], extra_param)))
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


def plot_function():
    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    return X, Y, Z




# Himmelblau's function is a multi-modal function, used to test the performance of optimization algorithms. The function is defined by:
# https://en.wikipedia.org/wiki/Himmelblau%27s_function
def f(X, Y, extra=[]):
    return (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2


# Function to compute the gradient
def grad(X, Y, extra=[]):
    dx = 4 * X * (X ** 2 + Y - 11) + 2 * (X + Y ** 2 - 7)
    dy = (2 * (X ** 2 + Y - 11) + 4 * Y * (X + Y ** 2 - 7))
    return np.array([dx, dy])

# ################################################################################
X, Y, Z = plot_function()
rand = np.random.RandomState(23)
XY_init = rand.uniform(0, 5, 2)
learning_rates = [0.05, 0.2, 0.5, 0.8]

w_history, f_history = gradient_descent(5, -1, XY_init, f, grad, [], learning_rate=0.02, momentum=0.8)

print(w_history)
print(f_history)
# ################################################################################


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, c='r')
ax.scatter(w_history[:, 0], w_history[:, 1], f_history[:, 0], 'xb-')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# plt.plot(w_history[:, 0], w_history[:, 1],f_history, 'ro')
## coloring couttour line
fig.show()


fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# lev = [1, 2, 3, 4, 6, 10, 20, 40, 100, 900]
lev = np.linspace(0, 200, 15)
CS = plt.contour(X, Y, Z, levels=lev)
# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
# ax.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
ax.clabel(CS, inline=1, fontsize=8)
ax.set_title('Himmelblau\'s function')
ax.plot(w_history[:, 0], w_history[:, 1],  'b*-')
fig.set_figwidth(12)    #  ширина и
fig.set_figheight(6)    #  высота "Figure"
show()
print("Plotting complete")
