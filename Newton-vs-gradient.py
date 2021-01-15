# About code.
# In this code researched Gradient or Cauchy method - first order method vs Newton's method -  second order method
# The code illustrates the advantages of the second order method over the first order method.
# The classical gradient algorithm has been writed and ploted its convergence.
# The Newton's method has been writed.
# This is a second-order method using the Hessian matrix - a matrix of estimates of the second derivatives.
# The convergence of Newton's method is investigated vs Gradient convergence.
# Coded by      Alex Titov
# Last update   13.01.2021
# E-mail        alexeytitovwork@gmail.com
# ################################################################################

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
mu, sigma = 0, 0.5  # mean and standard deviation

# ################################################################################
# Gradient descent or Cauchy method
# First order method
def gradient_descent(max_iters, threshold, XY_init,
                     func, grad_func,
                     learning_rate=0.05, extra_param=[]):
    X, Y = XY_init
    w = np.array([X, Y])
    w_history = X, Y
    f_history = func(X, Y, extra_param)
    delta_w = np.zeros(XY_init.shape)
    i = 0
    # start diff |f2 - f1| for stop criteria
    diff_f = 1.0e10
    eps_history_f = np.array([0.0])
    eps_history_xy = np.array([0.0 , 0.0])

    while i < max_iters and diff_f >= threshold:
        delta_w = -learning_rate * grad_func(w[0], w[1], extra_param)
        w = w + delta_w
        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, func(w[0], w[1], extra_param)))
        i += 1
        diff_f = np.absolute(func(3.0, 2.0,extra_param) - f_history[-1])
        # diff_f = np.absolute(f_history[-1] - f_history[-2])
        diff_xy = np.absolute(w_history[-1] - w_history[-2])

        eps_history_f = np.vstack((eps_history_f, diff_f))
        eps_history_xy = np.vstack((eps_history_xy, diff_xy))

    return w_history, f_history, eps_history_f, eps_history_xy

# ################################################################################
# Newton's method
# Second order method
def Newtons_method(max_iters, threshold, XY_init,
                     func, grad_func, second_grad,
                     learning_rate=0.05, extra_param=[]):
    X, Y = XY_init
    w = np.array([X, Y])
    w_history_N = X, Y
    f_history_N = func(X, Y, extra_param)
    delta_w = np.zeros(XY_init.shape)
    i = 0
    # start diff |f2 - f1| for stop criteria
    diff_f = 1.0e10
    eps_history_N = np.array([0.0])
    eps_history_xy_N = np.array([0.0 , 0.0])

    while i < max_iters and diff_f >= threshold:
        # print(grad_func(w[0], w[1], extra_param))
        # print(np.linalg.inv(second_grad(w[0], w[1], extra_param)))
        # Inverse Hessian matrix of second derivatives.
        inverseHessian = np.linalg.inv(second_grad(w[0], w[1], extra_param))
        # Hessian step calculation, it is not define descent direction.
        # The Hessian matrix characterizes the convexity or concavity of a
        # surface and can change the sign of the determinant at the inflection point.
        inverseHessian = np.abs(inverseHessian)
        delta_w = - learning_rate * np.dot(grad_func(w[0], w[1], extra_param), inverseHessian)
        # delta_w = - np.dot(grad_func(w[0], w[1], extra_param),inverseHessian)

        w = w + delta_w
        # store the history of w and f
        w_history_N = np.vstack((w_history_N, w))
        f_history_N = np.vstack((f_history_N, func(w[0], w[1], extra_param)))

        i += 1
        diff_f = np.absolute(func(3.0, 2.0,extra_param) - f_history_N[-1])
        # diff_f = np.absolute(f_history_N[-1] - f_history_N[-2])

        diff_xy = np.absolute(w_history_N[-1] - w_history_N[-2])

        eps_history_N = np.vstack((eps_history_N, diff_f))
        eps_history_xy_N= np.vstack((eps_history_xy_N, diff_xy))
    return w_history_N, f_history_N, eps_history_N, eps_history_xy_N

# ################################################################################
def plot_function(noised=False):
    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    # Gaussian distribution
    if noised:
        # Moved to the global parameters
        # mu, sigma = 0, 10  # mean and standard deviation
        noise = np.random.normal(mu, sigma, Z.shape)
        # Noised surface
        Z = np.add(Z, noise)

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    return X, Y, Z

# ################################################################################
# Himmelblau's function is a multi-modal function, used to test the performance of optimization algorithms. The function is defined by:
# https://en.wikipedia.org/wiki/Himmelblau%27s_function
def f(X, Y, extra=[]):
    # Moved to the global parameters
    # mu, sigma = 0, 10  # mean and standard deviation
    noise = np.random.normal(mu, sigma, 1)
    Zloc = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2 + noise
    return Zloc


# Function to compute the gradient
# Gradient Himmelblau's function
def grad(X, Y, extra=[]):
    dx = 4 * X * (X ** 2 + Y - 11) + 2 * (X + Y ** 2 - 7)
    dy = (2 * (X ** 2 + Y - 11) + 4 * Y * (X + Y ** 2 - 7))
    return np.array([dx, dy])

# Function to compute the gradient
# Gradient Himmelblau's function
def second_grad(X, Y, extra=[]):
    # dx = 4*X ** 3 + 4*X*Y - 44*X + 2*X + 2*Y ** 2 - 14
    # dx2 = 12 * X ** 2 + 4*Y - 44 + 2
    dx2 = 12*X**2 + 4*Y - 42

    dxy = 4*X + 4*Y

    # dy = 2*X**2 + 2*Y - 22 + 4*Y*X + 4*Y**3 - 28*Y
    # dy2 = 2 + 4*X + 12*Y**2 - 28
    dy2 = 12*Y**2 + 4*X - 26

    dyx = 4*X + 4*Y
    # Hessian
    # H = [[dx2, dxy],[dyx, dy2]]
    return np.array([[dx2, dxy], [dyx, dy2]])

# ################################################################################
# Start main code
# ################################################################################
X, Y, Z = plot_function(noised=True)
rand = np.random.RandomState(23)
# XY_init = rand.uniform(-0.5, 0.5, 2)
# XY_init = np.array([3.5, -2])
# XY_init = np.array([-2, -1.9])
XY_init = np.array([0.03, 1.9])

learning_rates = [0.05, 0.2, 0.5, 0.8]
max_iter = 500
threshold = 0.0001
# Gradient descent
w_history, f_history, eps_history, eps_history_xy_G = gradient_descent(max_iter, threshold, XY_init, f, grad,  learning_rate=0.001)
print("Grad")
print(w_history)
print(f_history)
# Newton descent
w_history_N, f_history_N, eps_history_N, eps_history_xy_N = Newtons_method(max_iter, threshold, XY_init, f, grad, second_grad,  learning_rate=0.5)
print("Newton")
print(w_history_N)
print(f_history_N)
# ################################################################################
# end main code
# ################################################################################
# Plotting surface and trace of optimal point search
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# mesh3d = ax.scatter(X, Y, Z, cm='b')
mesh3d = ax.plot_wireframe(X, Y, Z)
plt.setp(mesh3d, 'color', 'b', 'linewidth', 0.2)
ax.scatter(w_history[:, 0], w_history[:, 1], f_history[:, 0],  marker='+', alpha=0.5, c='r')
ax.scatter(w_history_N[:, 0], w_history_N[:, 1], f_history_N[:, 0], marker='+', alpha=0.5, c='g' )

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')
# plt.plot(w_history[:, 0], w_history[:, 1],f_history, 'ro')

## coloring couttour line
fig.show()
names = ['Gradient', 'Newton']
fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# lev = [1, 2, 3, 4, 6, 10, 20, 40, 100, 900]
lev = np.linspace(0, 200, 15)
CS = plt.contour(X, Y, Z, levels=lev)
# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
# ax.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
ax.clabel(CS, inline=1, fontsize=8)
ax.set_title('Himmelblau\'s function')
ax.plot(w_history[:, 0], w_history[:, 1],  'r*-', label=names[0])
ax.plot(w_history_N[:, 0], w_history_N[:, 1],  'g+-', label=names[1])
plt.legend()
ax.set_xlabel('X ')
ax.set_ylabel('Y ')
fig.set_figwidth(12)    #  ширина и
fig.set_figheight(6)    #  высота "Figure"
fig.savefig('images//Figure_4.png')
print("Plotting contour complete")

# ################################################################################
# Convergence of function f(x,y) plotting: Newton vs Gradient
plt.figure(figsize=(4, 2))
plt.subplot(411)
e_Gradent_line = plt.plot(eps_history, label = 'Gradient')
plt.setp(e_Gradent_line, 'color', 'r', 'linewidth', 1.1)
# plt.xlabel('n, iterations')
plt.ylabel('$\epsilon$ ')
# plt.title('$\epsilon$,  Gradient')
plt.grid(True)
plt.axis([0, 300, 0, 100])
plt.legend()


plt.subplot(412)
e_Newton_line = plt.plot(eps_history_N, label = 'Newton')
plt.setp(e_Newton_line, 'color', 'g', 'linewidth', 1.1)
# plt.xlabel('n, iterations')
plt.ylabel('$\epsilon$')
# plt.title('$\epsilon$, Newton')
plt.grid(True)
plt.axis([0, 300, 0, 100])
plt.legend()

# plt.set_xlabel('X Label')
# plt.set_ylabel('Y Label')
plt.suptitle('Functions and arguments convergence ')

# Convergence of argument [x,y] plotting: Newton vs Gradient

# names = ['eps Gradient', 'eps Newton']
plt.subplot(413)
# e_Gradent_line = plt.plot(eps_history_xy_G)
e_Gradent_line = plt.plot(w_history[:,0], label = 'x')
plt.setp(e_Gradent_line, 'color', 'r', 'linewidth', 1.2)
e_Gradent_line = plt.plot(w_history[:,1], label = 'y')
plt.setp(e_Gradent_line, 'color', 'r', 'linewidth', 1.2)
# plt.xlabel('n, iterations')
plt.ylabel('Gradient desc. args, $[x,y]$')
# plt.title('Conv. to optimal argument,  Gradient')
plt.grid(True)
plt.axis([0, 300, 0, 4])
plt.legend()


plt.subplot(414)
# e_Newton_line = plt.plot(eps_history_xy_N)
e_Newton_line = plt.plot(w_history_N[:,0],  label = 'x')
plt.setp(e_Newton_line, 'color', 'lime', 'linewidth', 1.2)
e_Newton_line = plt.plot(w_history_N[:,1],  label = 'y')
plt.setp(e_Newton_line, 'color', 'g', 'linewidth', 1.2)
plt.xlabel('n, iterations')
plt.ylabel('Newton args, $[x,y]$')
# plt.title('Conv. to optimal argument,  Newton')
plt.grid(True)
plt.axis([0, 300, 0, 4])
plt.legend()

show()
print("Plotting convergence complete")


