import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy as sc

def squared_exp(x, y):
    return np.exp(-0.5 * np.abs(x - y)**2)

def matern_12(x, x_prime):
    """
    this will be matern assuming that nu = 5/2
    """
    if x.shape:
        r = np.linalg.norm(x - x_prime)
    else:
        r = np.abs(x - x_prime)
    e = np.exp(-r)
    return e

def matern_32(x, x_prime):
    """
    this will be matern assuming that nu = 5/2
    """
    if x.shape:
        r = np.linalg.norm(x - x_prime)
    else:
        r = np.abs(x - x_prime)
    c = 1 + np.sqrt(3)*r
    e = np.exp(-np.sqrt(3)*r)
    return c*e

def matern_52(x, x_prime):
    """
    this will be matern assuming that nu = 5/2
    """
    if x.shape:
        r = np.linalg.norm(x - x_prime)
    else:
        r = np.abs(x - x_prime)
    c = 1 + np.sqrt(5)*r + (5*r**2)/(3)
    e = np.exp(-np.sqrt(5)*r)
    return c*e

def rbf(x, x_prime, a, l):
    if x.shape:
        r = np.linalg.norm(x - x_prime)
    else:
        r = np.abs(x - x_prime)

    c = (-1/(2*l**2))*r**2
    e = a**2

    return(e*np.exp(c))
    
def cov_from_kernel(x, k):
    n = x.shape[0]
    cov = np.zeros((n, n))
    for i in range(n):
        xi = x[i]
        for j in range(n):
            xj = x[j]
            cov[i, j] = k(xi, xj)
    return cov

def cov_from_kernel_b(x, k, a, l):
    n = x.shape[0]
    cov = np.zeros((n, n))
    for i in range(n):
        xi = x[i]
        for j in range(n):
            xj = x[j]
            cov[i, j] = k(xi, xj, a, l)
    return cov

def cov_from_kernel1(x, y, k):
    n = x.shape[0]
    nstar = y.shape[0]
    cov = np.zeros((n, nstar))

    for i in range(n):
        xi = x[i]
        for j in range(nstar):
            yj = y[j]
            cov[i, j] = k(xi, yj)
    
    return cov

def cov_from_kernel1_b(x, y, k, a, l):
    n = x.shape[0]
    nstar = y.shape[0]
    cov = np.zeros((n, nstar))

    for i in range(n):
        xi = x[i]
        for j in range(nstar):
            yj = y[j]
            cov[i, j] = k(xi, yj, a, l)
    
    return cov

def gen_funs():
    n = 200
    a = 2
    l = 1
    t = np.linspace(0, 4*np.pi, n)
    c1 = cov_from_kernel(t, squared_exp)

    num_fns = 5
    f_star = rng.multivariate_normal(np.zeros(n), c1, num_fns)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(num_fns):
        ax.plot(t, f_star[i, :], color = 'grey', alpha=0.5)
        
    ax.plot(t, np.zeros(n), color = 'magenta', label = 'prior mean')
    #ax.plot(t, np.zeros(n), color = 'cyan', alpha = 0.1, linewidth = 222)
    ax.fill_between(t, 1.96, -1.96, alpha=0.25, color='cyan')
    # ax.plot(t, obj(t), color = 'black', label = 'true func')
    ax.plot(t, obj(t), color = 'black', label = 'true func')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    
    #plt.show()

def pred_from_obs():
    n = 200
    num_fns = 3
    sig = 0.25
    train_x = np.array([np.pi/2, np.pi, 4, 5, 7, 9, 10, 12])
    train_f = np.array([obj(np.pi/2), obj(np.pi), obj(4), obj(5), obj(7), obj(9), obj(10), obj(12)])

    test_x = np.linspace(0, 4*np.pi, n)
    k = lambda x, y: squared_exp(x, y)

    K = cov_from_kernel(train_x, k)
    K_star = cov_from_kernel1(train_x, test_x, k)
    K_star_star = cov_from_kernel(test_x, k) 

    m1 = K_star.T @ np.linalg.inv(K) @ train_f
    cov = K_star_star - K_star.T@np.linalg.inv(K)@K_star

    test_f = rng.multivariate_normal(m1, cov, num_fns)
    pw_mean = np.mean(test_f, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(test_x, pw_mean, c='magenta', label = 'posterior mean')
    ax.plot(test_x, obj(test_x), color = 'black', label = 'true func')
    sd_up = pw_mean + 2*np.sqrt(np.diagonal(cov))
    sd_down = pw_mean - 2*np.sqrt(np.diagonal(cov))

    ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='cyan')
    
    ax.scatter(train_x, train_f, alpha=1,  color = 'black')
    for i in range(num_fns):
        ax.plot(test_x, test_f[i, :], color = 'grey', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    plt.show()

    post_samples = np.random.multivariate_normal(pw_mean, cov, size=20)
    plt.scatter(train_x, train_f, color = 'black')
    for i in range(num_fns):
        plt.plot(test_x, test_f[i, :], color = 'grey', alpha=0.25)
    
    plt.plot(test_x, pw_mean, color = 'magenta', label = 'posterior mean')
    plt.plot(test_x, post_samples.T, color = 'grey', alpha=0.25)
    plt.plot(test_x, obj(test_x), color = 'black', label = 'true func')
    plt.fill_between(test_x, sd_up, sd_down, alpha=0.25, color = 'cyan')
    plt.legend()
    #legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
    plt.show()

def gen_funs_noise():
    n = 200
    t = np.linspace(0, 4*np.pi, n)
    c1 = cov_from_kernel(t, squared_exp)

    num_fns = 5
    f_star = rng.multivariate_normal(np.zeros(n), c1, num_fns)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(num_fns):
        ax.plot(t, f_star[i, :], color = 'grey', alpha=0.5)
        
    ax.plot(t, np.zeros(n), color = 'magenta', label = 'prior mean')
    #ax.plot(t, np.zeros(n), color = 'cyan', alpha = 0.1, linewidth = 222)
    ax.fill_between(t, 1.96, -1.96, alpha=0.25, color='cyan')
    # ax.plot(t, obj(t), color = 'black', label = 'true func')
    ax.plot(t, obj_err(t, 0.25), color = 'black', label = 'true func with noise')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()

def pred_from_obs_noise():
    n = 200
    num_fns = 3
    sig = 0.25
    train_x = np.array([np.pi/2, np.pi, 4, 5, 7, 9, 10, 12])
    train_f = np.array([obj_err(np.pi/2, 0.25), obj_err(np.pi, 0.25), obj_err(4, 0.25), obj_err(5, 0.25), obj_err(7, 0.25), obj_err(9, 0.25), obj_err(10, 0.25), obj_err(12, 0.25)])

    # train_x = np.array([np.pi/2, 4, 7, 10])
    # train_f = np.array([obj_err(np.pi/2, 0.25), obj_err(4, 0.25), obj_err(7, 0.25), obj_err(10, 0.25)])

    test_x = np.linspace(0, 4*np.pi, n)
    k = lambda x, y : squared_exp(x, y)

    K = cov_from_kernel(train_x, k) + np.diag(np.ones(len(train_x))*(0.25))
    K_star = cov_from_kernel1(train_x, test_x, k)
    K_star_star = cov_from_kernel(test_x, k)

    # print("K_xx shape: {}".format(k_x_x.shape))
    # print("K_x_xstar shape: {}".format(k_x_xstar.shape))
    # print("K_xstar_xstar shape: {}".format(k_xstar_xstar.shape))

    m1 = K_star.T @ np.linalg.inv(K) @ train_f
    cov = K_star_star - K_star.T@np.linalg.inv(K)@K_star

    test_f = rng.multivariate_normal(m1, cov, num_fns)
    pw_mean = np.mean(test_f, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(test_x, pw_mean, c='magenta', label = 'posterior mean')
    ax.plot(test_x, obj(test_x), color = 'black', label = 'true func with noise')
    sd_up = pw_mean + 2*np.sqrt(np.diagonal(cov))
    sd_down = pw_mean - 2*np.sqrt(np.diagonal(cov))

    ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='cyan')
    
    ax.scatter(train_x, train_f, alpha=1,  color = 'black')
    for i in range(num_fns):
        ax.plot(test_x, test_f[i, :], color = 'grey', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    plt.show()

    post_samples = np.random.multivariate_normal(pw_mean, cov, size=20)
    plt.scatter(train_x, train_f, color = 'black')
    for i in range(num_fns):
        plt.plot(test_x, test_f[i, :], color = 'grey', alpha=0.25)
    
    plt.plot(test_x, pw_mean, color = 'magenta', label = 'posterior mean')
    plt.plot(test_x, post_samples.T, color = 'grey', alpha=0.25)
    plt.plot(test_x, obj(test_x), color = 'black', label = 'true func with error')
    plt.fill_between(test_x, sd_up, sd_down, alpha=0.25, color = 'cyan')
    plt.legend()
    #legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
    plt.show()


def obj_err(x, sig):
    return np.sin(x) + np.random.normal(0, sig)
    
def obj(x):
    return np.sin(x)

rng = default_rng(5050505)
gen_funs()
pred_from_obs()
# gen_funs_noise()
# pred_from_obs_noise()
