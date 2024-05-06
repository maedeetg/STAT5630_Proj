'''THE FOLLOWING .PY FILE WAS RUN IN GOOGLE COLAB .IPYNB 
SO I DID NOT HAVE TO INSTALL TOO MANY PACKAGES HERE. '''

### The following code can be found from the following link https://data.world/rickyhennessy/seinfeld-scripts

"""scrape Seinfeld scripts."""

from bs4 import BeautifulSoup
import urllib
install
import pandas as pd

BASE_URL = 'http://www.imsdb.com'
URL = 'http://www.imsdb.com/TV/Seinfeld.html'
#r = urllib.urlopen(URL).read()

import urllib.request
import urllib.parse

with urllib.request.urlopen(URL) as url:
    r = url.read()

soup = BeautifulSoup(r)

episodes = soup.findAll("p")

all_episodes = {'episode_num': [],
                'title': [],
                'air_date': [],
                'text': []}
episode_num = 0

# iterate through each episode
for episode in episodes:
    # get the URLs for each episode script and open that page
    encoded_string = urllib.parse.quote(episode.a['href'])
    episode_url = BASE_URL + encoded_string
  
    with urllib.request.urlopen(episode_url) as url:
      episode_page = url.read()


    #episode_page = urllib.urlopen(episode_url).read()
    episode_soup = BeautifulSoup(episode_page)

    # get link to script text and extract text
    script_details = episode_soup.findAll("table", class_="script-details")
    script_url = BASE_URL + script_details[0].findAll("a")[-1]["href"]

    with urllib.request.urlopen(script_url) as url:
      script_page = url.read()
    
    #script_page = urllib.urlopen(script_url).read()
    script_soup = BeautifulSoup(script_page)
    script = (script_soup
              .findAll("td", class_="scrtext")[0]
              .findAll("pre")[0]
              .getText)

    title = episode.a['title']
    script = str(script)
    episode_num += 1
    date = str(episode.a.next_sibling)[2:12]

    all_episodes['episode_num'].append(episode_num)
    all_episodes['title'].append(title)
    all_episodes['air_date'].append(date)
    all_episodes['text'].append(script)

    print("Episode " + str(episode_num) + "/176")

seinfeld_df = pd.DataFrame(all_episodes)
seinfeld_df.to_csv('seinfeld_scripts.csv', index_label='episode_num')

"""Create plot of line frequency for each character in Seinfeld by episode."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read Seinfeld data into dataframe
df = pd.read_csv('https://query.data.world/s/760e0r2dlwhccgesguyfo3nb6')

# get lines per episode for each character
df['lines_jerry'] = df['text'].str.count('JERRY')
df['lines_george'] = df['text'].str.count('GEORGE')
df['lines_elaine'] = df['text'].str.count('ELAINE')
df['lines_kramer'] = df['text'].str.count('KRAMER')

# calculate total number of lines in an episode for all four main characters
df['lines_total'] = df[['lines_jerry',
                        'lines_george',
                        'lines_elaine',
                        'lines_kramer']].sum(axis=1)

elaine_lines = df['lines_elaine']/df['lines_total']
a = elaine_lines.to_numpy()
episode_num = df['episode_num']
b = episode_num.to_numpy()

data = {'Episode Number': b[:25], 'Elaine Line Percentage': a[:25]}
elaine_df = pd.DataFrame(data, columns = ['Episode Number', 'Elaine Line Percentage'])
elaine_df

plt.scatter(b[:25], a[:25], color = 'magenta')
plt.xlabel('Episode Number')
plt.ylabel('Percentage of lines')
plt.title("Percentage of Lines Elaine")
plt.show()

###########################################################################################################################################################
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
def gen_funs_noise():
    n = 200
    t = np.linspace(1, 176, n)
    a = 0.5
    l = 5
    c1 = cov_from_kernel_b(t, rbf, a, l)

    num_fns = 5
    f_star = rng.multivariate_normal(np.zeros(n) + 0.5, c1, num_fns)

    fig = plt.figure(figsize = (20, 6))
    ax = fig.add_subplot(111)

    for i in range(num_fns):
        ax.plot(t, f_star[i, :], color = 'grey', alpha=0.5)
        
    ax.plot(t, np.zeros(n) + 0.5, color = 'magenta', label = 'prior mean')
    #ax.plot(t, np.zeros(n), color = 'cyan', alpha = 0.1, linewidth = 222)
    ax.fill_between(t, 0.5 + 1.96, 0.5-1.96, alpha=0.25, color='cyan')
    # ax.plot(t, obj(t), color = 'black', label = 'true func')
    #ax.plot(t, obj_err(t, 0.25), color = 'black', label = 'true func with noise')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()

def pred_from_obs_noise():
    n = 200
    num_fns = 7
    sig = 0.25
    a = 0.5
    l = 5
    train_x = sample['Episode Number'].to_numpy()
    train_f = sample['Elaine Line Percentage'].to_numpy()
  
    # train_x = np.array([np.pi/2, 4, 7, 10])
    # train_f = np.array([obj_err(np.pi/2, 0.25), obj_err(4, 0.25), obj_err(7, 0.25), obj_err(10, 0.25)])

    test_x = all['Episode Number'].to_numpy()
    k = lambda x, y, a, l: rbf(x, y, a, l)

    K = cov_from_kernel_b(train_x, k, a, l) + np.diag(np.ones(len(train_x))*(0.004))
    K_star = cov_from_kernel1_b(train_x, test_x, k, a, l)
    K_star_star = cov_from_kernel_b(test_x, k, a, l)

    # print("K_xx shape: {}".format(k_x_x.shape))
    # print("K_x_xstar shape: {}".format(k_x_xstar.shape))
    # print("K_xstar_xstar shape: {}".format(k_xstar_xstar.shape))

    m1 = K_star.T @ np.linalg.inv(K) @ train_f
    cov = K_star_star - K_star.T@np.linalg.inv(K)@K_star

    test_f = rng.multivariate_normal(m1, cov, num_fns)
    pw_mean = np.mean(test_f, axis=0)

    fig = plt.figure(figsize = (20, 6))
    ax = fig.add_subplot(111)

    ax.plot(test_x, pw_mean, c='magenta', label = 'posterior mean')
    # ax.plot(test_x, obj(test_x), color = 'black', label = 'true func with noise')
    sd_up = pw_mean + 2*np.sqrt(np.diagonal(cov))
    sd_down = pw_mean - 2*np.sqrt(np.diagonal(cov))
    ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='cyan')
    ax.scatter(all["Episode Number"], all["Elaine Line Percentage"], color = 'purple', label = 'actual data', s = 4)
    
    ax.scatter(train_x, train_f, alpha=1,  color = 'black', label = 'sample', s = 60)
    for i in range(num_fns):
        ax.plot(test_x, test_f[i, :], color = 'grey', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    plt.show()

    fig = plt.figure(figsize = (20, 6))
    post_samples = np.random.multivariate_normal(pw_mean, cov + np.diag(np.ones(len(pw_mean)))*(0.001), size=20)
    plt.scatter(train_x, train_f, color = 'black')
    for i in range(num_fns):
        plt.plot(test_x, test_f[i, :], color = 'grey', alpha=0.25)
    
    plt.plot(test_x, pw_mean, color = 'magenta', label = 'posterior mean')
    plt.plot(test_x, post_samples.T, color = 'grey', alpha=0.25)
    # plt.plot(test_x, obj(test_x), color = 'black', label = 'true func with error')
    plt.fill_between(test_x, sd_up, sd_down, alpha=0.25, color = 'cyan')
    plt.legend()
    #legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
    plt.show()

rng = default_rng(5050505)
gen_funs_noise()
pred_from_obs_noise()