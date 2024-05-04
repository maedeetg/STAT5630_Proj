# Introduction

Traditional optimization methods like grid or random searches are computationally expensive and inefficient when working with noisy or high-dimensional functions. We can remedy this issue by implementing Bayesian optimization, a method used to find the global maximum of an objective function, $f$.

Bayesian optimization is a nontrivial optimization technique that takes advantage of the key methods in Bayesian statistics to build an objective function, particularly focusing on the incorporation of prior knowledge and conjugate priors. By integrating prior information into the optimization process, we can capture the uncertainty of the unknown objective function, enabling more informed decisions. In addition, conjugate priors offer an advantage in simplifying the computation of the posterior, since we know the form of the distribution. After developing the necessary and relevant theoretical and numerical components, if time permits, I will apply the methods to a relevant dataset.

# The Problem
Consider the following real-valued objective function $f: \mathcal{X} → \mathbb{R}$. We are interested in searching for a point $x^{*} \in \mathcal{X}$ that attains the global maximum value $f^{*} = f(x^{*})$. 

# The Methods

Before deep diving into the Bayesian methods, we first need to consider a basic sequential global optimization method for the following optimization problem.

### Sequential Optimization

1. We begin with a dataset $\mathcal{D}$. Note that $\mathcal{D}$ can be empty.
2. $\mathcal{D}$ will grow incrementally by inspecting the avaliable data and selecting an optimal point $x\in\mathcal{X}$ as our next observation.

3. Observe corresponding $y$ value for each $x$ selected. 
4. Update $\mathcal{D}$ with new point $(x, y)$, i.e. $\mathcal{D} = \mathcal{D} \cup \{(x, y)\}$
5. Repeat steps 2 through 4 until we reach user inputted termination condition
6. Return $\mathcal{D}$

The algorithm outlined iteritavely constructs a set of observations $\mathcal{D}$ through choosing an optimal point $x$ and finding the corresponding $y$ value. The user is permited to choose the method to find the observation $x$ at each step and how the algorithm will be terminated. This algorithm provides a basic outline on how Bayesian optimization will be implemented. 
### Bayesian Inference

A natural question becomes: **How will we use Bayestian statistical methods?**  In simplicity, we will place a prior distribution on the unknown objective function, construct a likelihood function given observed data, and then construct a posterior distribution. As we obtain new observations, we set the posterior distribution as our prior for subsequent inference steps. Through this iterative process, we update our likelihood function with the new data and derive an updated posterior distribution. This allows us to continuously refine our estimation of $\phi$ based on the additional observations provided by the previous observations.

Now to answer this question in more detail, first consider Bayesian inference at a value of the objective function $\phi = f(x)$ for some $x\in\mathcal{X}$. 

* We start by establishing a *prior distribution*, representing possible values of $\phi$ given $x$. The prior is denoted as $p(\phi\text{ }|\text{ }x)$. 

* Next, we need to refine our beliefs given observed data, i.e. constructing a *likelihood function*. For a given point $x$ and an observation $y = f(x)$, we want to describe the distribution of $y$ given our value of interest $\phi$. The likelihood is denoted as $p(y \text{ }|\text{ }x, \phi)$.

* Lastly, we construct the *posterior distribution* of our value of interest $\phi$ given our observed value $y$. The posterior is denoted as $p(y\text{ }|\text{ }x)$ and can be expressed as $$p(y\text{ }|\text{ }x) = ∫ p(y \text{ } | \text{ } x, \phi)p(\phi \text{ } | \text{ } x) d\phi$$

Now, let's extend this to inference of the **entire** objective function, $\boldsymbol{\phi}$. The inference is analoug to the single point example described above. 

* We start by establishing a *prior process* over the objective function $\boldsymbol{\phi}$. The prior process is denoted as $p(\boldsymbol{\phi} \text{ }|\text{ } \boldsymbol{x})$ where $\boldsymbol{x}$ is a **finite** set. 

  * A convenient and common prior process is a *Gaussian process*. This will be the only prior process I will be using and more will be discussed on it in a later section

* Next, with a defined prior process, suppose that we have a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) $ where $\boldsymbol{x}$ is a set of observations and $\boldsymbol{y}$ are the corresponding values. Note that we assume that the set of measurements $y$ conditionally independent given the location $x$ and objective function $\phi = f(x)$. Given the information above, we can build the *likelihood function* as follows $$p(\boldsymbol{y} |\boldsymbol{x}, \boldsymbol{\phi}) = \prod_{i = 1}^{n}p(y_i | x_i, \phi_i)$$

* Finally, we can construct a *posterior process*, which is denoted as $p(f \text{ } | \text{ } \mathcal{D})$. It is constructed in the following steps

  * Given our prior process, $p(\boldsymbol{\phi} \text{ }|\text{ } \boldsymbol{x})$, $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y})$, and likelihood function, $p(\boldsymbol{y} \text{ } | \text{ } \boldsymbol{x}, \boldsymbol{\phi})$, we can form $$p(\boldsymbol{\phi} \text{ } | \text{ } \mathcal{D}) ∝ p(\boldsymbol{\phi} \text{ } | \text{ } \boldsymbol{x})p(\boldsymbol{y} \text{ } | \text{ } \boldsymbol{x}, \boldsymbol{\phi})$$

  * Extending the posterior on $\boldsymbol{\phi}$ to all of $f$, we get $$p(f\text{ }|\text{ }\mathcal{D}) = ∫ p(f \text{ } | \text{ } \boldsymbol{x}, \boldsymbol{\phi})p(\boldsymbol{\phi} \text{ } | \text{ } \mathcal{D}) d\boldsymbol{\phi}$$

# Gaussian Process: How to Build a Useful Prior Distribution

A Gaussian process (GP) is a mathematical framework used to model a collection of random variables representing functions, where any finite subset of these variables follows a multivariate normal distribution. In simpler terms, a GP is a way to represent a distribution over functions. Instead of modeling functions directly, GPs model the distribution of functions themselves.

Again, let's consider an objective function $f: \mathcal{X} → \mathbb{R}$ such that $\mathcal{X}$ is an infinite domain. We will consider the collection of random variables, where each random variable represents a function value for every point in the domain. 

Dealing a infinite collection of RVs can be daunting and using the infinite collection of RVs to create a useful distribution is infeasible. However, we can address the issue by condensing the problem to a finite set of function values due to the Kolmogorov extension theorem, a theorem justifying constructing a stochastic process from a finite collection of RVs. In the context of GPs, these finite collections of RVs each are multivariate normal distributions.  

A GP is specified by its mean function and a positive semidefinite covariance function. 

The mean function, $\mu: \mathcal{X} → \mathbb{R}$, calculates $\mu(x) = \mathbb{E}[\phi \text{ } | \text{ } x]$, the expected value of $\phi = f(x)$ at any point $x$. 

The covariance function, $K: \mathcal{X}\times\mathcal{X} → \mathbb{R}$, defines the structure of deviations from the mean, $\mu$, while capturing properties of the function's behavior. Define a new objective value $\phi' = f(x')$, we can define the covariance function as $K(x, x') = cov[\phi, \phi' \text{ } | \text{ } x, x']$ or $K(x, x') = \mathbb{E}[(\phi - \mu(x))(\phi' - \mu(x'))]$. 

The GP is then defined as $$p(f) = GP(f; \mu, K)$$

Using the defined mean and covariance functions, we can compute any finite-dimensional marginal distribution. For example, let $\boldsymbol{x} ⊂ \mathcal{X}$ be finite and $\boldsymbol{\phi} = f(\boldsymbol{x})$ be the respective function values. Using a GP, $\boldsymbol{\phi}$ is a multivaraite normal with the following parameters $$p(\boldsymbol{\phi} \text{ } | \text{ } \boldsymbol{x}) = \mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\mu}, \Sigma)$$ where $$\boldsymbol{\mu} = \mathbb{E}[\boldsymbol{\phi} \text{ } | \text{ } \boldsymbol{x}] = \mu(x)$$ and $$\Sigma = cov[\boldsymbol{\phi} \text{ } | \text{ } \boldsymbol{x}] = K(\boldsymbol{x}, \boldsymbol{x})$$

# Example

Now let's illustrate an example to put all the information above to good use. 

Let's consider a function $f: \mathcal{X} → \mathbb{R}$ where $\mathcal{X} = [0, 30]$. Let the mean function be identically zero, $\mu = 0$ and let the covariance function be the squared exponential, $K(x, x') = exp(-\frac{1}{2}|x - x'|^2)$. 

Note that $K(x, x) = exp(-\frac{1}{2}|x - x|^2) = exp(0) = 1$, so the squared exponential measures correlation and has the following interpretation: function values that are close to eachother are highly correlated and function values that are far way are independent. Below is a graph of the squared exponential to illustrate that larger distances between points will result in a smaller squared exponential. 

<img src="https://drive.google.com/uc?export=view&id=1Mw3dV3LmwFqITBcHWLYgDwtKP8T7XP-m" alt="Square Exponential" width="300"/>

To generate the GPs, we simulate vectors from a joint multivariate normal distribution with mean $\mu = 0$ and a positive semidefinite covariance $\Sigma = K(\boldsymbol{x}, \boldsymbol{x})$. In other words, we draw vectors from $\mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\mu}, \Sigma)$. Shown in the plot below, I simulate vectors $\boldsymbol{\phi_i} \in \mathbb{R}^n$ for $i = 1, 2, 3$ and $ n = 200$. Each $\boldsymbol{\phi_i}$ represents a GP or prior process. The shaded grey region represents a $95\%$ confidence interval. In this example, since the marginal distributions for any single function value is a univariate standard normal, the CI is $\mu \pm 1.96$.  

<img src="https://drive.google.com/uc?export=view&id=1KzhWOzDXc2pLMNwNfTLQaETLFP189xuQ" alt="Square Exponential" width="1000"/>

We have generated a prior process! We have one step out of the way! Now we move onto constructing a likelihood function from observing data
   
