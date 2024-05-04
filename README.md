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


   
