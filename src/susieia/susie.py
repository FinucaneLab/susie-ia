""" Base classes for models """
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from types import SimpleNamespace

import numpy as np
from scipy import special

def simple_linear_regression(xx, xy, sigma2, tau2):
    """ Moments and divergence term for effect size post """

    v = (xx / sigma2 + 1 / tau2)**-1
    b1 = v / sigma2 * xy
    b2 = v + b1**2
    divergence_to_prior = (-0.5 * (np.log(2 * np.pi * v) + 1)
        + 0.5 * np.log(2 * np.pi * tau2) + 0.5 / tau2 * b2)
    log_bayes_factor = (
        -0.5 / sigma2 * (xx * b2 - 2 * xy * b1)
        - divergence_to_prior
        )

    return b1, b2, log_bayes_factor, divergence_to_prior

class Estimator(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class Ensemble(Estimator):
    def __init__(self, estimators):
        self.estimators = estimators

class FixedUnivariateRegression(Estimator):
    def __init__(self, x, tau2):
        super().__init__()
        self._x = x
        self._x -= x.mean()
        self._tau2 = tau2
        self.b1 = None
        self.b2 = None
        self.log_bayes_factor = None
        self.divergence_to_prior = None
        self.gve_term = None

    def fit(self, y, sigma2):
        x1 = self._x
        x2 = x1**2

        xx = x2.sum()
        xy = x1 @ y

        self.b1, self.b2, self.log_bayes_factor, self.divergence_to_prior = (
            simple_linear_regression(xx, xy, sigma2, self._tau2)
            )
        self.gve_term = self.b2 * ((x2 - x1**2).sum() + x1.sum()**2)

    def predict(self, y, sigma2):
        x1 = self._x
        x2 = x1**2
        return x1 * self.b1, x2 * self.b2

class RandomUnivariateRegression(Estimator):
    """ Coordinate-ascent variational inference for the causal model for a SNP """

    def __init__(self, x, tau2):
        super().__init__()
        self._x = x
        self._support = np.array([0., 1., 2.])
        self._support -= (x @ np.array([0., 1., 2.])).mean()
        self._tau2 = tau2
        self.b1 = 0. # Zero is the best initialization
        self.b2 = 0.
        self.log_bayes_factor = None
        self.divergence_to_prior = None
        self.gve_term = None

        with np.errstate(divide='ignore'):
            self._log_x = np.log(x)

        eps = 1e-10
        self._log_x_thresh = self._log_x.copy()
        self._log_x_thresh[x == 0] = eps

    def fit(self, y, sigma2):
        tol = 1e-5
        converged = False
        prev_params = self._natural_params

        while not converged:
            x1, x2, divergence_to_prior_term = self._decode(y, sigma2)
            xx = x2.sum()
            xy = x1 @ y

            self.b1, self.b2, log_bayes_factor, divergence_to_prior = (
                simple_linear_regression(xx, xy, sigma2, self._tau2)
                )
            divergence_to_prior += divergence_to_prior_term
            log_bayes_factor -= divergence_to_prior_term

            grad = self._natural_params - prev_params

            if np.linalg.norm(grad) < tol:
                converged = True

            prev_params = self._natural_params

        self.divergence_to_prior = divergence_to_prior
        self.log_bayes_factor = log_bayes_factor
        self.gve_term = self.b2 * ((x2 - x1**2).sum() + x1.sum()**2)

    def predict(self, y, sigma2):
        x1, x2, *__ = self._decode(y, sigma2)
        return x1 * self.b1, x2 * self.b2

    def _decode(self, y, sigma2):
        """ Moments and divergence term for genotype post """

        params = self._log_x - 0.5 / sigma2 * (self._support**2 * self.b2
            - 2 * y.reshape(-1, 1) * self._support * self.b1)
        qx = special.softmax(params, axis=1)

        x1 = (qx * self._support).sum(axis=1)
        x2 = (qx * self._support**2).sum(axis=1)

        with np.errstate(divide='ignore'):
            log_qx = np.log(qx)

        # Avoids multiplication of inf and zero
        eps = -1e10
        log_qx[qx == 0] = eps

        divergence_to_prior = (qx * log_qx - qx * self._log_x_thresh).sum()

        return x1, x2, divergence_to_prior

    @property
    def _natural_params(self):
        v = self.b2 - self.b1**2

        if self.b1 != 0:
            eta1 = np.inf * np.sign(self.b1)
        else:
            eta1 = 0.

        eta2 = -np.inf

        if v != 0:
            eta1 = self.b1 / v
            eta2 = -0.5 / v

        return np.array([eta1, eta2])

class SingleEffectRegression(Ensemble):
    def __init__(self, estimators):
        super().__init__(estimators)

    def fit(self, y, pi, sigma2):
        for estimator in self.estimators:
            estimator.fit(y, sigma2)

        data = {}
        attrs = ['log_bayes_factor', 'divergence_to_prior', 'gve_term']

        for key in attrs:
            data[key] = np.array(
                [vars(estimator)[key] for estimator in self.estimators]
                )

        eps = -1e10

        with np.errstate(divide='ignore'):
            log_pi = np.log(pi)
            log_pi[log_pi == -np.inf] = eps
            pips = special.softmax(data['log_bayes_factor'] + log_pi)
            log_pips = np.log(pips)
            log_pips[log_pips == -np.inf] = eps

        self.pips = pips
        self.divergence_to_prior = (
            pips * data['divergence_to_prior']
            + pips * log_pips
            - pips * log_pi
            ).sum()
        self.gve_term = (pips * data['gve_term']).sum()

    def predict(self, y, sigma2):
        sample_size = len(y)

        mu1 = np.zeros(sample_size)
        mu2 = np.zeros(sample_size)

        for estimator, pip in zip(self.estimators, self.pips):
            mu1_term, mu2_term = estimator.predict(y, sigma2)
            mu1 += mu1_term * pip
            mu2 += mu2_term * pip

        return mu1, mu2

class SuSiE(Ensemble):
    def __init__(self, estimators, y, sigma2=None, pi=None):
        super().__init__(estimators)

        self._y = y
        self._sample_size = len(y)
        self._region_size = len(estimators[0].estimators)
        self.sigma2 = sigma2

        if sigma2 is None:
            self._set_sigma2 = True
        else:
            self._set_sigma2 = False

        if pi is None:
            self._pi = np.ones(self._region_size) / self._region_size
        else:
            self._pi = pi

        # Things that are useful to cache
        self._mu1_stack = None
        self._mu2_stack = None

    def fit(self, tol=1e-2, verbose=True):
        num_effects = len(self.estimators)
        self._mu1_stack = np.zeros((num_effects, self._sample_size)) # This is only fine because estimators initialize at zero
        self._mu2_stack = np.zeros_like(self._mu1_stack)

        converged = False
        prev_elbo = -np.inf
        t = 0

        if verbose:
            print(f't={t}, elbo={prev_elbo}')

        while not converged:
            t += 1

            r = self._y - self._mu1_stack.sum(axis=0)

            if self._set_sigma2:
                self.sigma2 = self._expected_residual_sum_of_squares / self._sample_size

            for i, estimator in enumerate(self.estimators):
                r += self._mu1_stack[i]
                estimator.fit(r, self._pi, self.sigma2)
                mu1, mu2 = estimator.predict(r, self.sigma2)
                r -= mu1
                self._mu1_stack[i] = mu1
                self._mu2_stack[i] = mu2

            elbo = self.evidence_lower_bound

            if elbo - prev_elbo < tol:
                converged = True
            
            prev_elbo = elbo

            if verbose:
                print(f't={t}, elbo={elbo}')

    def predict(self):
        mu1 = self._mu1_stack.sum(axis=0)
        mu2 = (
            self._mu2_stack.sum(axis=0)
            + mu1**2
            - (self._mu1_stack**2).sum(axis=0)
            )
        return mu1, mu2

    @property
    def _expected_residual_sum_of_squares(self):
        mu1_sum = self._mu1_stack.sum(axis=0)
        return (
            (self._y**2).sum()
            - 2 * self._y @ mu1_sum
            + self._mu2_stack.sum()
            - (self._mu1_stack**2).sum()
            + (mu1_sum**2).sum()
            )

    @property
    def evidence_lower_bound(self):
        divergences_to_priors = [
            estimator.divergence_to_prior
            for estimator in self.estimators
            ]

        return (
            - 0.5 * len(self._y) * np.log(2 * np.pi * self.sigma2)
            - 0.5 / self.sigma2
            * self._expected_residual_sum_of_squares
            - sum(divergences_to_priors)
            )

    @property
    def genetic_variance_explained(self):
        mu1 = self._mu1_stack
        mu2 = self._mu2_stack
        mu1_sum = mu1.sum(axis=0)
        mu2_sum = mu2.sum(axis=0)
        gve_term = sum(
            [estimator.gve_term for estimator in self.estimators]
            )

        return (
            (mu2_sum - mu1_sum**2).mean()
            + mu1_sum.var()
            + ((mu1.sum(axis=1))**2).sum() / self._sample_size**2
            - gve_term / self._sample_size**2
            )

    @property
    def probs(self):
        probs = []

        for estimator in self.estimators:
            probs.append(list(estimator.pips))

        return np.array(probs)

    @property
    def pips(self):
        return 1 - (1 - self.probs).prod(axis=0)

    @property
    def effect_sizes(self):
        effect_sizes = []

        for estimator in self.estimators:
            list_ = []

            for estimator_ in estimator.estimators:
                list_.append(vars(estimator_)['b1'])

            effect_sizes.append(list_)

        return (np.array(effect_sizes) * self.probs).sum(axis=0)

def fit(X, y, num_effects, tau2, sigma2=None, pi=None, verbose=False):
    sample_size = X.shape[0]
    region_size = X.shape[1]
    fixed = X.ndim == 2

    if np.isscalar(tau2):
        tau2 = np.ones(region_size) * tau2

    if fixed: 
        single_effect_regression = SingleEffectRegression(
            [FixedUnivariateRegression(X[:, i], tau2[i]) for i in range(region_size)]
            )
    else:
        single_effect_regression = SingleEffectRegression(
            [RandomUnivariateRegression(X[:, i], tau2[i]) for i in range(region_size)]
            )

    single_effect_regressions = [deepcopy(single_effect_regression) for __ in range(num_effects)]
    model = SuSiE(single_effect_regressions, y, sigma2, pi)
    model.fit(verbose=verbose)
    mu1, mu2 = model.predict()
    mse = (y**2 - 2 * y * mu1 + mu2).mean()

    ns = SimpleNamespace(
        pips=model.pips,
        effect_sizes=model.effect_sizes,
        mse=mse,
        gve=model.genetic_variance_explained
        )

    if sigma2 is not None:
        ns.sigma2 = model.sigma2

    ns.h2 = ns.gve / (ns.gve + ns.sigma2)

    return ns