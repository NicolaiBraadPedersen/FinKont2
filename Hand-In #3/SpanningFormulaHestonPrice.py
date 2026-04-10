import numpy as np
from sympy.parsing.sympy_parser import null

from HelpFunctions import *
from utils.plotting import use_earthy_style
import matplotlib.pyplot as plt
heston_fn = heston_fourier

use_earthy_style()

class SpanningFormulaPrice:
    def __init__(self):
        self.r = 0.02
        self.mu = 0.07
        self.s0 = 1
        self.v0 = 0.2 ** 2
        self.T = 30
        self.K = np.exp(self.r * self.T)
        self.a = 1
        self.N = 252
        self.theta = 0.2 ** 2
        self.kappa = 2
        self.epsilon = 1.0
        self.rho = -0.5

        self.sigma = 0.2
        self.a0 = 1
        self.gT = (self.a0 / (self.s0 ** self.a)) * np.exp(
            (1 - self.a) * (self.r + self.a * self.sigma ** 2 / 2) * self.T)
        self.s_prime = (self.K / self.gT) ** (1 / self.a)

        self.S = 0
        self.A = 0

    def price(self, points=100):
        def f_double_prime(k):
            if k < self.s_prime:
                return self.a * (1 - self.a) * self.gT * k ** (self.a - 2)
            return 0.0

        steps = np.linspace(0.00001, self.s_prime-0.00001, points)
        dk = steps[1] - steps[0]

        integrand = np.vectorize(
            lambda k: f_double_prime(k) * heston_fn(self.s0, self.T, k, self.r, 0,self.v0, self.theta, self.kappa, self.epsilon, self.rho)
        )(steps)

        dirac_term = self.a * self.gT * self.s_prime ** (self.a - 1) * heston_fn(self.s0, self.T, self.s_prime, self.r, 0,self.v0, self.theta, self.kappa, self.epsilon, self.rho)

        return np.sum(integrand) * dk + dirac_term

    def plot(self):
        point_grid = np.logspace(1, 4, 50).astype(int)
        prices = [self.price(points=n) for n in point_grid]

        fig, ax = plt.subplots()
        ax.plot([],[])
        ax.plot(point_grid, prices, label='Spanning Relpication Price')
        ax.set_xscale('log')
        ax.axhline(y=prices[-1], linestyle='--', label='Converged price')
        ax.set_xlabel('Number of discretization points')
        ax.set_ylabel('Price')
        ax.set_title('Convergence of spanning formula price')
        ax.legend()
        plt.tight_layout()
        plt.savefig('convergence_heston.png')

if __name__ == '__main__':
    span = SpanningFormulaPrice()
    span.plot()
    # point_grid = np.logspace(1, 4, 50).astype(int)
    # prices = [span.price(points=n) for n in point_grid]
    # print(prices[0], prices[-1])
    # _,x = span.price(points=50)
    # plt.scatter(np.arange(1, len(x) + 1), x)
    # plt.show()
