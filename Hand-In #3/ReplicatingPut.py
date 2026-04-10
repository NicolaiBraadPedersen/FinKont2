import numpy as np
import matplotlib.pyplot as plt
import HelpFunctions

from utils.plotting import use_earthy_style
use_earthy_style()

class HedgeExperiment:
    def __init__(self):
        self.r = 0.02
        self.mu = 0.07
        self.sigma = 0.2
        self.s0 = 1
        self.a0 = 1
        self.T = 30
        self.K = np.exp(self.r*self.T)
        self.a = 0.5
        self.N = 10*252

        self.S = 0
        self.A = 0

    def simulate_paths(self):
        self.S = self.sim_gbm()
        g = self.g_function()
        self.A = g[np.newaxis, :] * self.S ** self.a

    def sim_gbm(self, paths=100):
        N = self.N
        dt = 1 / N
        Z = np.random.normal(0, np.sqrt(dt), size=(paths, self.T * N))
        x = np.zeros((paths, self.T * N + 1))
        x[:, 0] = self.s0
        x[:, 1:] = self.s0 * np.exp(np.cumsum((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * Z, axis=1))
        return x

    def g_function(self):
        N = self.N
        dt = 1 / N
        t = np.arange(self.T * N + 1) * dt
        exponent = (1 - self.a) * (self.r + self.a * self.sigma ** 2 / 2) * t
        return (self.a0 / (self.s0**self.a)) * np.exp(exponent)

    def replicate_paths(self):

        delta_fn = HelpFunctions.put_delta
        price_fn = HelpFunctions.put_price

        N = self.N
        dt = 1 / N
        steps = self.T * N
        Omega = self.S.shape[0]

        t_grid = np.arange(steps + 1) * dt
        g = self.g_function()

        h_S = np.zeros((Omega, steps))
        h_B = np.zeros((Omega, steps))
        V = np.zeros((Omega, steps + 1))

        V[:, 0] = price_fn(self.A[:, 0], 0, self.T, self.K, self.r, self.a * self.sigma)
        for i in range(steps):
            t = t_grid[i]

            S_t = self.S[:, i]
            A_t_a = self.A[:, i]

            delta = delta_fn(A_t_a, t, self.T, self.K, self.r, self.a * self.sigma)
            h_S[:, i] = delta * self.a * g[i] * S_t ** (self.a - 1)

            h_B[:, i] = (V[:, i] - h_S[:, i] * S_t)

            S_next = self.S[:, i + 1]
            V[:, i + 1] = h_S[:, i] * S_next + h_B[:, i] * np.exp(self.r * dt)

        self.V = V

        return h_S, h_B, V

    def plot_replication(self):

        S_T = self.S[:, -1]  # stock price at maturity, shape (Omega,)
        A_T_a = self.A[:, -1]  # A_T^a at maturity,       shape (Omega,)
        V_T = self.V[:, -1]  # replication value at T,  shape (Omega,)

        payoff = np.maximum(self.K - A_T_a, 0)

        # Sort by S_T for clean line plots
        idx = np.argsort(S_T)
        S_T = S_T[idx]
        payoff = payoff[idx]
        V_T = V_T[idx]

        fig, ax = plt.subplots()

        ax.plot([], [])
        ax.plot(S_T, payoff, label='True payoff')
        ax.scatter(S_T, V_T, label='Replication portfolio')
        ax.set_xlabel('$S_T$')
        ax.set_ylabel('Payoff')
        ax.set_xlim(left = 0, right=6)
        ax.set_title(f'Put payoff vs replication')
        ax.legend()

        plt.tight_layout()
        plt.savefig('replicating_put.png')

if __name__ == '__main__':
    he = HedgeExperiment()
    he.simulate_paths()
    he.replicate_paths()
    he.plot_replication()
