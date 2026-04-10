import numpy as np
import matplotlib.pyplot as plt
import HelpFunctions

from utils.plotting import use_earthy_style
use_earthy_style()

class HedgeExperiment:
    def __init__(self):
        self.r = 0.02
        self.mu = 0.07
        self.s0 = 1
        self.v0 = 0.2**2
        self.T = 30
        self.K = np.exp(self.r*self.T)
        self.a = 1
        self.N = 252
        self.theta = 0.2**2
        self.kappa = 2
        self.epsilon = 1.0
        self.rho = -0.5


        self.S = 0
        self.A = 0

    def simulate_paths(self):
        self.S = self.sim_heston()

    def sim_heston(self, paths=100):
        N = self.N  # steps per year
        dt = 1 / N
        steps = self.T * N

        # Correlated Brownian increments
        Z1 = np.random.normal(0, 1, size=(paths, steps))
        Z2 = np.random.normal(0, 1, size=(paths, steps))
        W1 = np.sqrt(dt) * Z1
        W2 = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2)

        S = np.zeros((paths, steps + 1))
        V = np.zeros((paths, steps + 1))
        S[:, 0] = self.s0
        V[:, 0] = self.v0

        for t in range(steps):
            V_pos = np.maximum(V[:, t], 0)
            S[:, t + 1] = S[:, t] * np.exp((self.r - 0.5 * V_pos) * dt + np.sqrt(V_pos) * W1[:, t])
            V[:, t + 1] = V[:, t] + self.kappa * (self.theta - V_pos) * dt + self.epsilon * np.sqrt(V_pos) * W2[:, t]

        return S

    def replicate_paths(self):

        heston_fn = HelpFunctions.heston_fourier

        N = self.N
        dt = 1 / N
        steps = self.T * N
        Omega = self.S.shape[0]

        t_grid = np.arange(steps + 1) * dt

        h_S = np.zeros((Omega, steps))
        h_B = np.zeros((Omega, steps))
        V = np.zeros((Omega, steps + 1))

        V[:, 0] = heston_fn(self.s0, self.T, self.K, self.r, 0,self.v0, self.theta, self.kappa, self.epsilon, self.rho)
        for i in range(steps):
            t = t_grid[i]

            S_t = self.S[:, i]
            delta = np.vectorize(lambda s: heston_fn(s, self.T-t, self.K, self.r, 0, self.v0, self.theta, self.kappa, self.epsilon, self.rho, greek=2) - 1)(S_t)
            h_S[:, i] = delta
            h_B[:, i] = (V[:, i] - h_S[:, i] * S_t)

            S_next = self.S[:, i + 1]
            V[:, i + 1] = h_S[:, i] * S_next + h_B[:, i] * np.exp(self.r * dt)

        self.V = V

        return h_S, h_B, V

    def plot_replication(self):

        S_T = self.S[:, -1]  # stock price at maturity, shape (Omega,)
        V_T = self.V[:, -1]  # replication value at T,  shape (Omega,)

        payoff = np.maximum(self.K - S_T, 0)

        # Sort by S_T for clean line plots
        idx = np.argsort(S_T)
        S_T = S_T[idx]
        payoff = payoff[idx]
        V_T = V_T[idx]

        fig, ax = plt.subplots()

        ax.plot([], [])
        ax.plot(S_T, payoff, label='True payoff')
        ax.scatter(S_T, V_T, label='Replication portfolio')
        ax.set_xlabel('$A_T^a$')
        ax.set_ylabel('Payoff')
        ax.set_xlim(left = 0, right=10)
        ax.set_ylim(bottom = 0, top = 2)
        ax.set_title(f'Put payoff vs replication')
        ax.legend()

        plt.tight_layout()
        plt.savefig('replicating_heston_put.png')

if __name__ == '__main__':
    he = HedgeExperiment()
    he.simulate_paths()
    he.replicate_paths()
    he.plot_replication()
