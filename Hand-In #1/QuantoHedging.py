from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# # below is a personal style for plotting, will only work if the whole repo is cloned
# from utils.plotting import use_earthy_style
# use_earthy_style()

class QuantoPut():
    def __init__(self, hedge_type):
        # all the below values can be edited after initializing the class
        self.r_US = 0.03
        self.r_J = 0.00

        self.sig_X = np.array([0.1, 0.02])
        self.sig_J = np.array([0.0, 0.25])

        self.norm_J = np.sqrt(self.sig_J @ self.sig_J)
        self.dot_XJ = self.sig_X @ self.sig_J
        self.hedge_type = hedge_type

        self.quanto_adj = -self.dot_XJ

        if hedge_type == "3f":
            self.quanto_adj = self.dot_XJ
            self.mu_J = self.r_J + self.dot_XJ

        if hedge_type in ["3b", "3c"]:
            self.mu_X = self.r_US - self.r_J
            self.mu_J = self.r_J - self.dot_XJ

        if hedge_type == "3e":
            self.mu_X = 0
            self.mu_J = 0

        self.S_J0 = 30000.0
        self.K = 30000.0

        self.X0 = 0.01
        self.Y0 = 0.01
        self.capT = 1.0

        self.filepath = ''
        self.has_results = False
        self.has_risk_stats = False

    def F_QP(self, t, s):
        tau = self.capT - t
        K = self.K
        quanto_adj = self.quanto_adj
        norm_J = self.norm_J
        r_J, r_US = self.r_J, self.r_US

        d1 = (np.log(s / K) + (r_J + quanto_adj + 0.5 * norm_J ** 2) * tau) / (np.sqrt(tau) * norm_J)
        d2 = d1 - np.sqrt(tau) * norm_J

        return self.Y0 * np.exp(-r_US * tau) * (
                K * norm.cdf(-d2) - np.exp((r_J + quanto_adj) * tau) * s * norm.cdf(-d1)
        )

    def dF_QPds(self, t, s):
        tau = self.capT - t
        quanto_adj = self.quanto_adj
        norm_J = self.norm_J
        r_J, r_US = self.r_J, self.r_US

        d1 = (np.log(s / self.K) + (r_J + quanto_adj + 0.5 * norm_J ** 2) * tau) / (np.sqrt(tau) * norm_J)

        return self.Y0 * np.exp((r_J + quanto_adj - r_US) * tau) * (norm.cdf(d1) - 1)

    def simulate_hedge(self, m, n):
        self.n = n
        # below is a vectorized version of the dimension m (number of paths simulated)
        S_J = np.full(m, self.S_J0)
        X = np.full(m, self.X0)
        dt = self.capT / n

        Vpf = np.full(m, self.F_QP(0, self.S_J0))
        hS_J = np.full(m, self.dF_QPds(0, self.S_J0) / self.X0)

        if self.hedge_type == "3b":
            hX = np.zeros(m)
        else:
            hX = -hS_J * S_J

        hB_US = Vpf - hS_J * S_J * X - hX * X

        dW_all = np.random.normal(0, np.sqrt(dt), size=(n, m, 2))

        results = np.zeros((m, 3))
        self.InitialOutlay = self.F_QP(0, self.S_J0)
        print("Initial investment =", round(self.InitialOutlay, 4))

        for j in range(n):

            dW = dW_all[j]

            # Update X and S_J for all paths
            X *= np.exp((self.mu_X - 0.5 * self.sig_X @ self.sig_X) * dt + dW @ self.sig_X)
            S_J *= np.exp((self.mu_J - 0.5 * self.sig_J @ self.sig_J) * dt + dW @ self.sig_J)

            # Portfolio value
            Vpf = (
                    hS_J * S_J * X
                    + hX * np.exp(self.r_J * dt) * X
                    + hB_US * np.exp(self.r_US * dt)
            )

            if j < n - 1:
                t = (j + 1) * dt

                hS_J = self.dF_QPds(t, S_J) / X

                if self.hedge_type == "3b":
                    hX = np.zeros(m)
                else:
                    hX = -hS_J * S_J

                hB_US = Vpf - hS_J * S_J * X - hX * X

        results[:, 0] = Vpf
        results[:, 1] = self.Y0 * np.maximum(self.K - S_J, 0)
        results[:, 2] = S_J

        self.results = results
        self.has_results = True

    def get_risk_stats(self):
        self.sd_unhedged = np.exp(-self.r_US * self.capT) * np.std(self.results[:, 1]) / self.InitialOutlay
        self.sd_hedged = np.exp(-self.r_US * self.capT) * np.std(self.results[:, 0] - self.results[:, 1]) / self.InitialOutlay
        self.has_risk_stats = True

    def plot_hedge_vs_actual(self):
        if not self.has_results:
            print('No results are available for plotting. Run self.simulate_hedge!')
            return
        if not self.has_risk_stats:
            print('No risk stats are available for plotting. Run self.get_risk_stats!')
            return

        results = self.results

        dum = np.arange(0, int(1 + np.floor(np.max(results[:, 2]))))

        plt.figure()
        plt.scatter(results[:, 2], results[:, 0], s=10, alpha=0.4, label = 'Value of Hedge')
        plt.plot(dum, self.Y0 * np.maximum(self.K - dum, 0), linestyle = 'dashed', color="#724E25", label = 'Option Payoff')

        plt.xlabel("S_J(T)")
        plt.ylabel("V(T)")
        plt.title(f'{self.hedge_type} | n = {self.n}')

        plt.text(
            np.max(results[:, 2]),
            self.Y0 * self.K,
            f"sd(hedged)/sd(unhedged)={round(self.sd_hedged / self.sd_unhedged, 3)}",
            ha="right",
            fontsize=10,
        )

        plt.text(
            np.max(results[:, 2]),
            0.9 * self.Y0 * self.K,
            f"#hedge points/year={round(self.n / self.capT, 3)}",
            ha="right",
            fontsize=10,
        )

        plt.legend()

        if self.filepath:
            plt.savefig(self.filepath + '\\' + self.hedge_type + f'_n={self.n}' + '_plot_hedge_vs_actual.png')
        #plt.show()


if __name__ == '__main__':
    qp = [QuantoPut(hedge_type=x) for x in ['3b', '3c', '3e']]

    for n in [52,252,252*3,252*10]:
        for i in range(3):
            qp[i].simulate_hedge(m = 1000, n = n)
            qp[i].get_risk_stats()
            qp[i].plot_hedge_vs_actual()
