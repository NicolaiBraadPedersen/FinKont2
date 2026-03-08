import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# =============================================================================
# Parameters
# =============================================================================
S0     = 100
r      = 0.03
alpha  = 0.07
sigma  = 0.20
expiry = 1
strike = 100

n  = expiry * 12
dt = expiry / n

u = np.exp(alpha * dt + sigma * np.sqrt(dt))
d = np.exp(alpha * dt - sigma * np.sqrt(dt))
R = np.exp(r * dt)
q = (R - d) / (u - d)

tid   = dt * np.arange(n + 1)          # time grid
index = np.arange(1, n + 2)            # 1-indexed like R

# =============================================================================
# Initialise matrices  (1-indexed in R → 0-indexed in Python, size n+1 × n+1)
# =============================================================================
Smat = np.zeros((n + 1, n + 1))
put  = np.zeros((n + 1, n + 1))
ExBd = -np.ones(n + 1)
ExBd[n] = strike                        # ExBd[n+1] in R → ExBd[n] in Python

# Terminal conditions (column n, i.e. last column)
j_arr         = np.arange(n + 1)
Smat[:, n]    = S0 * u ** j_arr * d ** (n - j_arr)
put[:, n]     = np.maximum(strike - Smat[:, n], 0)

# =============================================================================
# Plot setup
# =============================================================================
xrange = (0, expiry + dt)
yrange = (Smat[:, n].min() - 20, Smat[:, n].max())

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_xlabel("time, t")
ax.set_ylabel("stock price, S(t)")

ax.text(0, yrange[1],        f"{n} steps per year", ha='left', va='top', fontsize=9)
ax.text(0, 0.95 * yrange[1], f"sigma = {sigma:.2f}",  ha='left', va='top', fontsize=9)
ax.text(0, 0.90 * yrange[1], f"alpha = {alpha:.2f}",  ha='left', va='top', fontsize=9)

# =============================================================================
# Backward induction + draw tree
# =============================================================================
for i in range(n - 1, -1, -1):             # i = n-1 down to 0  (R: n down to 1)
    for j in range(i + 1):                 # j = 0..i           (R: j = 1..i)
        S = S0 * u ** j * d ** (i - j)
        Smat[j, i] = S

        hold = (q * put[j + 1, i + 1] + (1 - q) * put[j, i + 1]) / R
        ev   = max(strike - S, 0)
        put[j, i] = max(ev, hold)

        # Update exercise boundary
        if abs(ev - put[j, i]) < 1e-6 and ExBd[i] < S <= strike:
            ExBd[i] = S

        # Draw tree branches
        ax.plot([i * dt, (i + 1) * dt], [Smat[j, i], Smat[j,     i + 1]],
                'k:', lw=0.25)
        ax.plot([i * dt, (i + 1) * dt], [Smat[j, i], Smat[j + 1, i + 1]],
                'k:', lw=0.25)

print(f"The arb' free time 0 put price = {put[0, 0]:.4f}")

# =============================================================================
# Exercise boundary
# =============================================================================
pick = ExBd > 0
ax.plot(tid[pick], ExBd[pick], 'ro-', lw=2, label='Exercise boundary')
ax.axhline(y=strike, color='red', lw=1)

# =============================================================================
# Simulated path
# =============================================================================
np.random.seed(1)
steps_up = np.concatenate([[0], np.cumsum(np.random.uniform(size=n) > 0.68)])
SPath1   = S0 * u ** steps_up * d ** (np.arange(n + 1) - steps_up)
ax.plot(tid, SPath1, 'b-o', lw=3, ms=4, label='Simulated path')

# =============================================================================
# Optimal stopping time
# =============================================================================
delay = 1

matches     = np.where(ExBd == SPath1)[0]
index_opt   = matches.min() if len(matches) > 0 else n
tau_opt     = tid[index_opt]
index_act   = index_opt + delay

ax.text(tid[index_opt],         SPath1[index_opt],         "O", fontsize=12, fontweight='bold', color='blue')
ax.text(tid[index_opt + delay], SPath1[index_opt + delay], "D", fontsize=12, fontweight='bold', color='blue')

ax.axvline(x=tau_opt,              color='gray', lw=1)
ax.axvline(x=tid[index_opt+delay], color='gray', lw=1)
ax.text(tau_opt,              yrange[0], r"$\tau^{opt}$",   ha='right', fontsize=9)
ax.text(tid[index_opt+delay], yrange[0], r"$\tau^{delay}$", ha='left',  fontsize=9)

ax.legend(loc='upper right')
plt.tight_layout()

# =============================================================================
# PnL: Liquidating
# =============================================================================
liquidate   = (strike - SPath1[index_opt]) * np.exp(r * dt * delay)
payout      = (strike - SPath1[index_opt + delay])
PnL_liq     = liquidate - payout
print(f"PnL w/ liquidating = {PnL_liq:.4f}")

# =============================================================================
# PnL: Freezing (static delta hedge)
# =============================================================================
m     = np.where(Smat[:, index_opt] == SPath1[index_opt])[0][0]
delta = (put[m + 1, index_opt] - put[m, index_opt]) / \
        (Smat[m + 1, index_opt] - Smat[m, index_opt])
bank      = put[m, index_opt - 1] - Smat[m, index_opt - 1] * delta
freeze    = delta * SPath1[index_opt + delay] + bank * np.exp(r * dt * (delay + 1))
PnL_freeze = freeze - payout
print(f"PnL w/ freezing = {PnL_freeze:.4f}")

# =============================================================================
# PnL: Dynamic hedging
# =============================================================================
m     = np.where(Smat[:, index_opt] == SPath1[index_opt])[0][0]
delta = (put[m + 1, index_opt] - put[m, index_opt]) / \
        (Smat[m + 1, index_opt] - Smat[m, index_opt])
bank  = put[m, index_opt - 1] - Smat[m, index_opt - 1] * delta

if delay > 0:
    for k in range(1, delay + 1):
        m = np.where(Smat[:, index_opt + k - 1] == SPath1[index_opt + k - 1])[0][0]
        ax.text(tid[index_opt - 1 + k], SPath1[index_opt + k - 1], "X", fontsize=10, color='green')
        Vpf   = Smat[m, index_opt + k - 1] * delta + bank * np.exp(r * dt)
        delta = (put[m + 1, index_opt + k] - put[m, index_opt + k]) / \
                (Smat[m + 1, index_opt + k] - Smat[m, index_opt + k])
        bank  = Vpf - Smat[m, index_opt + k - 1] * delta

Vpf         = SPath1[index_opt + delay] * delta + bank * np.exp(r * dt)
PnL_dynamic = Vpf - payout
print(f"PnL w/ continued dynamic hedging = {PnL_dynamic:.4f}")

path = r'C:\Users\nicol\OneDrive - University of Copenhagen\Desktop\4 år\FinKont2\HandIn2'
plt.savefig(path+r'\binomial_plots.png', dpi=150, bbox_inches='tight')
plt.show()