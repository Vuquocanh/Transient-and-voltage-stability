"""
46711 Assignment 2 - Sections 1.2.2 and 1.2.3
Figure 1: P-delta curves (pre-fault, during fault, post-fault)
Figure 2: Equal-area criterion with acceleration/deceleration areas and delta_cr
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters (Tables 1-3) ----------------
X_tr = 0.6            # transformer reactance [pu]
X_1 = 1.0             # TL1 [pu]
X_21, X_22, X_23 = 2/12, 5/12, 3/12   # TL2 sections [pu]
Xd_p = 0.3            # transient reactance X'd [pu]
E_B = 1.0             # infinite bus voltage [pu]
E_t = 1.0 * np.exp(1j * np.deg2rad(25))   # terminal voltage (pre-fault) [pu]

# ---------------- Pre-fault operating point (1.2.1) ----------------
X_2 = X_21 + X_22 + X_23
X_line = X_1 * X_2 / (X_1 + X_2)          # TL1 || TL2
X_tE = X_tr + X_line                      # terminal -> infinite bus
I = (E_t - E_B) / (1j * X_tE)             # line current
E_p = E_t + 1j * Xd_p * I                 # internal transient voltage E'
Ep_mag = abs(E_p)
delta0 = np.angle(E_p)                    # [rad]
Pm = (E_t * np.conj(I)).real              # mechanical power = Pe pre-fault

# ---------------- Transfer reactances and Pe,max (1.2.2) ----------------
X_a = Xd_p + X_tr
X_pre = X_a + X_line                          # both lines in service
X_dur = X_a + X_1 + X_a * X_1 / (X_21 + X_22) # star-delta, fault at B9
X_post = X_a + X_1                            # TL2 disconnected

Pmax = {
    "Pre-fault": Ep_mag * E_B / X_pre,
    "During fault": Ep_mag * E_B / X_dur,
    "Post-fault": Ep_mag * E_B / X_post,
}

# ---------------- Critical clearing angle (1.2.3) ----------------
P1, P2 = Pmax["During fault"], Pmax["Post-fault"]
delta_max = np.pi - np.arcsin(Pm / P2)
delta_cr = np.arccos((Pm * (delta_max - delta0) + P2 * np.cos(delta_max)
                      - P1 * np.cos(delta0)) / (P2 - P1))
A_acc = Pm * (delta_cr - delta0) + P1 * (np.cos(delta_cr) - np.cos(delta0))
A_dec = P2 * (np.cos(delta_cr) - np.cos(delta_max)) - Pm * (delta_max - delta_cr)

print(f"E' = {Ep_mag:.3f} pu, delta0 = {np.rad2deg(delta0):.2f} deg, Pm = {Pm:.3f} pu")
for name, X in zip(Pmax, (X_pre, X_dur, X_post)):
    print(f"{name:13s}: X_T = {X:.3f} pu, Pe,max = {Pmax[name]:.3f} pu")
print(f"delta_max = {np.rad2deg(delta_max):.2f} deg, delta_cr = {np.rad2deg(delta_cr):.2f} deg")
print(f"A_acc = {A_acc:.4f}, A_dec = {A_dec:.4f} pu*rad")

# ---------------- Plotting helpers ----------------
delta = np.linspace(0, np.pi, 500)
colors = {"Pre-fault": "tab:blue", "During fault": "tab:red", "Post-fault": "tab:green"}


def draw_curves(ax):
    """Draw the three P-delta curves and the Pm line on ax."""
    for name, P in Pmax.items():
        ax.plot(np.rad2deg(delta), P * np.sin(delta), color=colors[name],
                label=rf"{name}: $P_{{e,max}}$ = {P:.3f} pu")
    ax.axhline(Pm, color="k", linestyle="--", linewidth=1,
               label=rf"$P_m$ = {Pm:.3f} pu")
    ax.set_xlabel(r"Rotor angle $\delta$ [deg]")
    ax.set_ylabel(r"Electrical power $P_e$ [pu]")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.1 * max(Pmax.values()))
    ax.set_xticks(range(0, 181, 30))
    ax.grid(alpha=0.3)


# ---------------- Figure 1: P-delta curves (1.2.2) ----------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
draw_curves(ax1)
ax1.set_title(r"$P$-$\delta$ curves")
ax1.legend(loc="upper right")
fig1.tight_layout()
fig1.savefig("p_delta_curves.png", dpi=300)

# ---------------- Figure 2: Equal-area criterion (1.2.3) ----------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
draw_curves(ax2)

d_acc = np.linspace(delta0, delta_cr, 200)
d_dec = np.linspace(delta_cr, delta_max, 200)
ax2.fill_between(np.rad2deg(d_acc), P1 * np.sin(d_acc), Pm,
                 color="tab:red", alpha=0.25, label=rf"$A_\mathrm{{acc}}$ = {A_acc:.3f}")
ax2.fill_between(np.rad2deg(d_dec), Pm, P2 * np.sin(d_dec),
                 color="tab:green", alpha=0.25, label=rf"$A_\mathrm{{dec}}$ = {A_dec:.3f}")

for d, lbl in [(delta0, r"$\delta_0$"), (delta_cr, r"$\delta_{cr}$"),
               (delta_max, r"$\delta_{max}$")]:
    ax2.axvline(np.rad2deg(d), color="gray", linestyle=":", linewidth=1)
    ax2.text(np.rad2deg(d) + 1, 0.02, lbl)

ax2.set_title(rf"Equal-area criterion: $\delta_{{cr}}$ = {np.rad2deg(delta_cr):.1f}$^\circ$")
ax2.legend(loc="upper right", fontsize=9)
fig2.tight_layout()
fig2.savefig("equal_area_criterion.png", dpi=300)

plt.show()
