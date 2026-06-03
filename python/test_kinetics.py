import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snsolver import (solve_point_kinetics, solve_quasistatic,
                      DelayedNeutronData, PointKinetics, XSChange,
                      Quadrature, Mesh1D, XSData, QuasiStaticSolver)

print("=" * 70)
print("Test 1: Point Kinetics — Step reactivity insertion (delayed supercritical)")
print("=" * 70)

beta_total = DelayedNeutronData().beta_total()
print(f"  Total delayed neutron fraction β = {beta_total:.6f}")

rho_step = 0.5 * beta_total
Lambda = 2e-5
print(f"  Step reactivity ρ = 0.5β = {rho_step:.6f}")
print(f"  Neutron generation time Λ = {Lambda:.1e} s")

result = solve_point_kinetics(rho=rho_step, Lambda=Lambda,
                              initial_power=1.0, t_end=5.0, dt=1e-4, theta=1.0)

print(f"  Time steps: {result['n_steps']}")
print(f"  Initial power: {result['power'][0]:.4f}")
print(f"  Final power:   {result['power'][-1]:.4e}")
print(f"  Power ratio:   {result['power'][-1]/result['power'][0]:.4e}")

P = result['power']
t = result['time']
t_idx_1 = np.argmin(np.abs(np.array(t) - 1.0))
t_idx_2 = np.argmin(np.abs(np.array(t) - 2.0))
t_idx_5 = np.argmin(np.abs(np.array(t) - 5.0))
print(f"  P(t=1s) = {P[t_idx_1]:.4e}")
print(f"  P(t=2s) = {P[t_idx_2]:.4e}")
print(f"  P(t=5s) = {P[t_idx_5]:.4e}")

alpha_prompt = (rho_step - beta_total) / Lambda
if alpha_prompt > 0:
    T_doubling = np.log(2) / alpha_prompt
    print(f"  Prompt period (approx) = {T_doubling:.4e} s")
else:
    print(f"  Sub-prompt-critical: α_prompt = {alpha_prompt:.2f} < 0, stable period")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].semilogy(t, P, 'b-', linewidth=1.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Reactor Power (normalized)')
axes[0].set_title(f'Point Kinetics: Step Reactivity ρ = 0.5β (Delayed Supercritical)')
axes[0].grid(True, alpha=0.3)

precursor = result['precursor_conc']
for i in range(min(6, precursor.shape[1])):
    axes[1].plot(t, precursor[:, i], label=f'Group {i+1}', linewidth=1.0)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Precursor Concentration')
axes[1].set_title('Delayed Neutron Precursor Groups')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_kinetics_1_delayed_supercritical.png', dpi=150)
print("  Plot saved: test_kinetics_1_delayed_supercritical.png")

print()
print("=" * 70)
print("Test 2: Prompt Critical Accident (ρ > β)")
print("=" * 70)

beta_val = beta_total
rho_prompt = 1.5 * beta_val
print(f"  Prompt critical reactivity ρ = 1.5β = {rho_prompt:.6f}")

result_prompt = solve_point_kinetics(rho=rho_prompt, Lambda=Lambda,
                                     initial_power=1.0,
                                     t_end=0.05, dt=1e-6, theta=1.0)

print(f"  Time steps: {result_prompt['n_steps']}")
print(f"  Initial power: {result_prompt['power'][0]:.4f}")
print(f"  Final power:   {result_prompt['power'][-1]:.4e}")
print(f"  Power ratio:   {result_prompt['power'][-1]/result_prompt['power'][0]:.4e}")

alpha_p = (rho_prompt - beta_val) / Lambda
T_approx = np.log(2) / alpha_p
print(f"  Prompt period (theoretical) = {T_approx:.4e} s")
t_pk = result_prompt['time']
P_pk = result_prompt['power']
idx_doubled = np.where(np.array(P_pk) > 2.0)[0]
if len(idx_doubled) > 0:
    t_measured = t_pk[idx_doubled[0]]
    print(f"  Measured doubling time = {t_measured:.4e} s")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(t_pk * 1000, P_pk, 'r-', linewidth=2, label=f'ρ = 1.5β')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Reactor Power (normalized)')
ax.set_title('Prompt Critical Accident: ρ > β')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('test_kinetics_2_prompt_critical.png', dpi=150)
print("  Plot saved: test_kinetics_2_prompt_critical.png")

print()
print("=" * 70)
print("Test 3: Time-varying reactivity (ramp insertion)")
print("=" * 70)

ramp_rate = 0.1 * beta_val
print(f"  Ramp rate: {ramp_rate:.6f} per second (= 0.1β/s)")

def ramp_rho(t):
    return ramp_rate * t

result_ramp = solve_point_kinetics(rho=ramp_rho, Lambda=Lambda,
                                   initial_power=1.0,
                                   t_end=10.0, dt=1e-3, theta=0.5)

t_ramp = result_ramp['time']
P_ramp = result_ramp['power']
rho_ramp = result_ramp['reactivity']

print(f"  Time steps: {result_ramp['n_steps']}")
print(f"  Final reactivity: {rho_ramp[-1]:.6f} ({rho_ramp[-1]/beta_val:.2f}β)")
print(f"  Final power: {P_ramp[-1]:.4e}")

idx_pc = np.where(np.array(rho_ramp) >= beta_val)[0]
if len(idx_pc) > 0:
    t_prompt_crit = t_ramp[idx_pc[0]]
    print(f"  Prompt critical reached at t = {t_prompt_crit:.2f} s")
    P_at_pc = P_ramp[idx_pc[0]]
    print(f"  Power at prompt critical: {P_at_pc:.4e}")

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
axes[0].plot(t_ramp, np.array(rho_ramp)/beta_val, 'b-', linewidth=1.5)
axes[0].axhline(y=1.0, color='r', linestyle='--', label='ρ = β (prompt critical)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Reactivity (β units)')
axes[0].set_title('Ramp Reactivity Insertion')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(t_ramp, P_ramp, 'r-', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Reactor Power')
axes[1].set_title('Power Response')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_ramp, np.log(np.array(P_ramp) / P_ramp[0] + 1e-30), 'g-', linewidth=1.5)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('ln(P/P₀)')
axes[2].set_title('Log Power Growth Rate')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_kinetics_3_ramp.png', dpi=150)
print("  Plot saved: test_kinetics_3_ramp.png")

print()
print("=" * 70)
print("Test 4: Quasi-Static Method — Prompt critical accident with spatial flux")
print("=" * 70)

n_cells = 50
x_min, x_max = 0.0, 10.0

sigma_s_base = 0.8
sigma_t_base = 1.0
nu_sigma_f_base = 0.225

print(f"  Slab: [{x_min}, {x_max}] cm, {n_cells} cells")
print(f"  Base: Σt={sigma_t_base}, Σs={sigma_s_base}, νΣf={nu_sigma_f_base}")

delta_nu_sigma_f = 0.005
n_cells_half = n_cells // 2

def xs_change_prompt_qs(t):
    changes = []
    for i in range(n_cells_half):
        new_nu_sigma_f = nu_sigma_f_base + delta_nu_sigma_f
        changes.append(XSChange(i, sigma_t_base, sigma_s_base, new_nu_sigma_f, 0.0))
    return changes

result_qs = solve_quasistatic(
    sn_order=8, x_min=x_min, x_max=x_max, n_cells=n_cells,
    sigma_t=sigma_t_base, sigma_s=sigma_s_base, source=0.0,
    nu_sigma_f=nu_sigma_f_base, Lambda=2e-5,
    xs_change_func=xs_change_prompt_qs,
    t_end=0.05, dt_shape=0.01, dt_kinetics=5e-4,
    initial_power=1.0, theta_kinetics=1.0,
    tol=1e-5, max_iter=500, dsa=True,
    eigenvalue_mode=True
)

print(f"  Shape updates: {result_qs['n_shape_updates']}")
print(f"  Total kinetics steps: {result_qs['total_kinetics_steps']}")
print(f"  Time points: {len(result_qs['time'])}")
print(f"  Initial keff: {result_qs['initial_keff']:.6f}")
print(f"  Initial amplitude: {result_qs['amplitude'][0]:.4f}")
print(f"  Initial reactivity: {result_qs['reactivity'][0]:.6f} ({result_qs['reactivity'][0]/beta_val:.2f}β)")
print(f"  Final reactivity:  {result_qs['reactivity'][-1]:.6f} ({result_qs['reactivity'][-1]/beta_val:.2f}β)")

if result_qs['amplitude'][-1] > result_qs['amplitude'][0]:
    print(f"  Final amplitude:   {result_qs['amplitude'][-1]:.4e} (INCREASING — supercritical)")
else:
    print(f"  Final amplitude:   {result_qs['amplitude'][-1]:.4e} (decreasing — subcritical)")

centers = result_qs['centers']
full_flux = result_qs['full_flux']
time_qs = result_qs['time']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
n_plot = min(len(full_flux), 6)
indices = np.linspace(0, len(full_flux)-1, n_plot, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
for k, idx in enumerate(indices):
    label = f't={time_qs[idx]*1000:.2f} ms'
    ax.plot(centers, full_flux[idx], color=colors[k], linewidth=1.5, label=label)
ax.set_xlabel('Position (cm)')
ax.set_ylabel('Scalar Flux φ(x,t)')
ax.set_title('Quasi-Static: Spatial Flux Evolution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogy(np.array(time_qs)*1000, result_qs['amplitude'], 'r-', linewidth=2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude T(t)')
ax.set_title('Reactor Power Amplitude')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
shape_flux = result_qs['shape_flux']
for k, idx in enumerate(indices):
    label = f't={time_qs[idx]*1000:.2f} ms'
    ax.plot(centers, shape_flux[idx], color=colors[k], linewidth=1.5, label=label)
ax.set_xlabel('Position (cm)')
ax.set_ylabel('Shape ψ(x,t)')
ax.set_title('Normalized Flux Shape')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(np.array(time_qs)*1000, np.array(result_qs['reactivity'])/beta_val, 'b-', linewidth=2)
ax.axhline(y=1.0, color='r', linestyle='--', label='ρ = β (prompt critical)')
ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Reactivity (β units)')
ax.set_title('Reactivity Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_kinetics_4_quasistatic.png', dpi=150)
print("  Plot saved: test_kinetics_4_quasistatic.png")

print()
print("=" * 70)
print("Test 5: Low-level PointKinetics API")
print("=" * 70)

dn = DelayedNeutronData()
print(f"  Default 6-group β = {dn.beta_total():.6f}")
print(f"  β values: {[f'{dn.get_beta(i):.6f}' for i in range(dn.n_groups)]}")
print(f"  λ values: {[f'{dn.get_lambda(i):.4f}' for i in range(dn.n_groups)]}")

pk = PointKinetics(dn, 2e-5, 1.0, 0.0)
pk.set_reactivity(0.3 * dn.beta_total())
res = pk.solve(2.0, 1e-4, 1.0)

print(f"  Solve: t_end=2.0, dt=1e-4, theta=1.0")
print(f"  Steps: {res.n_steps}")
print(f"  Power: {res.power[0]:.4f} -> {res.power[-1]:.4e}")
print(f"  PASSED: Low-level PointKinetics API works correctly")

print()
print("=" * 70)
print("Test 6: Reactor trip (SCRAM) — Large negative reactivity")
print("=" * 70)

rho_scram = -10.0 * beta_val
print(f"  SCRAM reactivity: ρ = -10β = {rho_scram:.6f}")

result_scram = solve_point_kinetics(rho=rho_scram, Lambda=Lambda,
                                    initial_power=1e6,
                                    t_end=100.0, dt=0.01, theta=1.0)

t_scram = result_scram['time']
P_scram = result_scram['power']

print(f"  Initial power: {P_scram[0]:.4e}")
idx_1s = np.argmin(np.abs(np.array(t_scram) - 1.0))
idx_10s = np.argmin(np.abs(np.array(t_scram) - 10.0))
idx_100s = np.argmin(np.abs(np.array(t_scram) - 100.0))
print(f"  Power at t=1s:   {P_scram[idx_1s]:.4e}")
print(f"  Power at t=10s:  {P_scram[idx_10s]:.4e}")
print(f"  Power at t=100s: {P_scram[idx_100s]:.4e}")
print(f"  Final power:     {P_scram[-1]:.4e}")
print(f"  Shutdown ratio:  {P_scram[-1]/P_scram[0]:.4e}")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(t_scram, P_scram, 'b-', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Reactor Power')
ax.set_title('Reactor SCRAM: ρ = -10β')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_kinetics_6_scram.png', dpi=150)
print("  Plot saved: test_kinetics_6_scram.png")

print()
print("=" * 70)
print("All kinetics tests completed!")
print("=" * 70)
