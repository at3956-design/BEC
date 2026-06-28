"""
Extended Gross-Pitaevskii equation (eGPE) simulation of NaCs molecular BEC
time-of-flight (TOF) expansion — Will group, Columbia University.

Physics:
  i*hbar d/dt psi = [-hbar^2/(2M) nabla^2 + V_ext + g|psi|^2 + Phi_dd] psi

  - Contact interaction:  g = 4*pi*hbar^2*a_s/M
  - Dipolar mean-field:   Phi_dd(r) = integral V_dd(r-r') n(r') d^3r'
                          V_dd = (C_dd/4pi)(1-3cos^2 theta)/r^3
  - Computed via FFT convolution: Phi_dd(k) = n(k) * V_dd_hat(k)
                          V_dd_hat(k) = (C_dd/3)(3*k_z^2/k^2 - 1)

  Set a_dd = 0 to recover pure contact GPE.

Units: harmonic oscillator units along x
  length  : a_ho = sqrt(hbar / M / omega_x)
  energy  : hbar * omega_x
  time    : 1 / omega_x

Parameters match the NaCs BEC (Bigagli et al., Nature 631, 289 (2024)):
  - a_s ~ 1500 a0, a_dd ~ 1300 a0  (at microwave compensation point)
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"   # Metal too experimental; CPU gives float64 + JIT

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

jax.config.update("jax_enable_x64", True)   # float64 precision on CPU
print(f"JAX backend: {jax.default_backend()}")

# ── Physical constants ────────────────────────────────────────────────────────
hbar = 1.0546e-34
a0   = 5.292e-11

# ── NaCs molecular parameters ─────────────────────────────────────────────────
M_NaCs = (23 + 133) * 1.6605e-27

# ── Trap frequencies ──────────────────────────────────────────────────────────
freq_x_Hz = 10.0
freq_y_Hz = 10.0
freq_z_Hz = 50.0   # tightest axis

omega_x = 2 * np.pi * freq_x_Hz
omega_y = 2 * np.pi * freq_y_Hz
omega_z = 2 * np.pi * freq_z_Hz

a_ho      = np.sqrt(hbar / (M_NaCs * omega_x))
t_unit_ms = 1e3 / omega_x

wx = 1.0
wy = omega_y / omega_x
wz = omega_z / omega_x

# ── Interaction parameters ────────────────────────────────────────────────────
N_mol = 2000
a_s   = 1500.0 * a0   # set a_dd=0 below for pure contact
a_dd  = 1300.0 * a0

eps_dd    = a_dd / a_s
g_contact = 4 * np.pi * (a_s / a_ho)
g_dipolar = 4 * np.pi * (a_dd / a_ho)

# LHY beyond-mean-field correction (Wächtler & Santos PRA 2016)
# μ_LHY = g_lhy * (N|ψ|²)^(3/2),  f(ε_dd) integrates over Bogoliubov angles
from scipy.integrate import quad
def _lhy_f(eps):
    val, _ = quad(lambda u: np.real((1 + eps*(3*u**2 - 1))**2.5), 0, 1)
    return val

f_edd     = _lhy_f(eps_dd)
g_lhy     = (128 / (3 * np.sqrt(np.pi))) * 4 * np.pi * (a_s / a_ho)**2.5 * f_edd

print(f"NaCs eGPE:  trap ({freq_x_Hz},{freq_y_Hz},{freq_z_Hz}) Hz  "
      f"|  a_ho = {a_ho*1e6:.2f} µm  |  ε_dd = {eps_dd:.2f}")
print(f"LHY:  f(ε_dd) = {f_edd:.4f}  |  g_lhy = {g_lhy:.4e}")

# ── Grid ──────────────────────────────────────────────────────────────────────
Nx, Ny, Nz = 64, 64, 128
Lx, Ly, Lz = 12.0, 12.0, 10.0

x = np.linspace(-Lx, Lx, Nx, endpoint=False)
y = np.linspace(-Ly, Ly, Ny, endpoint=False)
z = np.linspace(-Lz, Lz, Nz, endpoint=False)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
dV = dx * dy * dz

print(f"Grid: {Nx}×{Ny}×{Nz}  |  dz = {dz*a_ho*1e6:.2f} µm")

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

with np.errstate(invalid='ignore', divide='ignore'):
    dipolar_kernel = np.where(K2 > 0, (3 * KZ**2 / K2 - 1) / 3.0, 0.0)

# Move everything to JAX arrays
X_j  = jnp.array(X);  Y_j = jnp.array(Y);  Z_j = jnp.array(Z)
K2_j = jnp.array(K2)
V_trap_j = jnp.array(0.5 * (wx**2 * X**2 + wy**2 * Y**2 + wz**2 * Z**2))
dk_j     = jnp.array(dipolar_kernel)

# ── JAX-compiled propagators ─────────────────────────────────────────────────
@jit
def kinetic_prop(dt):
    return jnp.exp(-0.5j * 0.5 * K2_j * dt)

@jit
def dipolar_potential(psi):
    n   = jnp.abs(psi)**2
    n_k = jnp.fft.fftn(n)
    return jnp.fft.ifftn(N_mol * g_dipolar * dk_j * n_k).real

@jit
def potential_step(psi, dt, V_ext):
    n        = jnp.abs(psi)**2
    Phi_dd   = dipolar_potential(psi)
    V_lhy    = g_lhy * (N_mol * n)**1.5   # LHY quantum pressure ~ n^(3/2)
    phase    = (V_ext + N_mol * g_contact * n + Phi_dd + V_lhy) * dt
    return psi * jnp.exp(-1j * phase)

@jit
def normalize(psi):
    return psi / jnp.sqrt(jnp.sum(jnp.abs(psi)**2) * dV)

@jit
def ssfm_step(psi, dt, V_ext, kp):
    psi = jnp.fft.ifftn(jnp.fft.fftn(psi) * kp)
    psi = potential_step(psi, dt, V_ext)
    psi = jnp.fft.ifftn(jnp.fft.fftn(psi) * kp)
    return psi

def chemical_potential(psi):
    kin    = jnp.fft.ifftn(0.5 * K2_j * jnp.fft.fftn(psi))
    Phi_dd = dipolar_potential(psi)
    n      = jnp.abs(psi)**2
    return float(jnp.real(jnp.sum(jnp.conj(psi) * (
        kin + (V_trap_j + N_mol * g_contact * n + Phi_dd) * psi
    )) * dV))

# ── 1. Ground state via imaginary-time propagation ────────────────────────────
print("\nFinding ground state...")

dt_imag  = -1j * 0.01
kp_imag  = kinetic_prop(dt_imag)

psi = jnp.array(
    np.exp(-(X**2/(2*1.5**2) + Y**2/(2*0.8**2) + Z**2/(2*0.6**2))).astype(complex)
)
psi = normalize(psi)

mu_prev = None
for step in range(3000):
    psi = ssfm_step(psi, dt_imag, V_trap_j, kp_imag)
    psi = normalize(psi)
    if step % 500 == 0:
        mu = chemical_potential(psi)
        print(f"  step {step:4d}  μ = {mu:.4f} ℏω_x")
        if mu_prev is not None and abs(mu - mu_prev) < 1e-6:
            print(f"  converged early at step {step}")
            break
        mu_prev = mu

print("Ground state found.\n")
psi_gs = psi

# ── 2. Real-time TOF expansion ────────────────────────────────────────────────
print("Running TOF expansion...")

dt_real = 0.005
kp_real = kinetic_prop(dt_real)

save_times_ms = [0, 5, 10, 15, 20]
save_times    = [t_ms / t_unit_ms for t_ms in save_times_ms]
save_steps    = {int(round(ts / abs(dt_real))): t_ms
                 for ts, t_ms in zip(save_times, save_times_ms)}
n_tof_steps   = max(save_steps.keys())

snapshots                          = {}
waist_times_ms                     = []
waist_x, waist_y, waist_z         = [], [], []

def rms_waist(n3d, coord):
    norm = float(jnp.sum(n3d)) * dV
    mean = float(jnp.sum(n3d * coord)) * dV / norm
    return float(jnp.sqrt(jnp.sum(n3d * (coord - mean)**2) * dV / norm)) * a_ho * 1e6

V_off = jnp.zeros_like(V_trap_j)
psi   = psi_gs

for step in range(n_tof_steps + 1):
    n3d = jnp.abs(psi)**2
    if step in save_steps:
        t_ms = save_steps[step]
        snapshots[t_ms] = np.array(n3d)
        print(f"  saved snapshot at t = {t_ms} ms")
    waist_times_ms.append(step * abs(dt_real) * t_unit_ms)
    waist_x.append(rms_waist(n3d, X_j))
    waist_y.append(rms_waist(n3d, Y_j))
    waist_z.append(rms_waist(n3d, Z_j))
    if step < n_tof_steps:
        psi = ssfm_step(psi, dt_real, V_off, kp_real)

# ── 3. Density snapshot plots ─────────────────────────────────────────────────
print("\nPlotting...")

x_um = x * a_ho * 1e6
y_um = y * a_ho * 1e6
z_um = z * a_ho * 1e6

fig = plt.figure(figsize=(16, 8))
fig.suptitle(
    f"NaCs eGPE TOF  |  trap ({freq_x_Hz},{freq_y_Hz},{freq_z_Hz}) Hz  "
    f"|  N={N_mol}  |  ε_dd={eps_dd:.2f}",
    fontsize=12
)
gs = GridSpec(2, len(save_times_ms), figure=fig, hspace=0.4, wspace=0.3)

vmax_xy = max(snapshots[t].sum(axis=2).max() * dz for t in save_times_ms)
vmax_xz = max(snapshots[t].sum(axis=1).max() * dy for t in save_times_ms)

for col, t_snap in enumerate(save_times_ms):
    n3d = snapshots[t_snap]
    ax = fig.add_subplot(gs[0, col])
    ax.imshow((n3d.sum(axis=2) * dz).T, origin='lower', aspect='equal',
              extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
              cmap='Blues', vmin=0, vmax=vmax_xy)
    ax.set_title(f"t = {t_snap} ms", fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    if col == 0: ax.set_ylabel("y (µm)", fontsize=8)

    ax2 = fig.add_subplot(gs[1, col])
    ax2.imshow((n3d.sum(axis=1) * dy).T, origin='lower', aspect='equal',
               extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
               cmap='Blues', vmin=0, vmax=vmax_xz)
    ax2.set_xlabel("x (µm)", fontsize=8)
    if col == 0: ax2.set_ylabel("z (µm)", fontsize=8)

fig.text(0.01, 0.73, "XY\nview", va='center', ha='left', fontsize=9)
fig.text(0.01, 0.27, "XZ\nview", va='center', ha='left', fontsize=9)
plt.savefig("tof_expansion.png", dpi=150, bbox_inches='tight')
print("Saved: tof_expansion.png")
plt.show(block=False)

# ── 4. Waist evolution + Castin-Dum theory ───────────────────────────────────
from scipy.integrate import solve_ivp

def castin_dum(t, y):
    bx, by, bz, dbx, dby, dbz = y
    vol = bx * by * bz
    return [dbx, dby, dbz,
            wx**2 / (bx * vol),
            wy**2 / (by * vol),
            wz**2 / (bz * vol)]

t_max_ho = max(waist_times_ms) / t_unit_ms
t_cd     = np.linspace(0, t_max_ho, 500)
sol      = solve_ivp(castin_dum, [0, t_max_ho], [1, 1, 1, 0, 0, 0],
                     t_eval=t_cd, rtol=1e-10, atol=1e-12)
bx_cd, by_cd, bz_cd = sol.y[0], sol.y[1], sol.y[2]
t_cd_ms  = t_cd * t_unit_ms

# σ_i(t) = σ_i(0) * b_i(t)
sx0, sy0, sz0 = waist_x[0], waist_y[0], waist_z[0]

fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(waist_times_ms, waist_x, color='steelblue',      label=f"eGPE x  ({freq_x_Hz} Hz)")
ax.plot(waist_times_ms, waist_y, color='cornflowerblue', label=f"eGPE y  ({freq_y_Hz} Hz)", ls='--')
ax.plot(waist_times_ms, waist_z, color='tomato',         label=f"eGPE z  ({freq_z_Hz} Hz)")
ax.plot(t_cd_ms, sx0 * bx_cd, color='steelblue',      ls=':', lw=2, alpha=0.7, label="C-D x")
ax.plot(t_cd_ms, sy0 * by_cd, color='cornflowerblue', ls=':', lw=2, alpha=0.7, label="C-D y")
ax.plot(t_cd_ms, sz0 * bz_cd, color='tomato',         ls=':', lw=2, alpha=0.7, label="C-D z")
ax.set_xlabel("Time of flight (ms)", fontsize=12)
ax.set_ylabel("RMS waist (µm)", fontsize=12)
ax.set_title(f"NaCs BEC waist evolution  (ε_dd={eps_dd:.2f},  N={N_mol})", fontsize=12)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig("tof_waists.png", dpi=150, bbox_inches='tight')
print("Saved: tof_waists.png")
plt.show(block=False)

# ── 5. Aspect ratio ───────────────────────────────────────────────────────────
aspect_zx = np.array(waist_z) / np.array(waist_x)
aspect_zy = np.array(waist_z) / np.array(waist_y)

# C-D gives scaling factors b_i(0)=1; multiply by initial aspect ratio from simulation
ar0_zx = aspect_zx[0]
ar0_zy = aspect_zy[0]
cd_zx  = ar0_zx * bz_cd / bx_cd
cd_zy  = ar0_zy * bz_cd / by_cd

def find_inversion(ratio, times):
    """Return time when ratio crosses 1 from below, or None if it never does."""
    idx = np.where(np.diff(np.sign(ratio - 1.0)) > 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return float(np.interp(1.0, ratio[i:i+2], times[i:i+2]))

t_inv_sim_zx = find_inversion(aspect_zx, waist_times_ms)
t_inv_sim_zy = find_inversion(aspect_zy, waist_times_ms)
t_inv_cd_zx  = find_inversion(cd_zx,    t_cd_ms)
t_inv_cd_zy  = find_inversion(cd_zy,    t_cd_ms)

print(f"\nAspect ratio inversion times:")
for label, t in [("eGPE σ_z/σ_x", t_inv_sim_zx), ("eGPE σ_z/σ_y", t_inv_sim_zy),
                 ("C-D  σ_z/σ_x", t_inv_cd_zx),  ("C-D  σ_z/σ_y", t_inv_cd_zy)]:
    print(f"  {label}: {f'{t:.2f} ms' if t is not None else 'not reached'}")

fig3, ax = plt.subplots(figsize=(7, 5))
ax.plot(waist_times_ms, aspect_zx, color='tomato',      label="eGPE  σ_z / σ_x")
ax.plot(waist_times_ms, aspect_zy, color='darkorange',  label="eGPE  σ_z / σ_y", ls='--')
ax.plot(t_cd_ms, cd_zx, color='tomato',    label="C-D theory  b_z / b_x", ls=':', lw=2, alpha=0.7)
ax.plot(t_cd_ms, cd_zy, color='darkorange', label="C-D theory  b_z / b_y", ls=':', lw=2, alpha=0.7)
ax.axhline(1.0, color='gray', lw=1, ls=':', label="aspect ratio = 1")
if t_inv_sim_zx is not None:
    ax.axvline(t_inv_sim_zx, color='tomato', lw=1, ls='--', alpha=0.5,
               label=f"inv sim z/x  {t_inv_sim_zx:.1f} ms")
if t_inv_cd_zx is not None:
    ax.axvline(t_inv_cd_zx,  color='tomato', lw=1, ls=':',  alpha=0.5,
               label=f"inv C-D z/x  {t_inv_cd_zx:.1f} ms")
if t_inv_sim_zy is not None:
    ax.axvline(t_inv_sim_zy, color='darkorange', lw=1, ls='--', alpha=0.5,
               label=f"inv sim z/y  {t_inv_sim_zy:.1f} ms")
if t_inv_cd_zy is not None:
    ax.axvline(t_inv_cd_zy,  color='darkorange', lw=1, ls=':',  alpha=0.5,
               label=f"inv C-D z/y  {t_inv_cd_zy:.1f} ms")
ax.set_xlabel("Time of flight (ms)", fontsize=12)
ax.set_ylabel("Aspect ratio", fontsize=12)
ax.set_title(f"Aspect ratio inversion  (ε_dd={eps_dd:.2f},  N={N_mol})\n"
             f"trap ({freq_x_Hz},{freq_y_Hz},{freq_z_Hz}) Hz", fontsize=11)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
fig3.tight_layout()
plt.savefig("tof_aspect_ratio.png", dpi=150, bbox_inches='tight')
print("Saved: tof_aspect_ratio.png")
plt.show(block=True)
