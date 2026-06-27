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

Units: harmonic oscillator units along x
  length  : a_ho = sqrt(hbar / M / omega_x)
  energy  : hbar * omega_x
  time    : 1 / omega_x

Parameters match the NaCs BEC (Bigagli et al., Nature 631, 289 (2024)):
  - Final trap: omega/(2pi) = (23, 49, 58) Hz
  - N ~ 2000 molecules at BEC onset
  - a_s ~ 1500 a0, a_dd ~ 1300 a0  (at microwave compensation point)
  - TOF = 17 ms for imaging
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Physical constants ────────────────────────────────────────────────────────
hbar  = 1.0546e-34   # J·s
a0    = 5.292e-11    # Bohr radius (m)

# ── NaCs molecular parameters ─────────────────────────────────────────────────
M_NaCs   = (23 + 133) * 1.6605e-27   # kg, NaCs mass
d0_NaCs  = 4.75 * 3.336e-30          # C·m, permanent electric dipole moment

# ── Trap frequencies (final BEC trap, Bigagli et al. Nature 2024) ─────────────
freq_x_Hz = 23.0    # Hz
freq_y_Hz = 49.0    # Hz
freq_z_Hz = 58.0    # Hz  (tightest axis, gravity along -z)

omega_x = 2 * np.pi * freq_x_Hz   # rad/s
omega_y = 2 * np.pi * freq_y_Hz
omega_z = 2 * np.pi * freq_z_Hz

# Harmonic oscillator length along x (our length unit)
a_ho = np.sqrt(hbar / (M_NaCs * omega_x))   # m  (~1.68 µm)
t_unit_s  = 1.0 / omega_x                    # s per dimensionless time unit
t_unit_ms = t_unit_s * 1e3                   # ms per dimensionless time unit

# Dimensionless trap frequency ratios (omega_x = 1 in code)
wx = 1.0
wy = omega_y / omega_x
wz = omega_z / omega_x

# ── Interaction parameters (at microwave compensation point) ──────────────────
N_mol = 2000             # molecule number (BEC onset, Bigagli et al. 2024)
a_s   = 1500.0 * a0     # s-wave scattering length (m) — tunable via field-linked resonances
a_dd  = 1300.0 * a0     # dipolar length (m) — add = M*C_dd/(12*pi*hbar^2)
                          # set a_dd = 0 to recover pure contact GPE

eps_dd = a_dd / a_s      # relative dipolar strength (0.87 for NaCs BEC paper)

g_contact = 4 * np.pi * (a_s / a_ho)      # dimensionless contact coupling
g_dipolar = 4 * np.pi * (a_dd / a_ho)     # dimensionless dipolar coupling

print(f"NaCs BEC parameters:")
print(f"  a_ho = {a_ho*1e6:.2f} µm")
print(f"  t_unit = {t_unit_ms:.2f} ms")
print(f"  Trap: ({freq_x_Hz}, {freq_y_Hz}, {freq_z_Hz}) Hz  "
      f"→ ratios (1, {wy:.2f}, {wz:.2f})")
print(f"  a_s = {a_s/a0:.0f} a0,  a_dd = {a_dd/a0:.0f} a0,  eps_dd = {eps_dd:.2f}")
print(f"  g_contact = {g_contact:.4f},  g_dipolar = {g_dipolar:.4f}")

# ── Grid ──────────────────────────────────────────────────────────────────────
# z is the tight axis; needs finer resolution to resolve initial condensate
Nx, Ny, Nz = 64, 64, 128
Lx, Ly, Lz = 12.0, 12.0, 10.0   # box half-widths in a_ho

x = np.linspace(-Lx, Lx, Nx, endpoint=False)
y = np.linspace(-Ly, Ly, Ny, endpoint=False)
z = np.linspace(-Lz, Lz, Nz, endpoint=False)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
dV = dx * dy * dz

print(f"\nGrid: {Nx}×{Ny}×{Nz},  dz = {dz:.3f} a_ho = {dz*a_ho*1e6:.2f} µm")

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Momentum-space grids
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# ── Dipolar convolution kernel in k-space ────────────────────────────────────
# V_dd_hat(k) = (C_dd/3)(3*k_z^2/k^2 - 1)  [standard result for z-polarised dipoles]
# In dimensionless units the prefactor is absorbed into g_dipolar.
# k=0 is set to 0 (net dipolar energy vanishes for an isotropic system).
with np.errstate(invalid='ignore', divide='ignore'):
    dipolar_kernel = np.where(K2 > 0, (3 * KZ**2 / K2 - 1) / 3.0, 0.0)

def dipolar_potential(psi):
    """Return dimensionless dipolar mean-field potential Phi_dd(r)."""
    n = np.abs(psi)**2
    n_k = np.fft.fftn(n)
    Phi_k = N_mol * g_dipolar * dipolar_kernel * n_k
    return np.fft.ifftn(Phi_k).real

# ── Trap potential ────────────────────────────────────────────────────────────
V_trap = 0.5 * (wx**2 * X**2 + wy**2 * Y**2 + wz**2 * Z**2)

# ── Split-step propagators ────────────────────────────────────────────────────
def kinetic_prop(dt):
    return np.exp(-0.5j * 0.5 * K2 * dt)   # hbar=M=1

def potential_step(psi, dt, V_ext):
    n = np.abs(psi)**2
    Phi_dd = dipolar_potential(psi) if g_dipolar != 0 else 0.0
    phase = (V_ext + N_mol * g_contact * n + Phi_dd) * dt
    return psi * np.exp(-1j * phase)

def normalize(psi):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dV)

def ssfm_step(psi, dt, V_ext, kin_prop_arr):
    psi_k = np.fft.fftn(psi)
    psi_k *= kin_prop_arr
    psi   = np.fft.ifftn(psi_k)
    psi   = potential_step(psi, dt, V_ext)
    psi_k = np.fft.fftn(psi)
    psi_k *= kin_prop_arr
    psi   = np.fft.ifftn(psi_k)
    return psi

# ── 1. Imaginary-time propagation → ground state ──────────────────────────────
print("\nFinding ground state via imaginary-time propagation...")

dt_imag   = -1j * 0.01
kin_imag  = kinetic_prop(dt_imag)

# Initial guess: Gaussian matching expected TF widths
psi = np.exp(-(X**2/(2*1.5**2) + Y**2/(2*0.8**2) + Z**2/(2*0.6**2))).astype(complex)
psi = normalize(psi)

n_imag = 3000
for step in range(n_imag):
    psi = ssfm_step(psi, dt_imag, V_trap, kin_imag)
    psi = normalize(psi)
    if step % 750 == 0:
        kin  = np.fft.ifftn(0.5 * K2 * np.fft.fftn(psi))
        Phi_dd = dipolar_potential(psi) if g_dipolar != 0 else 0.0
        n = np.abs(psi)**2
        mu = np.real(np.sum(np.conj(psi) * (
             kin + (V_trap + N_mol * g_contact * n + Phi_dd) * psi
        )) * dV)
        print(f"  step {step:4d}  μ = {mu:.4f} ℏω_x = {mu * hbar * omega_x / (hbar * 2*np.pi * 1e3):.1f} kHz")

print("Ground state found.\n")
psi_gs = psi.copy()

# ── 2. Real-time TOF expansion ─────────────────────────────────────────────────
print("Running time-of-flight expansion...")

dt_real   = 0.005
kin_real  = kinetic_prop(dt_real)

save_times_ms = [0, 5, 10, 15, 17]                         # 17 ms matches Bigagli 2024
save_times    = [t_ms / t_unit_ms for t_ms in save_times_ms]
save_steps    = {int(round(ts / abs(dt_real))): t_ms
                 for ts, t_ms in zip(save_times, save_times_ms)}
n_tof_steps   = max(save_steps.keys())

snapshots     = {}
waist_times_ms, waist_x, waist_y, waist_z = [], [], [], []

def rms_waist(n3d, coord):
    norm = n3d.sum() * dV
    mean = (n3d * coord).sum() * dV / norm
    return np.sqrt((n3d * (coord - mean)**2).sum() * dV / norm) * a_ho * 1e6   # µm

psi   = psi_gs.copy()
V_off = np.zeros_like(V_trap)

for step in range(n_tof_steps + 1):
    n3d = np.abs(psi)**2
    if step in save_steps:
        t_ms = save_steps[step]
        snapshots[t_ms] = n3d
        print(f"  saved snapshot at t = {t_ms} ms")
    waist_times_ms.append(step * abs(dt_real) * t_unit_ms)
    waist_x.append(rms_waist(n3d, X))
    waist_y.append(rms_waist(n3d, Y))
    waist_z.append(rms_waist(n3d, Z))
    if step < n_tof_steps:
        psi = ssfm_step(psi, dt_real, V_off, kin_real)

# ── 3. Density snapshot plots ─────────────────────────────────────────────────
print("\nPlotting density snapshots...")

fig = plt.figure(figsize=(16, 8))
fig.suptitle(
    f"NaCs BEC TOF — eGPE (contact + dipolar),  "
    f"N={N_mol},  trap ({freq_x_Hz},{freq_y_Hz},{freq_z_Hz}) Hz,  "
    f"ε_dd={eps_dd:.2f}",
    fontsize=12
)

gs = GridSpec(2, len(save_times_ms), figure=fig, hspace=0.4, wspace=0.3)

vmax_xy = max(snapshots[t].sum(axis=2).max() * dz for t in save_times_ms)
vmax_xz = max(snapshots[t].sum(axis=1).max() * dy for t in save_times_ms)

# Spatial axes in µm
x_um = x * a_ho * 1e6
y_um = y * a_ho * 1e6
z_um = z * a_ho * 1e6

for col, t_snap in enumerate(save_times_ms):
    n3d = snapshots[t_snap]

    # XY projection (integrate over z) — camera view along z
    n_xy = n3d.sum(axis=2) * dz * a_ho   # µm^-2 (×a_ho to convert dz from a_ho to m)
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(n_xy.T, origin='lower', aspect='equal',
              extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
              cmap='Blues', vmin=0, vmax=vmax_xy)
    ax.set_title(f"t = {t_snap} ms", fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    if col == 0:
        ax.set_ylabel("y (µm)", fontsize=8)

    # XZ projection (integrate over y) — side view
    n_xz = n3d.sum(axis=1) * dy * a_ho
    ax2 = fig.add_subplot(gs[1, col])
    ax2.imshow(n_xz.T, origin='lower', aspect='equal',
               extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
               cmap='Blues', vmin=0, vmax=vmax_xz)
    ax2.set_xlabel("x (µm)", fontsize=8)
    if col == 0:
        ax2.set_ylabel("z (µm)", fontsize=8)

fig.text(0.01, 0.73, "XY\nview", va='center', ha='left', fontsize=9)
fig.text(0.01, 0.27, "XZ\nview", va='center', ha='left', fontsize=9)

plt.savefig("tof_expansion.png", dpi=150, bbox_inches='tight')
print("Saved: tof_expansion.png")

# ── 4. Waist evolution plot ───────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(waist_times_ms, waist_x, label=f"x  ({freq_x_Hz} Hz)", color='steelblue')
ax.plot(waist_times_ms, waist_y, label=f"y  ({freq_y_Hz} Hz)", color='cornflowerblue', ls='--')
ax.plot(waist_times_ms, waist_z, label=f"z  ({freq_z_Hz} Hz)", color='tomato')
ax.set_xlabel("Time of flight (ms)", fontsize=12)
ax.set_ylabel("RMS waist (µm)", fontsize=12)
ax.set_title(f"NaCs BEC waist evolution  (ε_dd = {eps_dd:.2f},  N = {N_mol})", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig("tof_waists.png", dpi=150, bbox_inches='tight')
print("Saved: tof_waists.png")
plt.show()
