"""
3D Gross-Pitaevskii Equation (GPE) simulation of BEC time-of-flight (TOF) expansion.
Simple contact-interaction only (no dipolar term).

For the full NaCs eGPE with dipolar interactions see egpe_tof.py.

Uses the split-step Fourier method (SSFM) to propagate:
    i*hbar * d/dt psi = [ -hbar^2/(2m) nabla^2 + V(r) + g|psi|^2 ] psi

Units: harmonic oscillator units of the x-axis trap
  length : l0 = sqrt(hbar / m / omega_x)
  energy : hbar * omega_x
  time   : 1 / omega_x
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Physical parameters ──────────────────────────────────────────────────────
# Trap frequencies in Hz
freq_x_Hz = 20.0
freq_y_Hz = 20.0
freq_z_Hz = 160.0

hbar = 1.0546e-34
m    = 1.443e-25    # kg (87Rb)

omega_x_SI = 2 * np.pi * freq_x_Hz
omega_y_SI = 2 * np.pi * freq_y_Hz
omega_z_SI = 2 * np.pi * freq_z_Hz

a_ho = np.sqrt(hbar / (m * omega_x_SI))
t_unit_ms = 1e3 / omega_x_SI

omega_x = 1.0
omega_y = omega_y_SI / omega_x_SI
omega_z = omega_z_SI / omega_x_SI

N_atoms  = 5_000
a_s      = 5.2e-9
g3D      = 4 * np.pi * (a_s / a_ho)

# ── Grid ─────────────────────────────────────────────────────────────────────
Nx, Ny, Nz = 64, 64, 256
Lx, Ly, Lz = 20.0, 20.0, 20.0

x = np.linspace(-Lx, Lx, Nx, endpoint=False)
y = np.linspace(-Ly, Ly, Ny, endpoint=False)
z = np.linspace(-Lz, Lz, Nz, endpoint=False)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
dV = dx * dy * dz

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

V_trap = 0.5 * (omega_x**2 * X**2 + omega_y**2 * Y**2 + omega_z**2 * Z**2)

# ── Split-step propagators ────────────────────────────────────────────────────
def kinetic_propagator(dt):
    return np.exp(-0.5j * 0.5 * K2 * dt)

def potential_propagator(psi, dt, V, g):
    return psi * np.exp(-1j * (V + g * N_atoms * np.abs(psi)**2) * dt)

def normalize(psi):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dV)

def ssfm_step(psi, dt, V, g, kin_prop):
    psi_k = np.fft.fftn(psi)
    psi_k *= kin_prop
    psi   = np.fft.ifftn(psi_k)
    psi   = potential_propagator(psi, dt, V, g)
    psi_k = np.fft.fftn(psi)
    psi_k *= kin_prop
    psi   = np.fft.ifftn(psi_k)
    return psi

# ── 1. Imaginary-time propagation → ground state ─────────────────────────────
print("Finding ground state via imaginary-time propagation...")

dt_imag  = -1j * 0.01
kin_imag = kinetic_propagator(dt_imag)

psi = np.exp(-(X**2 + Y**2 + Z**2) / (2 * 1.5**2)).astype(complex)
psi = normalize(psi)

for step in range(2000):
    psi = ssfm_step(psi, dt_imag, V_trap, g3D, kin_imag)
    psi = normalize(psi)
    if step % 500 == 0:
        mu = (np.sum(np.conj(psi) * (
              np.fft.ifftn(0.5 * K2 * np.fft.fftn(psi)) +
              (V_trap + g3D * N_atoms * np.abs(psi)**2) * psi
        )) * dV).real
        print(f"  step {step:4d}  μ = {mu:.4f} ℏω_x")

print("Ground state found.\n")
psi_gs = psi.copy()

# ── 2. Real-time TOF expansion ────────────────────────────────────────────────
print("Running time-of-flight expansion...")

dt_real = 0.005
kin_real = kinetic_propagator(dt_real)

save_times_ms = [0, 5, 10, 15, 20]
save_times = [t_ms / t_unit_ms for t_ms in save_times_ms]
save_steps = {int(round(ts / abs(dt_real))): t_ms
              for ts, t_ms in zip(save_times, save_times_ms)}
n_tof_steps = max(save_steps.keys())
snapshots   = {}

psi   = psi_gs.copy()
V_off = np.zeros_like(V_trap)

for step in range(n_tof_steps + 1):
    if step in save_steps:
        t_ms = save_steps[step]
        snapshots[t_ms] = np.abs(psi)**2
        print(f"  saved snapshot at t = {t_ms} ms")
    if step < n_tof_steps:
        psi = ssfm_step(psi, dt_real, V_off, g3D, kin_real)

# ── 3. Plot ───────────────────────────────────────────────────────────────────
print("\nPlotting...")

fig = plt.figure(figsize=(16, 8))
fig.suptitle("BEC Time-of-Flight Expansion — simple GPE (87Rb, contact only)", fontsize=13)
gs = GridSpec(2, len(save_times_ms), figure=fig, hspace=0.4, wspace=0.3)

vmax_xy = max(snapshots[t].sum(axis=2).max() * dz for t in save_times_ms)
vmax_xz = max(snapshots[t].sum(axis=1).max() * dy for t in save_times_ms)

for col, t_snap in enumerate(save_times_ms):
    n3d = snapshots[t_snap]

    ax = fig.add_subplot(gs[0, col])
    ax.imshow((n3d.sum(axis=2) * dz).T, origin='lower', aspect='equal',
              extent=[-Lx, Lx, -Ly, Ly], cmap='Blues', vmin=0, vmax=vmax_xy)
    ax.set_title(f"t = {t_snap} ms", fontsize=9)
    ax.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax.set_ylabel("y / l₀", fontsize=8)

    ax2 = fig.add_subplot(gs[1, col])
    ax2.imshow((n3d.sum(axis=1) * dy).T, origin='lower', aspect='equal',
               extent=[-Lx, Lx, -Lz, Lz], cmap='Blues', vmin=0, vmax=vmax_xz)
    ax2.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax2.set_ylabel("z / l₀", fontsize=8)

fig.text(0.01, 0.73, "XY\nview", va='center', ha='left', fontsize=9)
fig.text(0.01, 0.27, "XZ\nview", va='center', ha='left', fontsize=9)

plt.savefig("tof_expansion_gpe.png", dpi=150, bbox_inches='tight')
print("Saved: tof_expansion_gpe.png")
plt.show()
