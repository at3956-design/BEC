"""
3D Gross-Pitaevskii Equation (GPE) simulation of BEC time-of-flight (TOF) expansion.

Uses the split-step Fourier method (SSFM) to propagate:
    i*hbar * d/dt psi = [ -hbar^2/(2m) nabla^2 + V(r) + g|psi|^2 ] psi

Workflow:
  1. Imaginary-time propagation to find ground state in harmonic trap.
  2. Real-time propagation with trap off (TOF).
  3. Plot column-integrated density snapshots.

Units: harmonic oscillator units of the x-axis trap
  length : l0 = sqrt(hbar / m / omega_x)
  energy : hbar * omega_x
  time   : 1 / omega_x
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Physical parameters ──────────────────────────────────────────────────────
# Trap frequencies (in units of omega_x, so omega_x = 1)
omega_x = 1.0
omega_y = 1.0   # change to make cigar/pancake shape
omega_z = 8.0   # tight z-axis → pancake-like (common in labs)

N_atoms  = 5_000          # number of atoms
a_s      = 5.2e-9         # s-wave scattering length (m), e.g. 87Rb
a_ho     = 1e-6           # harmonic oscillator length l0 (m)
g3D      = 4 * np.pi * (a_s / a_ho)  # dimensionless coupling

# ── Grid ─────────────────────────────────────────────────────────────────────
Nx, Ny, Nz = 128, 128, 64         # grid points (keep powers of 2 for FFT)
Lx, Ly, Lz = 40.0, 40.0, 20.0   # box half-widths in l0 — large enough for TOF=20

x = np.linspace(-Lx, Lx, Nx, endpoint=False)
y = np.linspace(-Ly, Ly, Ny, endpoint=False)
z = np.linspace(-Lz, Lz, Nz, endpoint=False)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
dV = dx * dy * dz

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Momentum-space grids
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# ── Potentials ────────────────────────────────────────────────────────────────
V_trap = 0.5 * (omega_x**2 * X**2 + omega_y**2 * Y**2 + omega_z**2 * Z**2)

# ── Split-step propagators ────────────────────────────────────────────────────
def kinetic_propagator(dt):
    """Half-step kinetic phase in k-space."""
    return np.exp(-0.5j * 0.5 * K2 * dt)   # hbar=m=1

def potential_propagator(psi, dt, V, g):
    """Full-step potential + interaction phase in real-space."""
    n = np.abs(psi)**2
    phase = (V + g * N_atoms * n) * dt
    return psi * np.exp(-1j * phase)

def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dV)
    return psi / norm

def ssfm_step(psi, dt, V, g, kin_prop):
    """One split-step Fourier step."""
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

dt_imag  = -1j * 0.01          # imaginary time step
kin_imag = kinetic_propagator(dt_imag)

# Initial guess: Gaussian
sigma = 1.5
psi = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)).astype(complex)
psi = normalize(psi)

n_imag_steps = 2000
for step in range(n_imag_steps):
    psi = ssfm_step(psi, dt_imag, V_trap, g3D, kin_imag)
    psi = normalize(psi)   # restore norm after imaginary-time decay
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

dt_real = 0.005                         # real-time step (1/omega_x)
kin_real = kinetic_propagator(dt_real)

t_tof_max  = 20.0                       # total TOF duration
n_tof_steps = int(t_tof_max / abs(dt_real))
save_times  = [0, 2, 5, 10, 15, 20]    # snapshots in units of 1/omega_x
snapshots   = {}

psi = psi_gs.copy()
V_off = np.zeros_like(V_trap)           # trap is OFF during TOF

t = 0.0
for step in range(n_tof_steps + 1):
    t_round = round(t, 6)
    if any(abs(t_round - ts) < abs(dt_real) / 2 for ts in save_times):
        key = min(save_times, key=lambda ts: abs(t_round - ts))
        if key not in snapshots:
            snapshots[key] = np.abs(psi)**2
            print(f"  saved snapshot at t = {t:.2f} / ω_x")
    if step < n_tof_steps:
        psi = ssfm_step(psi, dt_real, V_off, g3D, kin_real)
    t += abs(dt_real)

# ── 3. Plot column-integrated density (integrate along z) ────────────────────
print("\nPlotting...")

fig = plt.figure(figsize=(16, 8))
fig.suptitle("BEC Time-of-Flight Expansion (3D GPE, split-step Fourier)", fontsize=14)

gs = GridSpec(2, len(save_times), figure=fig, hspace=0.4, wspace=0.3)

vmax_xy = max(snapshots[t].sum(axis=2).max() * dz for t in save_times)
vmax_xz = max(snapshots[t].sum(axis=1).max() * dy for t in save_times)

for col, t_snap in enumerate(save_times):
    n3d = snapshots[t_snap]

    # XY projection (integrate over z)
    n_xy = n3d.sum(axis=2) * dz
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(n_xy.T, origin='lower', aspect='equal',
              extent=[-Lx, Lx, -Ly, Ly], cmap='inferno', vmin=0, vmax=vmax_xy)
    ax.set_title(f"t = {t_snap} / ω_x", fontsize=9)
    ax.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax.set_ylabel("y / l₀", fontsize=8)

    # XZ projection (integrate over y)
    n_xz = n3d.sum(axis=1) * dy
    ax2 = fig.add_subplot(gs[1, col])
    ax2.imshow(n_xz.T, origin='lower', aspect='equal',
               extent=[-Lx, Lx, -Lz, Lz], cmap='inferno', vmin=0, vmax=vmax_xz)
    ax2.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax2.set_ylabel("z / l₀", fontsize=8)

# Row labels
fig.text(0.01, 0.73, "XY\nprojection", va='center', ha='left', fontsize=9, rotation=0)
fig.text(0.01, 0.27, "XZ\nprojection", va='center', ha='left', fontsize=9, rotation=0)

plt.savefig("tof_expansion.png", dpi=150, bbox_inches='tight')
print("Saved: tof_expansion.png")
plt.show()
