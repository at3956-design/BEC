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
# Trap frequencies in Hz (angular: omega = 2*pi*f)
freq_x_Hz = 20.0    # Hz
freq_y_Hz = 20.0    # Hz
freq_z_Hz = 160.0   # Hz  (tight axis → pancake)

hbar = 1.0546e-34   # J·s
m    = 1.443e-25    # kg (87Rb)

omega_x_SI = 2 * np.pi * freq_x_Hz   # rad/s
omega_y_SI = 2 * np.pi * freq_y_Hz
omega_z_SI = 2 * np.pi * freq_z_Hz

# Harmonic oscillator length along x (our length unit)
a_ho = np.sqrt(hbar / (m * omega_x_SI))   # meters
# Time unit: 1/omega_x in seconds
t_unit_s = 1.0 / omega_x_SI               # seconds per dimensionless time unit
t_unit_ms = t_unit_s * 1e3                # milliseconds per dimensionless time unit

# Dimensionless trap frequency ratios (omega_x = 1 in code)
omega_x = 1.0
omega_y = omega_y_SI / omega_x_SI
omega_z = omega_z_SI / omega_x_SI

N_atoms  = 5_000
a_s      = 5.2e-9                          # s-wave scattering length (m), 87Rb
g3D      = 4 * np.pi * (a_s / a_ho)       # dimensionless coupling

# ── Grid ─────────────────────────────────────────────────────────────────────
Nx, Ny, Nz = 64, 64, 256          # high Nz needed: initial R_z ~ 0.5 l0 requires dz << 0.5
Lx, Ly, Lz = 20.0, 20.0, 20.0   # dz = 40/256 ~ 0.16 l0, resolves tight z-axis cleanly

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

save_times_ms = [0, 5, 10, 15, 20]                         # snapshots in ms
save_times = [t_ms / t_unit_ms for t_ms in save_times_ms]  # convert to dimensionless
save_steps = {int(round(ts / abs(dt_real))): t_ms
              for ts, t_ms in zip(save_times, save_times_ms)}
n_tof_steps = max(save_steps.keys())   # run exactly to the last snapshot step
snapshots   = {}

psi = psi_gs.copy()
V_off = np.zeros_like(V_trap)           # trap is OFF during TOF

def rms_waist(n3d, coord, dV):
    """RMS width along one axis given 3D density and coordinate array."""
    norm = n3d.sum() * dV
    mean = (n3d * coord).sum() * dV / norm
    return np.sqrt((n3d * (coord - mean)**2).sum() * dV / norm)

waist_times_ms = []
waist_x, waist_y, waist_z = [], [], []

for step in range(n_tof_steps + 1):
    n3d = np.abs(psi)**2
    if step in save_steps:
        t_ms = save_steps[step]
        snapshots[t_ms] = n3d
        print(f"  saved snapshot at t = {t_ms} ms")
    waist_times_ms.append(step * abs(dt_real) * t_unit_ms)
    waist_x.append(rms_waist(n3d, X, dV) * a_ho * 1e6)   # convert to µm
    waist_y.append(rms_waist(n3d, Y, dV) * a_ho * 1e6)
    waist_z.append(rms_waist(n3d, Z, dV) * a_ho * 1e6)
    if step < n_tof_steps:
        psi = ssfm_step(psi, dt_real, V_off, g3D, kin_real)

# ── 3. Plot column-integrated density (integrate along z) ────────────────────
print("\nPlotting...")

fig = plt.figure(figsize=(16, 8))
fig.suptitle("BEC Time-of-Flight Expansion (3D GPE, split-step Fourier)", fontsize=14)

gs = GridSpec(2, len(save_times_ms), figure=fig, hspace=0.4, wspace=0.3)

vmax_xy = max(snapshots[t].sum(axis=2).max() * dz for t in save_times_ms)
vmax_xz = max(snapshots[t].sum(axis=1).max() * dy for t in save_times_ms)

for col, t_snap in enumerate(save_times_ms):
    n3d = snapshots[t_snap]

    # XY projection (integrate over z)
    n_xy = n3d.sum(axis=2) * dz
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(n_xy.T, origin='lower', aspect='equal',
              extent=[-Lx, Lx, -Ly, Ly], cmap='Blues', vmin=0, vmax=vmax_xy)
    ax.set_title(f"t = {save_times_ms[col]} ms", fontsize=9)
    ax.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax.set_ylabel("y / l₀", fontsize=8)

    # XZ projection (integrate over y)
    n_xz = n3d.sum(axis=1) * dy
    ax2 = fig.add_subplot(gs[1, col])
    ax2.imshow(n_xz.T, origin='lower', aspect='equal',
               extent=[-Lx, Lx, -Lz, Lz], cmap='Blues', vmin=0, vmax=vmax_xz)
    ax2.set_xlabel("x / l₀", fontsize=8)
    if col == 0:
        ax2.set_ylabel("z / l₀", fontsize=8)

# Row labels
fig.text(0.01, 0.73, "XY\nprojection", va='center', ha='left', fontsize=9, rotation=0)
fig.text(0.01, 0.27, "XZ\nprojection", va='center', ha='left', fontsize=9, rotation=0)

plt.savefig("tof_expansion.png", dpi=150, bbox_inches='tight')
print("Saved: tof_expansion.png")

# ── 4. Waist evolution plot ───────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(waist_times_ms, waist_x, label=f"x  (ω_x = {freq_x_Hz} Hz)", color='steelblue')
ax.plot(waist_times_ms, waist_y, label=f"y  (ω_y = {freq_y_Hz} Hz)", color='cornflowerblue', linestyle='--')
ax.plot(waist_times_ms, waist_z, label=f"z  (ω_z = {freq_z_Hz} Hz)", color='tomato')
ax.set_xlabel("Time of flight (ms)", fontsize=12)
ax.set_ylabel("RMS waist (µm)", fontsize=12)
ax.set_title("BEC cloud waist during TOF", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig("tof_waists.png", dpi=150, bbox_inches='tight')
print("Saved: tof_waists.png")
plt.show()
