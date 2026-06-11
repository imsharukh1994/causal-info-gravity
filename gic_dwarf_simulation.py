import numpy as np
import matplotlib.pyplot as plt

# Parameters
M_bar = 2e8          # M_sun
r_s = 1.0            # kpc
a0 = 1.24e-23        # kpc/s^2
G = 4.302e-3         # kpc*(km/s)^2/M_sun
Lambda_GIC = 5e-21   # s^-2
r_min, r_max, N = 0.01, 20.0, 500
max_iter, tol, omega = 20000, 1e-8, 1.2

def plummer_density(r, M, rs):
    return (3 * M) / (4 * np.pi * rs**3) * (1 + (r/rs)**2)**(-5/2)

def compute_source(r, phi, rho_bar, a0, Lambda_GIC, dr):
    dphi_dr = np.gradient(phi, dr)
    acc = np.abs(dphi_dr)
    k = 20.0
    x = k * (acc / a0 - 1.0)
    # Stable logistic: clip exponent to avoid overflow
    alpha = np.where(x > 100, 0.0, 1.0 / (1.0 + np.exp(x)))
    source = 4.0 * np.pi * G * rho_bar + alpha * Lambda_GIC
    return source

def solve_poisson(r, rho_bar, a0, Lambda_GIC, phi_guess=None):
    dr = r[1] - r[0]
    N = len(r)
    phi = np.zeros(N) if phi_guess is None else phi_guess.copy()
    for it in range(max_iter):
        phi_old = phi.copy()
        source = compute_source(r, phi, rho_bar, a0, Lambda_GIC, dr)
        for i in range(1, N-1):
            A = 1.0 / dr**2
            B = 1.0 / (r[i] * dr)
            num = A * (phi[i+1] + phi[i-1]) + B * (phi[i+1] - phi[i-1]) - source[i]
            denom = 2.0 * A
            phi_new = num / denom
            phi[i] = (1.0 - omega) * phi[i] + omega * phi_new
        phi[0] = phi[1]
        M_total = np.trapezoid(4 * np.pi * r**2 * rho_bar, r)
        phi[-1] = -G * M_total / r[-1]
        diff = np.max(np.abs(phi - phi_old))
        if diff < tol:
            print(f"Converged after {it+1} iterations (diff={diff:.2e})")
            break
    else:
        print(f"Warning: did not converge after {max_iter} iterations (diff={diff:.2e})")
    return phi

def rotation_curve(r, phi):
    dr = r[1] - r[0]
    dphi_dr = np.gradient(phi, dr)
    return np.sqrt(np.abs(r * dphi_dr))

# Run
r = np.linspace(r_min, r_max, N)
rho_bar = plummer_density(r, M_bar, r_s)
phi_newton = solve_poisson(r, rho_bar, a0, 0.0)
phi_gic = solve_poisson(r, rho_bar, a0, Lambda_GIC, phi_guess=phi_newton)
v_newton = rotation_curve(r, phi_newton)
v_gic = rotation_curve(r, phi_gic)

# Plot (same as before)
obs_r = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
obs_v = np.array([20, 28, 38, 44, 50, 52, 53, 53, 52, 50])
obs_err = np.array([5, 5, 5, 4, 4, 3, 3, 3, 4, 5])

plt.figure(figsize=(8,5))
plt.plot(r, v_newton, 'b--', label='Baryons only (Newton)')
plt.plot(r, v_gic, 'r-', label='GIC (Coherent vacuum)')
plt.errorbar(obs_r, obs_v, yerr=obs_err, fmt='ko', capsize=3, label='NGC 6822 data')
plt.xlabel('Radius (kpc)'); plt.ylabel('Circular velocity (km/s)')
plt.title('Rotation curve of dwarf galaxy NGC 6822')
plt.xlim(0,12); plt.ylim(0,70); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('gic_rotation_curve.png'); plt.show()

plt.figure(figsize=(8,5))
plt.loglog(r, rho_bar, 'b-', label='Baryonic density (Plummer core)')
rho_nfw = 1e6 * r**(-1)
plt.loglog(r, rho_nfw, 'k:', label='NFW cusp (r⁻¹)')
plt.xlabel('Radius (kpc)'); plt.ylabel('Density (M_sun / kpc³)')
plt.title('Density profile: core vs cusp')
plt.legend(); plt.grid(alpha=0.3); plt.savefig('gic_density_profile.png'); plt.show()

print("Done. Plots saved.")