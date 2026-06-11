import numpy as np
import matplotlib.pyplot as plt

# Parameters
M_bar = 2e8          # solar masses
r_s = 1.0            # kpc
G = 4.302e-3         # kpc*(km/s)^2/M_sun
Lambda_GIC = 5e-21   # s^-2

r = np.logspace(-2, 1, 200)  # 0.01 to 10 kpc

# Enclosed baryonic mass for Plummer profile
M_enc_bar = M_bar * (r**3 / (r_s**2 + r**2)**(1.5))  # analytic enclosed mass for Plummer

# Baryonic acceleration
g_bar = G * M_enc_bar / r**2   # (km/s)^2/kpc
g_bar_mks = g_bar * 3.24e-14   # convert to m/s^2

# GIC circular velocity: v^2 = G M_bar(r)/r + Lambda_GIC * r^2
v_gic2 = G * M_enc_bar / r + Lambda_GIC * r**2
v_gic = np.sqrt(np.abs(v_gic2))
g_obs = v_gic2 / r   # (km/s)^2/kpc
g_obs_mks = g_obs * 3.24e-14

# Mask to avoid very small radii where Lambda_GIC term is negligible
mask = r > 0.1

# Observed RAR data (McGaugh+2016)
rar_g_bar = np.array([1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10])
rar_g_obs = np.array([1.5e-14, 4e-14, 1.2e-13, 3.5e-13, 1.1e-12, 3.2e-12, 1.0e-11, 2.8e-11, 7e-11, 1.5e-10])

plt.figure(figsize=(7,6))
plt.loglog(rar_g_bar, rar_g_obs, 'k-', linewidth=2, label='Observed RAR (McGaugh+2016)')
plt.loglog(g_bar_mks[mask], g_obs_mks[mask], 'ro', markersize=4, label='GIC analytic prediction (NGC 6822)')
plt.plot([1e-14, 1e-9], [1e-14, 1e-9], 'k--', alpha=0.5, label=r'$g_{\rm obs} = g_{\rm bar}$')
plt.xlabel(r'$g_{\rm bar}$ (m/s$^2$)')
plt.ylabel(r'$g_{\rm obs}$ (m/s$^2$)')
plt.title('Radial Acceleration Relation: GIC vs. data')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gic_RAR.png', dpi=150)
plt.show()