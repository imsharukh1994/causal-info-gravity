import numpy as np
import matplotlib.pyplot as plt

G = 4.302e-3          # kpc (km/s)^2 / M_sun
Lambda_GIC = 5e-21    # s^-2
a0 = 1.24e-23         # kpc/s^2 (not used directly)

# Coma cluster: baryonic mass model (stars + hot gas)
M_bar_total = 1e14     # M_sun
r_s = 300.0            # kpc
r = np.logspace(np.log10(10), np.log10(3000), 500)

def nfw_enc(r, M_vir, r_s):
    x = r / r_s
    return M_vir * (np.log(1+x) - x/(1+x)) / (np.log(1+200) - 200/(1+200))

M_enc_bar = nfw_enc(r, M_bar_total, r_s)
g_bar = G * M_enc_bar / r**2                 # (km/s)^2/kpc
g_bar_mks = g_bar * 3.24e-14                 # m/s^2

v_gic = np.sqrt(G * M_enc_bar / r + Lambda_GIC * r**2)
g_obs_gic = v_gic**2 / r
g_obs_gic_mks = g_obs_gic * 3.24e-14

# Observed acceleration from lensing (Okabe+10, approximate)
obs_r = np.array([100, 200, 300, 500, 700, 1000, 1500, 2000, 2500])
obs_g = np.array([5e-15, 2e-15, 1e-15, 5e-16, 3e-16, 1.5e-16, 8e-17, 5e-17, 3e-17])

plt.figure(figsize=(7,5))
plt.loglog(r, g_obs_gic_mks, 'r-', label=r'GIC prediction (baryons + $\Lambda r^2$)')
plt.loglog(obs_r, obs_g, 'ko', label='Observed total acceleration (Okabe+10)')
plt.loglog(r, g_bar_mks, 'b--', label='Baryonic acceleration only')
plt.xlabel(r'$r$ (kpc)')
plt.ylabel(r'$g_{\rm obs}$ (m/s$^2$)')
plt.title(r'Coma cluster: GIC fails (needs larger $\Lambda$)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('coma_fails.png', dpi=150)
plt.show()

ratio = np.interp(obs_r, r, g_obs_gic_mks) / obs_g
print(f'GIC under-predicts observed acceleration by factor ~ {np.mean(ratio):.1f}')