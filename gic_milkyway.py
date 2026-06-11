import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 4.302e-3          # kpc (km/s)^2 / M_sun
Lambda_GIC = 5e-21    # s^-2 (same as for dwarfs)

# Milky Way baryonic mass model (simplified)
M_bulge = 1.5e10      # M_sun
r_bulge = 0.5         # kpc
M_disk = 5e10         # M_sun
r_disk = 3.5          # kpc
M_gas = 1e10          # M_sun
r_gas = 7.0           # kpc

r = np.linspace(0.1, 30, 500)

def plummer_enc(r, M, rs):
    return M * r**3 / (rs**2 + r**2)**1.5

def expdisk_enc(r, M, rd):
    return M * (1 - (1 + r/rd) * np.exp(-r/rd))

v_bulge = np.sqrt(G * plummer_enc(r, M_bulge, r_bulge) / r)
v_disk  = np.sqrt(G * expdisk_enc(r, M_disk, r_disk) / r)
v_gas   = np.sqrt(G * expdisk_enc(r, M_gas, r_gas) / r)

v_bar = np.sqrt(v_bulge**2 + v_disk**2 + v_gas**2)
v_gic = np.sqrt(v_bar**2 + Lambda_GIC * r**2)

# Observed Milky Way rotation curve (Sofue 2016, approximate)
obs_r = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0])
obs_v = np.array([150, 180, 200, 210, 215, 220, 222, 225, 225, 224, 223,
                  222, 220, 218, 215, 212, 205, 198])
obs_err = np.array([15, 10, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 10])

plt.figure(figsize=(8,6))
plt.plot(r, v_bar, 'b--', label='Baryons only (bulge+disk+gas)')
plt.plot(r, v_gic, 'r-', lw=2, label=r'GIC (baryons + $\Lambda_{\rm GIC} r^2$)')
plt.errorbar(obs_r, obs_v, yerr=obs_err, fmt='ko', capsize=3,
             label='Observed total (Sofue 2016)')
plt.xlabel('Radius (kpc)')
plt.ylabel('Circular velocity (km/s)')
plt.title('Milky Way rotation curve: GIC prediction vs. data')
plt.xlim(0, 30)
plt.ylim(0, 250)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gic_milkyway_rotation.png', dpi=150)
plt.show()

print("Milky Way GIC plot saved as gic_milkyway_rotation.png")