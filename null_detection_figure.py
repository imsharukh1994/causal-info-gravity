import numpy as np
import matplotlib.pyplot as plt

# XENONnT 2024 upper limits (approx, from arXiv:2405.01642)
wimp_mass = np.array([6, 10, 20, 30, 50, 70, 100, 200, 500, 1000])
upper_limit = np.array([3e-46, 5e-47, 1.5e-47, 8e-48, 4e-48, 3e-48, 2.5e-48, 2e-48, 1.8e-48, 1.7e-48])

plt.figure(figsize=(8,6))
plt.loglog(wimp_mass, upper_limit, 'k-', linewidth=2, label='XENONnT 2024 upper limit (90% CL)')
plt.fill_between(wimp_mass, upper_limit, 1e-40, color='gray', alpha=0.3, label='Excluded by XENONnT')

# GIC prediction: horizontal line at a very low cross-section (effectively zero)
plt.axhline(y=1e-55, color='r', linestyle='--', linewidth=2)

# Add an explanatory text box (not a data column)
plt.text(200, 3e-54, 'GIC prediction:\nzero events (no particle)', fontsize=12, ha='center', color='red',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.xlim(5, 2000)
plt.ylim(1e-55, 1e-44)
plt.xlabel('WIMP mass (GeV/c²)', fontsize=12)
plt.ylabel('Spin-independent cross-section (cm²)', fontsize=12)
plt.title('Direct detection: GIC vs. XENONnT limits', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gic_null_detection.png', dpi=150)
plt.show()