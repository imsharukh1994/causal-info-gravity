import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sprinkle_2d(N, L):
    """Sprinkle N points uniformly in [0,L]x[0,L]."""
    return np.random.rand(N, 2) * L

def diamond_count(pts, v, eps):
    """Number of points in causal diamond centered at v with half-height eps (2D Minkowski)."""
    tv, xv = pts[v]
    count = 0
    for i, (t, x) in enumerate(pts):
        dt = t - tv
        dx = x - xv
        ds2 = dt*dt - dx*dx
        if ds2 >= 0 and abs(dt) <= eps and ds2 <= eps*eps:
            count += 1
    return count

def phi_v(pts, v, eps):
    n = diamond_count(pts, v, eps)
    if n == 0:
        return 0.0
    return 1.0 / n

def run_simulation(N_vals, L=10.0, eps=0.5, n_real=10):
    results = []
    for N in N_vals:
        rho = N / (L*L)
        print(f"\nN={N}, ρ={rho:.2f}")
        all_phi = []
        for rep in range(n_real):
            pts = sprinkle_2d(N, L)
            for v in range(N):
                all_phi.append(phi_v(pts, v, eps))
        all_phi = np.array(all_phi)
        mean_phi = np.mean(all_phi)
        var_phi = np.var(all_phi)
        results.append((rho, mean_phi, var_phi))
        print(f"  mean Φ = {mean_phi:.6f}, var Φ = {var_phi:.6e}")
    return np.array(results)

if __name__ == "__main__":
    N_list = [100, 200, 400, 800]
    data = run_simulation(N_list, L=10.0, eps=0.5, n_real=10)
    rho = data[:,0]
    mean_phi = data[:,1]
    var_phi = data[:,2]
    
    inv_rho = 1.0 / rho
    def linear(x, A): return A * x
    popt, _ = curve_fit(linear, inv_rho, mean_phi)
    A = popt[0]
    print(f"\nFit: <Φ> = {A:.4f} / ρ")
    
    delta = var_phi * rho**2
    print("\nδ = var_phi * ρ^2 :", delta)
    print(f"Mean δ = {np.mean(delta):.4f} ± {np.std(delta):.4f}")
    
    plt.figure(figsize=(6,4))
    plt.errorbar(inv_rho, mean_phi, fmt='o', capsize=3)
    x_fit = np.linspace(0, max(inv_rho), 100)
    plt.plot(x_fit, A * x_fit, 'r--', label=f'<Φ> = {A:.3f}/ρ')
    plt.xlabel('1/ρ')
    plt.ylabel(r'$\langle \Phi \rangle$')
    plt.title('Scaling of information flux (2D Minkowski)')
    plt.legend()
    plt.grid(True)
    plt.savefig('flux_scaling.png', dpi=150)
    plt.show()