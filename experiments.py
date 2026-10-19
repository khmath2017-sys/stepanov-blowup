import numpy as np
from norms import L2_norm, H1_norm, stepanov_norm

def static_profile_test():
    """
    Reproduces Table 1 (Relative change under 10% noise for the sharp spike profile).
    Runs 200 independent realizations.
    """
    print("Running Static Profile Test (Table 1)...")
    N = 1000
    x = np.linspace(0, 1, N)
    u_clean = -np.tanh((x - 0.5) / 0.02)
    
    L2_changes, H1_changes, S_changes = [], [], []
    
    for _ in range(200):
        # Add 10% Gaussian noise
        noise = 0.1 * np.random.randn(len(x)) * np.max(np.abs(u_clean))
        u_noisy = u_clean + noise
        
        # Compute norms
        L2_c, L2_n = L2_norm(u_clean, x), L2_norm(u_noisy, x)
        H1_c, H1_n = H1_norm(u_clean, x), H1_norm(u_noisy, x)
        S_c = stepanov_norm(u_clean, x, radius=0.1, sigma=0.05)
        S_n = stepanov_norm(u_noisy, x, radius=0.1, sigma=0.05)
        
        # Calculate relative changes
        L2_changes.append(np.abs(L2_n - L2_c) / L2_c)
        H1_changes.append(np.abs(H1_n - H1_c) / H1_c)
        S_changes.append(np.abs(S_n - S_c) / S_c)
        
    print(f"L2 Mean rel. change: {np.mean(L2_changes)*100:.3f}%")
    print(f"H1 Mean rel. change: {np.mean(H1_changes)*100:.3f}%")
    print(f"Stepanov Mean rel. change: {np.mean(S_changes)*100:.3f}%")
    print("Test completed successfully.")

if __name__ == "__main__":
    static_profile_test()
