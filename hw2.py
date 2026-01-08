import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

def hw2_svd_analysis():
   
    try:
        img = data.camera()
    except:
   
        print("Warning: scikit-image not found or error. Using random matrix.")
        img = np.random.rand(512, 512)

    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    
    A = img.astype(np.float64)
    
    print(f"Image shape: {A.shape}")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    eigenvalues = s ** 2
    
    energy_A_matrix = np.sum(A**2)
  
    energy_A_sigma = np.sum(eigenvalues)
    
    print(f"Total Energy (Matrix calc): {energy_A_matrix:.4f}")
    print(f"Total Energy (Sigma sum):   {energy_A_sigma:.4f}")

    # --- 3. 計算 SNR vs r (1 <= r <= 200) 並作圖 ---
    r_values = range(1, 201)
    snr_values = []

    for r in r_values:
   
        energy_noise = np.sum(eigenvalues[r:])
       
        if energy_noise <= 0:
            snr = float('inf')
        else:
            # SNR = 10 * log10( ||A||^2 / ||Nr||^2 )
            snr = 10 * np.log10(energy_A_sigma / energy_noise)
        
        snr_values.append(snr)

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, snr_values, color='red', label='ASNR[r]')
    plt.title('SNR vs Rank r (SVD Approximation)')
    plt.xlabel('r (Rank)')
    plt.ylabel('SNR (dB)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('snr_plot.png')
    plt.show()

    # --- 4. 程式驗證 (Verification) ---
    print("\n--- Verification of Formulas ---")
    
    r_check = 50
    print(f"Checking for r = {r_check}:")


    right_side_val = np.sum(eigenvalues[r_check:])

    S_r = np.diag(s[:r_check])
    A_bar = U[:, :r_check] @ S_r @ Vt[:r_check, :]
    
    Nr = A - A_bar
    
    left_side_val = np.linalg.norm(Nr, 'fro') ** 2

    print(f"1. Sum of tail eigenvalues (Theory): {right_side_val:.4f}")
    print(f"2. ||A - A_bar||_F^2 (Actual):       {left_side_val:.4f}")
    
    diff = abs(right_side_val - left_side_val)
    print(f"Difference: {diff:.4e}")
    
    if np.isclose(right_side_val, left_side_val):
        print(">> Verification SUCCESS: Formula holds true.")
    else:
        print(">> Verification FAILED.")

if __name__ == "__main__":
    hw2_svd_analysis()
