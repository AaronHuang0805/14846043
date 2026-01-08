import numpy as np
import matplotlib.pyplot as plt

def hw3_fourier_fit():
   
    pts = 50
    x = np.linspace(-2, 2, pts)
    y = np.zeros(x.shape)

    pts2 = pts // 2
    y[0:pts2] = -1
    y[pts2:] = 1

    T0 = np.max(x) - np.min(x)
    f0 = 1.0 / T0
    omega0 = 2.0 * np.pi * f0
    n = 5
  
    col0 = np.ones((pts, 1))
   
    cos_terms = []
    for k in range(1, n + 1):
        cos_terms.append(np.cos(k * omega0 * x).reshape(-1, 1))
   
    sin_terms = []
    for k in range(1, n + 1):
        sin_terms.append(np.sin(k * omega0 * x).reshape(-1, 1))
    
 
    X = np.hstack([col0] + cos_terms + sin_terms)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    S_inv = np.diag(1.0 / s)
   
    a = Vt.T @ S_inv @ U.T @ y

    y_bar = X @ a

    plt.figure(figsize=(8, 6))
   
    plt.plot(x, y_bar, 'g-', label='predicted values')
    
    plt.plot(x, y, 'b-', label='true values')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Fourier Series Fitting (n={n})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)
   
    plt.savefig('hw3_result.png')
    plt.show()

if __name__ == "__main__":
    hw3_fourier_fit()
