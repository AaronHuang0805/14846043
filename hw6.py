import numpy as np
import matplotlib.pyplot as plt

def generate_dataset():
   
    np.random.seed(42) 

 
    mean1 = np.array([0, 5])
    sigma1 = np.array([[0.3, 0.2], [0.2, 1]])
    N1 = 200
    X1 = np.random.multivariate_normal(mean1, sigma1, N1)

    mean2 = np.array([3, 4])
    sigma2 = np.array([[0.3, 0.2], [0.2, 1]])
    N2 = 100
    X2 = np.random.multivariate_normal(mean2, sigma2, N2)
    
    return X1, X2

def fisher_lda_direction(X1, X2):
    
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)
    
    diff1 = X1 - m1
    diff2 = X2 - m2
    Sw = (diff1.T @ diff1) + (diff2.T @ diff2)
    
    w_unscaled = np.linalg.inv(Sw) @ (m1 - m2)
    
    w = w_unscaled / np.linalg.norm(w_unscaled)
    
    return w

def project_data(X, w):
   
    scalar_proj = X @ w  # shape: (N, )
    
    vec_proj = scalar_proj[:, np.newaxis] @ w[np.newaxis, :]
    
    return vec_proj

def visualize_lda(X1, X2, w):
 
    P1 = project_data(X1, w)
    P2 = project_data(X2, w)
    
    plt.figure(figsize=(8, 6))
   
    plt.scatter(X1[:, 0], X1[:, 1], c='r', s=10, label='Class 1 (Original)')
    plt.scatter(X2[:, 0], X2[:, 1], c='g', s=10, label='Class 2 (Original)')
    
    plt.scatter(P1[:, 0], P1[:, 1], c='r', s=15, alpha=0.3, marker='.', label='Projected')
    plt.scatter(P2[:, 0], P2[:, 1], c='g', s=15, alpha=0.3, marker='.')
    
    all_proj = np.vstack([P1, P2])
    x_min, x_max = all_proj[:, 0].min(), all_proj[:, 0].max()
    y_min, y_max = all_proj[:, 1].min(), all_proj[:, 1].max()
  
    plt.plot([x_min - 0.5, x_max + 0.5], 
             [y_min - 0.5 * (w[1]/w[0]), y_max + 0.5 * (w[1]/w[0])], 
             'k--', alpha=0.5, linewidth=1, label='Projection Vector w')

    plt.title(f"Fisher's Linear Discriminant Analysis\nw = [{w[0]:.3f}, {w[1]:.3f}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('hw6_result.png')
    plt.show()

if __name__ == "__main__":
   
    data1, data2 = generate_dataset()
    best_w = fisher_lda_direction(data1, data2)
    visualize_lda(data1, data2, best_w)
