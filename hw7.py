import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def model_predict(x, w):
   
    return w[0] + w[1] * np.sin(w[2] * x + w[3])

def compute_cost(x, y, w):
    y_pred = model_predict(x, w)
    error = y - y_pred
    return np.sum(error ** 2)

def gradient_analytic(x, y, w):
    
    u = w[2] * x + w[3]
    y_pred = w[0] + w[1] * np.sin(u)
    error = y - y_pred # error corresponds to e_i
    
    grad = np.zeros_like(w)
    
    grad[0] = -2.0 * np.sum(error * 1.0)
  
    grad[1] = -2.0 * np.sum(error * np.sin(u))
 
    grad[2] = -2.0 * np.sum(error * w[1] * x * np.cos(u))

    grad[3] = -2.0 * np.sum(error * w[1] * np.cos(u))
    
    return grad

def gradient_numeric(x, y, w):
    
    epsilon = 1e-8
    grad = np.zeros_like(w)
    current_cost = compute_cost(x, y, w)
    
    for k in range(len(w)):
        w_perturbed = w.copy()
        w_perturbed[k] += epsilon
        cost_perturbed = compute_cost(x, y, w_perturbed)
        
        grad[k] = (cost_perturbed - current_cost) / epsilon
        
    return grad

def train_model(x, y, w_init, grad_func, alpha=0.05, max_iters=500):
   
    w = w_init.copy()
    cost_history = []

    for i in range(max_iters):
      
        grad = grad_func(x, y, w)
     
        w = w - alpha * grad
        
        if i % 100 == 0:
            cost = compute_cost(x, y, w)
            cost_history.append(cost)
            
    return w


def get_data(filename='data/hw7.csv'):
  
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        data = df.to_numpy(dtype=np.float64)
        return data[:, 0], data[:, 1]
    else:
        print("Warning: CSV not found. Generating dummy data compatible with the problem.")
   
        np.random.seed(42)
        x_dummy = np.linspace(0, 1, 15)
  
        y_dummy = np.sin(2 * np.pi * x_dummy) + np.random.normal(0, 0.1, 15)
        return x_dummy, y_dummy

if __name__ == "__main__":
   
    x, y = get_data()
    
    w_init = np.array([-0.1607108, 2.0808538, 0.3277537, -1.5511576])
    
    ALPHA = 0.05
    MAX_ITERS = 500

    print("--- Starting Gradient Descent ---")
    print("1. Training with Analytic Gradient...")
    w_analytic = train_model(x, y, w_init, gradient_analytic, ALPHA, MAX_ITERS)
    print(f"   Final w: {w_analytic}")

    print("2. Training with Numeric Gradient...")
    w_numeric = train_model(x, y, w_init, gradient_numeric, ALPHA, MAX_ITERS)
    print(f"   Final w: {w_numeric}")

    plt.figure(figsize=(10, 6))
    
    plt.scatter(x, y, color='black', label='Data points')
   
    x_smooth = np.linspace(min(x), max(x), 200)
   
    plt.plot(x_smooth, model_predict(x_smooth, w_analytic), 
             'b-', linewidth=3, label='Analytic method')
   
    plt.plot(x_smooth, model_predict(x_smooth, w_numeric), 
             'r--', linewidth=2, label='Numeric method')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Non-linear Regression: Analytic vs Numeric Gradient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('hw7_result.png')
    plt.show()
