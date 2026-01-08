import numpy as np

def gram_schmidt(S1: np.ndarray):
        
    m, n = S1.shape
    S2 = np.zeros(S1.shape)
    
    for r in range(n):
       
        v = S1[:, r]
        
        u = v.copy()
        
        for i in range(r):
            e_prev = S2[:, i] # 取出之前的基底 e_i
           
            coefficient = np.dot(v, e_prev)
            
            u = u - coefficient * e_prev
        
        norm_u = np.linalg.norm(u)
       
        if norm_u > 1e-10:
            e = u / norm_u
        else:
            e = u 
  
        S2[:, r] = e

    return S2

# --- 以下為 5x5 矩陣測試 ---
if __name__ == "__main__":
    
    np.random.seed(42)
    
    print("--- 5x5 矩陣 Gram-Schmidt 測試 ---")
   
    test_S1 = np.random.rand(5, 5)
    
    print("\n原始矩陣 S1 (5x5):")
    print(test_S1)
    
    result_S2 = gram_schmidt(test_S1)
    
    print("\n正交化結果 S2 (5x5):")
    
    np.set_printoptions(precision=4, suppress=True)
    print(result_S2)
    
    identity_check = result_S2.T @ result_S2
    
    print("\n驗證正交性 (S2.T @ S2 應為單位矩陣):")
    print(identity_check)
    
    is_orthogonal = np.allclose(identity_check, np.eye(5))
    print(f"\n驗證結果: {'成功' if is_orthogonal else '失敗'}")
