import numpy as np

def scale_to_range(X: np.ndarray, to_range=(0, 1), byrow=False):
       
    if X.ndim == 1:
        axis = 0
    else:
        axis = 1 if byrow else 0

   
    X_min = X.min(axis=axis, keepdims=True)
    X_max = X.max(axis=axis, keepdims=True)
    
    a, b = to_range
    
    denom = X_max - X_min
    
    denom[denom == 0] = 1.0 
    
    Y = a + ((X - X_min) / denom) * (b - a)
    
    return Y

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    print("--- Test Case 1: 1D Array ---")
    data_1d = np.array([3, 5, 1, 4])
    result_1d = scale_to_range(data_1d, to_range=(5, 7))
    print(f"Input:  {data_1d}")
    print(f"Output: {result_1d}") 
    # Expected: [6.  7.  5.  6.5]

    print("\n--- Test Case 2: 2D Array ---")
    data_2d = np.array([
        [1, 4, 3, 6],
        [6, 7, 9, 5],
        [11, 14, 13, 16]
    ])
    print("Input Matrix:")
    print(data_2d)

    print("\n(A) Row-wise mapping to [0, 1]:")
    result_row = scale_to_range(data_2d, to_range=(0, 1), byrow=True)
    print(result_row)
    # Row 1 (1~6 -> 0~1): [0. 0.6 0.4 1.]
    # Row 2 (5~9 -> 0~1): [0.25 0.5 1. 0.]
    
    print("\n(B) Column-wise mapping to [0, 1]:")
    result_col = scale_to_range(data_2d, to_range=(0, 1), byrow=False)
    print(result_col)
    # Col 1 (1~11 -> 0~1): [0. 0.5 1.]
