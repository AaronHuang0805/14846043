import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class GasFlowCorrector:
    def __init__(self, filename='data/hw9.csv'):
        self.filename = filename
        self.t, self.flow_velocity = self._load_or_generate_data()
      
        self.dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0.01

    def _load_or_generate_data(self):
       
        if os.path.exists(self.filename):
            print(f"Loading data from {self.filename}...")
            
            try:
                df = pd.read_csv(self.filename)
                data = df.to_numpy(dtype=np.float64)
                return data[:, 0], data[:, 1]
            except:
                
                data = pd.read_csv(self.filename, header=None).to_numpy()
                return data[:, 0], data[:, 1]
        else:
            print("CSV not found. Generating synthetic sensor data with drift...")
            
            t = np.arange(0, 36, 0.01)
           
            freq = 0.5 # Hz
            raw_signal = 200 * np.sin(2 * np.pi * freq * t)
            
            raw_signal += 50 * np.sin(4 * np.pi * freq * t)
            raw_signal += np.random.normal(0, 10, len(t))
           
            bias = -85.0 
            flow_with_bias = raw_signal + bias
            
            return t, flow_with_bias

    def solve_drift(self):
        
        vol_original = np.cumsum(self.flow_velocity) * self.dt
       
        bias_estimated = np.mean(self.flow_velocity)
        print(f"Detected Sensor Bias: {bias_estimated:.2f} ml/sec")
        
        velocity_corrected = self.flow_velocity - bias_estimated
       
        vol_corrected = np.cumsum(velocity_corrected) * self.dt
        
        return vol_original, vol_corrected

    def plot_results(self, vol_original, vol_corrected):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].plot(self.t, vol_original, 'r', label='Original Integration')
        axes[0].set_title('Problem: Accumulated Error (Drift)')
        axes[0].set_ylabel('Net Volume (ml)')
        axes[0].set_xlabel('time in seconds')
        axes[0].grid(True, alpha=0.3)
      
        axes[1].plot(self.t, vol_corrected, 'r', label='Corrected Flow')
        axes[1].set_title('Solution: Bias Removed')
        axes[1].set_ylabel('Net Volume (ml)')
        axes[1].set_xlabel('time in seconds')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hw9_result.png')
        plt.show()

if __name__ == "__main__":
 
    analyzer = GasFlowCorrector('data/hw9.csv')
    vol_drift, vol_fixed = analyzer.solve_drift()
    analyzer.plot_results(vol_drift, vol_fixed)
