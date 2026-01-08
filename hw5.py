import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os

class ConcentrationAnalysis:
    def __init__(self, filename='hw5.csv'):
        self.filename = filename
        self.df = self._load_or_create_data()
        
    def _load_or_create_data(self):
        
        if os.path.exists(self.filename):
            print(f"Loading data from {self.filename}...")
            return pd.read_csv(self.filename)
        else:
            print(f"Warning: {self.filename} not found. Generating dummy data for demonstration.")
  
            t = np.logspace(0.5, 2.3, 20)
            noise = np.random.normal(0, 0.05, len(t))
            
            c = 50 * np.power(t, -0.6) * np.exp(noise)
            return pd.DataFrame({'Time': t, 'Concentration': c})

    def fit_and_plot(self, ax, transform_x=None, transform_y=None, 
                     model_type='poly', degree=1, title="", show_raw=True):
        
        x_raw = self.df['Time'].values
        y_raw = self.df['Concentration'].values
        
        x_data = transform_x(x_raw) if transform_x else x_raw
        y_data = transform_y(y_raw) if transform_y else y_raw
       
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        X = x_sorted.reshape(-1, 1)
        
        if model_type == 'poly':
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y_sorted)
            y_pred = model.predict(X_poly)
            model_desc = f"Polynomial Regression (deg={degree})"
        else:
            # Linear regression (Standard)
            model = LinearRegression()
            model.fit(X, y_sorted)
            y_pred = model.predict(X)
            model_desc = "Linear Regression"

        r2 = r2_score(y_sorted, y_pred)

   
        if show_raw:
            ax.scatter(x_sorted, y_sorted, color='red', s=10, label='Data Points')
        
        ax.plot(x_sorted, y_pred, color='blue', linewidth=1.5, 
                label=f'Prediction\n({model_desc})\n$R^2$={r2:.3f}')
        
        ax.set_title(title)
        ax.set_xlabel("Log(Time)" if transform_x else "Time")
        ax.set_ylabel("Log(Concentration)" if transform_y else "Concentration")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return model, r2

    def run_analysis(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        print("--- Regression 1: Polynomial on Raw Data ---")
        self.fit_and_plot(
            ax=axes[0],
            transform_x=None, 
            transform_y=None,
            model_type='poly', 
            degree=5,  # 使用 5 階多項式來示範擬合
            title="Raw Data: Concentration vs Time\n(Polynomial Fit)"
        )

        print("--- Regression 2: Linear on Log-Log Data ---")
        self.fit_and_plot(
            ax=axes[1],
            transform_x=np.log10, 
            transform_y=np.log10,
            model_type='linear',
            degree=1,
            title="Log-Log Scale: Log(Conc) vs Log(Time)\n(Power Law Fit)"
        )
        
        plt.tight_layout()
        plt.savefig('hw5_result.png')
        plt.show()

if __name__ == "__main__":
    analysis = ConcentrationAnalysis('hw5.csv')
    analysis.run_analysis()
