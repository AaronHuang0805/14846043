import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os

class NonLinearClassifier:
    def __init__(self, filename='hw8.csv'):
        self.filename = filename
        
    def get_data(self):
        
        if os.path.exists(self.filename):
            print(f"Loading data from {self.filename}...")
            df = pd.read_csv(self.filename)
        
            data = df.to_numpy()
            X = data[:, :2]  
            y = data[:, -1]  
            return X, y
        else:
            print("CSV not found. Generating synthetic 'Flower' dataset...")
            return self._generate_flower_data(n_samples=600)

    def _generate_flower_data(self, n_samples):
        
        np.random.seed(42)
       
        r = np.random.uniform(2, 10, n_samples)         
        theta = np.random.uniform(0, 2*np.pi, n_samples) 
        
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
     
        sector_idx = (theta // (np.pi / 3)).astype(int)
        y = sector_idx % 2
    
        angle_mod = theta % (np.pi / 3)
        margin = 0.2 
        mask = (angle_mod > margin) & (angle_mod < (np.pi/3 - margin))
        
        
        X += np.random.normal(0, 0.4, X.shape)
        
        return X[mask], y[mask]

    def plot_decision_boundary(self, model, X, y):
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
   
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = 0.1  
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
   
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
        
        unique_labels = np.unique(y)
        colors = ['blue', 'red']
        
        for i, label in enumerate(unique_labels):
            idx = (y == label)
            ax.scatter(X[idx, 0], X[idx, 1], c=colors[i % 2], 
                       label=f'Class {int(label)+1} ($\omega_{int(label)+1}$)',
                       edgecolor='white', s=30)
            
        plt.title('Non-Linear Classification (SVM with RBF Kernel)')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='upper right')
        plt.grid(False)
        
        plt.savefig('hw8_result.png')
        plt.show()

    def run(self):
    
        X, y = self.get_data()
        print("Training SVM (RBF Kernel)...")
        clf = SVC(kernel='rbf', C=10, gamma='scale')
        clf.fit(X, y)
     
        self.plot_decision_boundary(clf, X, y)

if __name__ == "__main__":
    app = NonLinearClassifier('hw8.csv')
    app.run()
