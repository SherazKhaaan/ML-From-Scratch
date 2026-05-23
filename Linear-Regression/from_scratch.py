import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, decay_rate=0.01, max_features_svd=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.decay_rate = decay_rate
        self.max_features_svd = max_features_svd
        self.tol = tolerance

        self.engine_used = None
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        
        """
        # 1) Initialise parameters
        # m = number of samples (rows), n = number of features (columns)
        m, n = X.shape 

        # 2) Use Normal Equation with SVD if number of features is small, otherwise use Gradient Descent
        if n <= self.max_features_svd:
            self.engine_used = "Normal Equation (SVD)"
            # 2.1) Add a column of ones to X for the bias term 
            X_b = np.c_[np.ones((m,1)), X]

            # 2.2) Calculate weights using the Normal Equation with SVD for stability
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y 

            # 2.3) Get bias and weights from theta
            self.bias = theta[0]
            self.weights = theta[1:]

        # 3) If number of features is large, use Gradient Descent
        else:
            self.engine_used = "Gradient Descent"
            # 3.1) start weights and bias at 0
            self.weights = np.zeros(n)
            self.bias = 0

            # 3.2) Gradient Descent Loop
            for i in range(self.n_iterations):
                
                # 3.2i) Predict for each sample 
                # Multiply each feature with corresponding weight and bias to find y_hat
                y_hat = np.dot(X, self.weights) + self.bias

                # 3.2ii) Errors for each sample 
                # Calculate difference between each corresponding y_hat and true y 
                error = y_hat - y 

                # 3.2iii) Gradients w.r.t to each feature 
                # Transpose input matrix to look at one feature at a time only 
                dw = np.dot(X.T, error) * (2/m)
                
                # 3.2iv) Sum the errors to get bias value
                db = np.sum(error) * (2/m) 

                # If the gradients are very small, we can stop early to save time
                if np.max(np.abs(dw)) < self.tol and np.abs(db) < self.tol:
                    print(f"Early stopping at iteration {i} due to small gradients.")
                    break

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db




    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 
    

    def score(self, X, y):
        """
        Calculates the R-squared value of the model
        """
        # 1) Get predictions
        y_hat = self.predict(X)

        # 2) Calculate the mean of the true target values
        y_mean = np.mean(y)

        # 3) Calculate SSR (sum of squares of residuals)
        ssr = np.sum((y - y_hat) ** 2)

        # 4) Calculate SST (total sum of squares)
        sst = np.sum((y - y_mean) ** 2)

        # 5) Calculate R-squared
        r2 = 1 - (ssr / sst) if sst != 0 else 0 # In case variance is 0 

        return r2

if __name__ == "__main__":
    # 3 features: [Years Experience, Projects, Cert Level]
    X = np.array([
        [1, 2, 0],
        [2, 5, 1],
        [3, 3, 1],
        [4, 8, 2],
        [5, 7, 3]
    ])
    y = np.array([45, 58, 63, 79, 88])

    # 1) Train Scikit-Learn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)

    # 2) Train Model from scratch 
    my_model = LinearRegressionScratch(learning_rate=0.01, n_iterations=10000, decay_rate=0.01)
    my_model.fit(X, y)


    # 3) Compare Results    
    print("--- WEIGHTS COMPARISON ---")
    print(f"Sklearn Weights: {sklearn_model.coef_}")
    print(f"Scratch Weights: {my_model.weights}\n")

    print(f"Sklearn Bias: {sklearn_model.intercept_:.2f}")
    print(f"Scratch Bias: {my_model.bias:.2f}\n")

    print("--- PREDICTION COMPARISON ---")
    new_candidate = np.array([[6, 10, 2]])
    print(f"Sklearn predicts: ${sklearn_model.predict(new_candidate)[0]:.2f}k")
    print(f"Scratch predicts: ${my_model.predict(new_candidate)[0]:.2f}k")

    print("\n--- ACCURACY (R-squared) ---")
    print(f"Sklearn R2: {sklearn_model.score(X, y):.4f}")
    print(f"Scratch R2: {my_model.score(X, y):.4f}")