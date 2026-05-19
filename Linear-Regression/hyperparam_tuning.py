"""
Good brute force method for tiny datasets and few hyperparameters. 


"""

import numpy as np
from LinReg_scratch import LinearRegressionScratch

def grid_search(X, y):
    # 1) Define a range of learning rates to test
    learning_rates_to_test = [lr for lr in np.arange(0.001, 0.1, 0.001)]

    best_rate = None
    lowest_error = float('inf')


    # 2) Iterate through each learning rate, train a model, and evaluate its performance
    for lr in learning_rates_to_test:
        model = LinearRegressionScratch(learning_rate=lr, n_iterations=10000, decay_rate=0.01)

        # 2.1) Use a try-except to catch any exploding gradients 
        try: 
            model.fit(X,y) 
            predictions = model.predict(X)

            # 2.2) Calculate final MSE for this learning rate
            mse = np.mean((predictions - y) **2)

            # 2.3) Check if this is the best learning rate so far
            if mse < lowest_error:
                lowest_error = mse
                best_rate = lr
        
        except OverflowError:
            print(f"Learning rate {lr} caused overflow. Skipping.")
            continue
            
    
    return best_rate, lowest_error


if __name__ == "__main__":
    # 3 features: [Years Exp, Projects, Cert Level]
    X = np.array([
        [1, 2, 0],
        [2, 5, 1],
        [3, 3, 1],
        [4, 8, 2],
        [5, 7, 3]
    ])
    y = np.array([45, 58, 63, 79, 88])

    best_lr, best_mse = grid_search(X,y)
    print(f"Best Learning Rate: {best_lr}, with MSE: {best_mse}")