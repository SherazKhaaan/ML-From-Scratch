import numpy as np
from sklearn.linear_model import LinearRegression


def build_regression_model(X, y):
	"""
	Given input features X and target values y, this function trains a Linear Regression model and prints the learned parameters.
	Parameters:
		X (numpy array): A 2D array of shape (m, n) containing the input features, where m is the number of samples and n is the number of features.
		y (numpy array): A 1D array of shape (m,) containing the target values.
	Returns:
		coefficients (numpy array): The learned coefficients for each feature.
		intercept (float): The learned intercept of the model.

	
	"""
    
	# 1) Create a LinearRegression model instance
	model = LinearRegression()
	
	# 2) Train the model (finds best fit line)
	model.fit(X, y)
	
	# 3) Extract the learned parameters
	coefficients = model.coef_  
	intercept = model.intercept_
	
	# 4) Print the learned parameters
	print(f"Learned intercept: {intercept: .2f}k")
	
	# Loop through all the coefficients and print them 
	# There is one coefficient for each feature 
	for i, coef in enumerate(coefficients):
		print(f"Learned coefficient for feature {i}:{coef: .2f} k per year")
	

	return coefficients, intercept 
    


def predict_salary(features, coefficient, intercept):
	"""
	Given a set of features, coefficient, and intercept, this function predicts the salary using the linear regression formula.

	Parameters:
		features (numpy array): A 1D array containing the values of the features for which we want to predict the salary.
		coefficient (numpy array): The learned coefficients from the linear regression model.
		intercept (float): The learned intercept from the linear regression model.
	
	Returns:
		float: The predicted salary based on the input features.
	"""
	# Loop through all the coefficient and features to compute the prediction 
	model_prediction = 0
	for coefficient, feature in zip(coefficient, features):
		model_prediction += coefficient * feature 
	
	# Add the intercept to the prediction
	model_prediction += intercept

	return model_prediction




if __name__ == "__main__":
	
	'''
	Dummy data (X = years of experience, y = salary in thousands)
		- X is a 2D array (m, n) where m is the number of samples and n is the number of features.
		- m represents the number of data points, and n represents the number of features (in this case, just 1 feature: years of experience).
		- y is a 1D array (m,) where m is the number of samples. Each element corresponds to the target value for the respective row in X. 

	'''
	X = np.array([[1], [2], [3], [4], [5]])
	y = np.array([45, 50, 62, 71, 78])


	# 1) Building a simple linear regression model with one feature (years of experience)
	coefficient, intercept = build_regression_model(X, y)

	# 2) Example prediction for 6 years of experience
	print("\nPredicting salary for 6 years of experience with the simple linear regression model:")
	predicted_salary = predict_salary(np.array([6]), coefficient, intercept)
	print(f"Predicted salary for 6 years of experience: {predicted_salary: .1f}k")


	''' 
	Dummy Data with 3 features: [Years Experience, Projects, Certification Level]
		- Shape is (5, 3) -> 5 samples, 3 features
	'''

	X_multi = np.array([
		[1, 2, 0],
		[2, 5, 1],
		[3, 3, 1],
		[4, 8, 2],
		[5, 7, 3]
	])
	y_multi = np.array([45, 58, 63, 79, 88])

	# 3) Building a model with multiple features
	coefficient, intercept = build_regression_model(X_multi, y_multi)
	print("\nBuilding a linear regression model with multiple features:")

	# 4) Example prediction for [6 years experience, 10 projects, certification level 2]
	predicted_salary_multi = predict_salary(np.array([6, 10, 2]), coefficient, intercept)
	print(f"Predicted salary for [6 years experience, 10 projects, certification level 2]: {predicted_salary_multi: .1f}k")
	
	predicted_salary_multi = predict_salary(np.array([6, 0, 0]), coefficient, intercept)
	print(f"Predicted salary for [6 years experience, 0 projects, certification level 0]: {predicted_salary_multi: .1f}k")