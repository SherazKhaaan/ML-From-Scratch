# Omitted Variable Bias

When we leave out an important variable that affects the target, the model can produce biased coefficients and misleading predictions.

- Example: If we only use years of experience to predict salary but ignore education level, the model may overestimate or underestimate the true effect of experience.
- Concrete case: For a candidate with 6 years of experience but no projects and no certifications, represented as [6, 0, 0], the multi-feature model predicted 78.3k while the single-feature model predicted 87.3k.
- Why this happens: In the single-feature model, the coefficient for experience absorbs some of the effect of hidden variables. Since people with more experience often also have more projects and certifications, the model gives experience too much credit.
- Result: The experience coefficient dropped from 8.70k to 7.17k after adding more relevant features. The earlier model was biased because it omitted important variables.


# Feature Credit Attribution

Feature credit attribution describes how the model distributes the prediction effect across the features it can see.

- In the multi-feature model, the algorithm can separate the contribution of each feature.
- It reduces the experience coefficient and assigns part of the salary increase to projects and certifications instead.
- This explains why [6, 0, 0] gets a lower prediction in the multi-feature model: once the model can see projects and certifications, having zero of both is no longer silently treated as average or implied by experience.
- Key distinction: Omitted variable bias is the problem caused by leaving out an important feature. Feature credit attribution is the explanation of how the model redistributes importance when those features are included.


# Normal Equation / SVD limitations
- Normal Equation and SVD are exact methods for solving linear regression, but they can be computationally expensive for large datasets.
- They require inverting a matrix, which can be unstable if the matrix is not full rank (e.g., when features are highly correlated).
- If you have more columns than rows, the Normal Equation cannot be applied because the matrix to invert is not square. SVD can handle this case but may still be computationally intensive.

# Definition List

- Omitted variable bias: A bias that happens when a model leaves out a relevant feature that affects the target and is also related to the included features. Because of that omission, the coefficients of the included features can become misleading.
- Feature credit attribution: The way a model splits the prediction effect across the features it can see. When new relevant features are added, the model can reassign some of the credit that was previously lumped into another feature.
- Relationship between them: Omitted variable bias explains why a coefficient can be wrong in a simpler model. Feature credit attribution explains what happens to that coefficient after the missing variables are added.


# Use Cases of Linear Regression

Linear regression is used to predict a continuous target variable based on one or more input features.

- Predicting house prices based on features like size, location, and number of bedrooms.
- Predicting sales based on advertising spend and other marketing variables.
- Estimating the impact of different factors on student performance in educational settings.