# Linear Regression from Scratch — Learning Plan

This plan teaches you the **math** behind linear regression and the **NumPy functions** you'll need. No code samples — you write everything yourself. Each stage ends with a **checkpoint** so you can verify your output before moving on.

---

## The Big Picture

Linear regression tries to draw the "best" straight line (or hyperplane, if you have many features) through a cloud of data points.

You have:
- **Inputs** `X` — a matrix of features, shape `(m, n)` where `m` is the number of examples and `n` is the number of features.
- **Outputs** `y` — a vector of targets, shape `(m,)`.

You want to find:
- **Weights** `w` — a vector, shape `(n,)`, one weight per feature.
- **Bias** `b` — a single number.

Such that for any example `x`, your prediction is:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

In matrix form, for *all* examples at once:

$$\hat{y} = Xw + b$$

The whole game is: **find `w` and `b` that make `ŷ` as close to `y` as possible.**

---

## Stage 0 — Setup & Data

### Math
Nothing yet. Just get data.

### What to do
1. Generate a synthetic dataset where you already know the "true" weights and bias. This is gold for debugging — if your model recovers the truth, you know it works.
2. Pick something like: 100 examples, 1 or 2 features, true weights e.g. `[3.0, -2.0]`, true bias e.g. `5.0`, add a small amount of Gaussian noise to `y` so it's realistic.

### NumPy functions to know
- `np.random.seed(...)` — makes results reproducible. Use it.
- `np.random.randn(m, n)` — random numbers from a standard normal distribution. Good for `X`.
- `np.random.normal(loc, scale, size)` — Gaussian noise.
- `@` operator or `np.dot(...)` — matrix multiplication. (`X @ w` is the cleanest way.)
- `.shape` — always sanity-check shapes.

### Checkpoint
- `X.shape` should be `(m, n)`.
- `y.shape` should be `(m,)` — a 1D vector, **not** `(m, 1)`. Mixing these up causes silent broadcasting bugs later.
- `print(X[:5])` and `print(y[:5])` — does it look sensible?

---

## Stage 1 — The Prediction (Forward Pass)

### Math
Given current `w` and `b`, compute predictions for every example:

$$\hat{y} = Xw + b$$

This is one matrix-vector multiplication plus a scalar added to every element.

### What to do
1. Initialize `w` to zeros (or small random numbers) of shape `(n,)`.
2. Initialize `b` to `0.0`.
3. Write a function that takes `X`, `w`, `b` and returns `ŷ`.

### NumPy functions to know
- `np.zeros(shape)` — initialize weights.
- `X @ w` — does `(m, n) @ (n,) → (m,)`. The shape arithmetic matters; get it wrong and you'll get a cryptic error.
- Scalar `+ b` broadcasts automatically across the vector.

### Checkpoint
- With `w = zeros` and `b = 0`, your prediction should be **all zeros**, shape `(m,)`.
- Set `w` and `b` to the *true* values you used in Stage 0 — predictions should match `y` very closely (within the noise you added).

---

## Stage 2 — The Loss (Mean Squared Error)

### Math
We need a single number that says "how wrong are we?" The standard choice is **Mean Squared Error**:

$$L = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Why squared?
- Penalizes big errors more than small ones.
- Always positive.
- Smooth and differentiable — important for gradient descent.

Some textbooks use `1/(2m)` instead of `1/m`. The `2` is purely a convenience that cancels when you take the derivative. Either works. Pick one and stick with it.

### What to do
Write a function that takes `ŷ` and `y` and returns the loss as a single number.

### NumPy functions to know
- `(yhat - y)` — element-wise subtraction.
- `**2` or `np.square(...)` — element-wise squaring.
- `np.mean(...)` — averages all elements. Saves you from writing `np.sum(...) / m`.

### Checkpoint
- If `ŷ == y`, loss should be exactly `0.0`.
- If `ŷ` is all zeros, loss should equal `np.mean(y**2)`. Compute that by hand and compare.
- Loss must be a **scalar**, not an array.

---

## Stage 3 — The Gradients (The Most Important Part)

### Math
Gradient descent needs to know: *how does the loss change if I nudge `w` or `b`?*

Starting from:

$$L = \frac{1}{m} \sum (Xw + b - y)^2$$

Take partial derivatives. Let `e = ŷ - y` (the error vector, shape `(m,)`).

**Gradient with respect to `w`:**

$$\frac{\partial L}{\partial w} = \frac{2}{m} X^T e$$

This is a vector of shape `(n,)` — one gradient per weight.

**Gradient with respect to `b`:**

$$\frac{\partial L}{\partial b} = \frac{2}{m} \sum_i e_i$$

This is a scalar.

### Why those formulas (quick intuition)
- `Xᵀ e` asks: "for each feature, how correlated is that feature with our current errors?" If feature `j` is positively correlated with the errors, increasing `w_j` makes the error bigger — so we should decrease `w_j`.
- The bias gradient is just the average error — if you're overshooting on average, shrink `b`.

If you dropped the `2` by using `1/(2m)` in the loss, drop it from the gradients too.

### What to do
Write a function that takes `X`, `y`, `ŷ` and returns `dw` (shape `(n,)`) and `db` (scalar).

### NumPy functions to know
- `X.T` or `X.transpose()` — transpose. Turns `(m, n)` into `(n, m)`.
- `X.T @ e` — gives shape `(n,)`. Sanity-check this in your head: `(n, m) @ (m,) → (n,)`. ✓
- `np.sum(e)` — for the bias gradient.
- `e.mean()` — `np.mean(e)` is equivalent to `np.sum(e) / m`, so you can fold the `1/m` directly in.

### Checkpoint
- Shapes: `dw.shape == w.shape == (n,)`. `db` is a scalar.
- **Finite-difference check** (this is the most powerful debugging tool you have):
  1. Pick one weight `w_j`. Compute loss `L`.
  2. Add a tiny `ε` (e.g. `1e-6`) to `w_j`, compute loss `L_plus`.
  3. Subtract `ε`, compute loss `L_minus`.
  4. The numerical gradient is `(L_plus - L_minus) / (2 * ε)`.
  5. This should match `dw[j]` to ~5–6 decimal places. If it doesn't, your analytical gradient is wrong. **Do this. It will save you hours.**

---

## Stage 4 — The Update Step (Gradient Descent)

### Math
Now we take a step *downhill* — opposite to the gradient — scaled by a **learning rate** `α`:

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

The learning rate controls step size:
- Too small → painfully slow.
- Too big → overshoots, loss explodes (you'll see NaNs or loss growing).
- Sweet spot for your synthetic data is probably `0.01` to `0.1`. Tune by watching the loss curve.

### What to do
1. Write a training loop. Each iteration: forward pass → loss → gradients → update.
2. Store the loss every iteration so you can plot it.
3. Run for some number of iterations (start with 1000).

### NumPy functions to know
- Just `-=` and scalar multiplication. Nothing fancy.
- `list.append(...)` for tracking loss history (or preallocate an array).

### Checkpoint
- Loss should **decrease monotonically** (or close to it) over iterations. If it bounces up or explodes, your learning rate is too high.
- After enough iterations, `w` should approach the **true weights** you used in Stage 0, and `b` should approach the **true bias**. This is the real test.
- Try `matplotlib.pyplot.plot(loss_history)` — the curve should drop fast at first, then flatten.

---

## Stage 5 — The Closed-Form Solution (The Normal Equation)

### Math
Gradient descent is iterative. But for linear regression specifically, there's an exact, one-shot solution. Set the gradient to zero and solve algebraically. You get the **normal equation**:

$$w^* = (X^T X)^{-1} X^T y$$

This is the optimal weights, computed directly, no iteration.

### Handling the bias
The normal equation as written has no bias term. The trick: prepend a column of `1`s to `X`. Then the first "weight" *is* the bias. So `X` becomes shape `(m, n+1)` and `w` becomes shape `(n+1,)`.

### Why bother with gradient descent then?
- The normal equation requires inverting an `(n, n)` matrix — that's `O(n³)`. Fine for small `n`, terrible for `n = 10,000`.
- Gradient descent scales to huge datasets and works for models where no closed-form exists (every neural net).
- But for linear regression, the normal equation is a perfect sanity check: your gradient-descent solution should converge to the same answer.

### What to do
Implement the normal equation in a single line (well, two — the prepend, then the formula).

### NumPy functions to know
- `np.ones((m, 1))` — column of ones.
- `np.hstack(...)` or `np.concatenate(..., axis=1)` — glue the ones column to `X`.
- `np.linalg.inv(M)` — matrix inverse. Works but is **numerically unstable** and slow.
- `np.linalg.solve(A, b)` — solves `Ax = b` directly. **Prefer this.** Use it as `np.linalg.solve(X.T @ X, X.T @ y)`.
- Even better: `np.linalg.lstsq(X, y, rcond=None)` — handles rank deficiency and is the production-quality answer.

### Checkpoint
- Compare the weights from the normal equation to the weights from gradient descent (Stage 4). They should be nearly identical (within ~1e-4 or so, depending on how long you trained GD).
- Both should be close to the *true* weights from Stage 0.

---

## Stage 6 — Evaluation Metrics

### Math
Loss is fine for training, but you'll want human-friendly metrics:

**Mean Absolute Error:** $\text{MAE} = \frac{1}{m} \sum |\hat{y}_i - y_i|$

**Root Mean Squared Error:** $\text{RMSE} = \sqrt{\text{MSE}}$ — same units as `y`, easier to interpret than MSE.

**R² (coefficient of determination):**

$$R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2}$$

Where `ȳ` is the mean of `y`. R² = 1 is perfect, R² = 0 means your model is no better than predicting the mean, R² < 0 means you're worse than predicting the mean.

### NumPy functions to know
- `np.abs(...)`, `np.sqrt(...)`, `np.mean(...)`, `np.sum(...)`.

### Checkpoint
- On your synthetic data, R² should be very close to 1 (say, > 0.95). If not, training didn't converge.

---

## When You Hand Me Your Code

Once you've gone through all stages, share the code and I'll review for:
1. **Correctness** — does the math match what's above?
2. **Vectorization** — any sneaky Python loops that should be NumPy operations?
3. **Numerical stability** — `inv` vs `solve`, divisions by zero, etc.
4. **Idiom** — clean shape handling, readable variable names, no redundant computation.

---

## Cheat Sheet of Shapes (Keep This Handy)

| Object | Shape |
|---|---|
| `X` | `(m, n)` |
| `y` | `(m,)` |
| `w` | `(n,)` |
| `b` | scalar |
| `ŷ = Xw + b` | `(m,)` |
| `e = ŷ - y` | `(m,)` |
| `dw = (2/m) Xᵀ e` | `(n,)` |
| `db = (2/m) sum(e)` | scalar |

If a shape doesn't match this table, stop and fix it before continuing.

---

## Order of Operations

1. Stage 0: data ✓ before anything
2. Stage 1: forward pass ✓ before loss
3. Stage 2: loss ✓ before gradients
4. Stage 3: gradients (**run the finite-difference check!**) ✓ before training
5. Stage 4: training loop ✓ before claiming victory
6. Stage 5: normal equation ✓ as a cross-check
7. Stage 6: metrics ✓ to quantify how well you did

Go one stage at a time. Verify the checkpoint. Then move on.
