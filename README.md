# Linear Regression from Scratch 

This is a foundational Machine Learning project implementing Linear Regression entirely from scratch using pure Python, NumPy and Calculus. No automated ML libraries (like Scikit-Learn) were used for the model training.
This repository compares two fundamental optimization algorithms: **Batch Gradient Descent (BGD)** and **Stochastic Gradient Descent (SGD)**, evaluating their performance on a standard Experience vs. Salary dataset.

## The Dataset
The model trains on a univariate dataset (`Experience-Salary.csv`) mapping months of professional experience to salary figures. The script fetches this data dynamically via a raw URL, meaning no local data downloads are required to run the code.

## Mathematical Foundation

The goal of this univariate linear regression is to fit a straight line through the data that minimizes the error between the predicted values and the actual values.

### 1. The Hypothesis
The model predicts the target variable ($\hat{y}$) based on the input feature ($x$), the weight/slope ($w$), and the bias/intercept ($b$):
$$\hat{y} = wx + b$$

### 2. The Cost Function (Mean Squared Error)
To measure how "wrong" the model's current line is, we use the Mean Squared Error (MSE) cost function, denoted as $J(w,b)$. It calculates the average squared difference between actual salaries ($y$) and predicted salaries ($\hat{y}$):

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

*(Where m is the total number of data points).*

### 3. The Gradients (Calculus)
To minimize the Cost Function, we calculate the partial derivatives (gradients) with respect to $w$ and $b$. These derivatives point in the direction of the steepest ascent; we take the negative of this to step downhill towards the minimum error.

Gradient for weight ($dw$):

$$dw = \frac{\partial J}{\partial w} = -\frac{1}{m} \sum_{i=1}^{m} x_i (y_i - \hat{y}_i)$$

Gradient for bias ($db$):

$$db = \frac{\partial J}{\partial b} = -\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)$$

### 4. The Update Rule
We update the parameters using a defined learning rate ($\alpha$), which dictates the size of the steps taken downhill:

$$w = w - \alpha \cdot dw$$ 

$$b = b - \alpha \cdot db$$

---

## Algorithms Implemented

### Batch Gradient Descent (BGD)
BGD calculates the error for the **entire dataset** before taking a single step (updating the weights). 
* **Pros:** Smooth, stable descent directly to the global minimum.
* **Cons:** Computationally expensive and slow for massive datasets.

### Stochastic Gradient Descent (SGD)
SGD updates the weights after evaluating a **single, randomly selected data point**. The gradients drop the summation and the $\frac{1}{m}$ term:

$$dw = -x_i (y_i - \hat{y}_i)$$
* **Pros:** Extremely fast iterations, capable of escaping local minima.
* **Cons:** The descent path is noisy and zig-zags, oscillating around the minimum rather than settling perfectly.

### Locally Weighted Regression (LWR)
*Also known as LOESS (Locally Estimated Scatterplot Smoothing)*

While standard Linear Regression is a **Parametric** algorithm (fitting a single, rigid global line to the entire dataset), Locally Weighted Regression is a **Non-Parametric** algorithm. It does not train a single model; instead, it trains hundreds of tiny, localized models on the fly to fit highly non-linear, curved datasets.

### 1. The Core Concept
To predict a value for a specific target point ($x_{target}$), LWR gives a "weight" to every training point in the dataset. 
* Training points **close** to the target get a weight near $1.0$ (high influence).
* Training points **far away** from the target get a weight near $0.0$ (ignored).

### 2. The Math: Gaussian Kernel (Influence Weights)
The algorithm calculates the influence weight ($w^{(i)}$) for each training data point using a bell-curve (Gaussian) formula:

$$w^{(i)} = \exp\left( -\frac{(x^{(i)} - x_{target})^2}{2\tau^2} \right)$$

### 3. The Math: Weighted Cost Function
Once the weights are calculated for a specific query point, we inject those weights into our standard Mean Squared Error cost function.

$$J(w,b) = \frac{1}{2} \sum_{i=1}^{m} w^{(i)} \cdot (y_i - (\theta x_i + b))^2$$

When we minimize this new cost function (either via Gradient Descent or the Normal Equation matrix shortcut), the algorithm automatically ignores distant points and perfectly fits a tiny straight line just for the local neighborhood of $x_{target}$. 

### 4. The Hyperparameter: Bandwidth ($\tau$)
The flexibility of the curve is controlled entirely by $\tau$ (Tau), known as the Bandwidth. 
* **Small $\tau$:** A narrow bell curve. The model only looks at immediate neighbors, resulting in a highly flexible, wiggly line (prone to overfitting).
* **Large $\tau$:** A wide bell curve. The model looks at almost all data points, resulting in a stiff, rigid line (prone to underfitting, acting like standard linear regression).

### Execution Note
Because the weights change depending on where $x_{target}$ is, LWR cannot be trained once. The entire algorithm (calculating weights and minimizing the cost function) must be re-run from scratch for **every single point** you want to predict.

---

## Evaluation & Results
To evaluate the "Accuracy", this project implements a basic division based accuracy calculation. 

$$Accuracy=\frac{\hat{y}}{y}$$


**Project Results:**
* **Batch Gradient Descent:** ~69.995\%
<p align="center">
 <img width="300" height="300" alt="Screenshot 2026-05-16 235845" src="https://github.com/user-attachments/assets/c19c5253-c7b8-4282-99da-74cb1a11791a" />
</p>

* **Stochastic Gradient Descent:** ~69.345\%
<p align="center">
 <img width="300" height="300" alt="Screenshot 2026-05-17 000200" src="https://github.com/user-attachments/assets/5aeb051f-0730-4ce9-8b6c-1cd90c6e77a7" />
</p>

*Conclusion: Both algorithms successfully mapped the linear relationship, proving that roughly 70% of the variance in a person's salary in this dataset is dictated strictly by their months of experience.*
