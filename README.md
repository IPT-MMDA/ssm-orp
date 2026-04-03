# Data Analysis: Quadratic Constraints (QC) for Nonlinearities

This repository contains the mathematical framework for representing nonlinear functions (like ReLU) using Quadratic Constraints. This approach is essential for the robust analysis and verification of neural networks and control systems.

---

## 1. Quadratic Constraint (QC) for Function $\phi$

For a function $\phi: \mathbb{R}^n \to \mathbb{R}^n$, the QC is defined as:

$$
\begin{pmatrix}x \\ \phi(x) \\ 1\end{pmatrix}^\top Q \begin{pmatrix}x \\ \phi(x) \\ 1\end{pmatrix} \ge 0, \quad \forall x \in X
$$

* **$Q \in \mathcal{S}^{2n+1}$**: A symmetric matrix defining the constraint.
* **$X \subset \mathbb{R}^n$**: The input domain.
* **Function Graph**: $G(\phi) = \{ (x,y) \mid y = \phi(x), x \in X \}$.
* **Core Idea**: The QC defines a quadratic region in space that "encloses" the function's graph.

---

## 2. Sector-Bounded Nonlinearity

A scalar function $\phi: \mathbb{R} \to \mathbb{R}$ is sector-limited in the interval $[\alpha, \beta]$ if:

$$(\phi(x) - \alpha x)(\phi(x) - \beta x) \le 0$$

* **Geometry**: The graph of the function lies strictly between the lines $y = \alpha x$ and $y = \beta x$.
* **Vector Case**: For $\phi: \mathbb{R}^n \to \mathbb{R}^n$:
  $$(\phi(x) - K_1 x)^\top (\phi(x) - K_2 x) \le 0, \quad K_2 - K_1 \ge 0$$

---

## 3. Slope-Restricted Nonlinearity

For a function $\phi$ with slope-restricted nonlinearity in the interval $[\alpha, \beta]$:

$$\bigl( \phi(x) - \phi(x') - \alpha(x - x') \bigr)^\top \bigl( \phi(x) - \phi(x') - \beta(x - x') \bigr) \le 0, \quad \forall x, x' \in X$$

* **One-dimensional interpretation**:
  $$\alpha \le \frac{\phi(x) - \phi(x')}{x - x'} \le \beta$$

---

## 4. Repeated Nonlinearities 

When $\phi(x)$ is applied element-wise, $\phi(x) = [\phi(x_1), \dots, \phi(x_n)]^\top$, we reduce conservatism by capturing interactions between neurons:

$$\sum_{i \neq j} \lambda_{ij} \bigl( \phi(x_i) - \phi(x_j) - \alpha(x_i - x_j) \bigr) \bigl( \phi(x_i) - \phi(x_j) - \beta(x_i - x_j) \bigr) \le 0$$

* **Matrix Representation**: $T = \sum_{i < j} \lambda_{ij} (e_i - e_j)(e_i - e_j)^\top$ is used to structure the global QC.

---

### 5. Global QC for ReLU 
For the ReLU-type function $\phi(x) = \max(\alpha x, \beta x)$, the global QC matrix $Q$ is structured as:

$$
\begin{bmatrix} x \\ \phi(x) \\ 1 \end{bmatrix}^\top
\begin{bmatrix}
Q_{11} & Q_{12} & Q_{13} \\
Q_{12}^\top & Q_{22} & Q_{23} \\
Q_{13}^\top & Q_{23}^\top & Q_{33}
\end{bmatrix}
\begin{bmatrix} x \\ \phi(x) \\ 1 \end{bmatrix} \ge 0
$$

**Where the components are defined as:**

$$
\begin{aligned}
Q_{11} &= -2\alpha\beta (\text{diag}(\lambda) + T) \\
Q_{12} &= (\alpha + \beta)(\text{diag}(\lambda) + T) \\
Q_{13} &= -\beta\nu - \alpha\eta \\
Q_{22} &= -2(\text{diag}(\lambda) + T) \\
Q_{23} &= \nu + \eta \\
Q_{33} &= 0
\end{aligned}
$$

* **$\nu, \eta \ge 0$**: Non-negative weights representing active/inactive neuron states.
* **$T$**: A Z-matrix that accounts for cross-neuron interactions, reducing conservatism.
---

## 6. QC for Input Set $X$ 

If the input set $X$ is a hyperrectangle defined by $\underline{x} \le x \le \bar{x}$, it can be represented as a quadratic inequality using the matrix $P$:

$$
P = \begin{bmatrix}
-\Gamma & \Gamma \frac{(\bar{x} + \underline{x})}{2} \\
\left( \Gamma \frac{(\bar{x} + \underline{x})}{2} \right)^\top & -\underline{x}^\top \Gamma \bar{x}
\end{bmatrix}, \quad \Gamma \ge 0 \text{ (Diagonal)}
$$

* **Note**: This allows us to treat input bounds as just another QC, simplifying the optimization problem.