#Questão 4 (item c)

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Sistema dinâmico
# ---------------------------
def multi_stable_system(t, X, W, G, alpha1, beta1, p, Omega, alpha2, beta2):
    x1, x1_dot, x2, x2_dot = X

    dx1dt = x1_dot
    dx1_dotdt = (-W * x1_dot
                + 2 * G * (x2_dot - x1_dot)
                - (1 + alpha1) * x1
                - beta1 * x1**3
                + p * Omega**2 * (x2 - x1))

    dx2dt = x2_dot
    dx2_dotdt = (-2 * G * (x2_dot - x1_dot)
                 - alpha2 * x2
                 - beta2 * x2**3
                 - p * Omega**2 * (x2 - x1))

    return [dx1dt, dx1_dotdt, dx2dt, dx2_dotdt]

# ---------------------------
# Parâmetros
# ---------------------------
W = 0.025
G = 0.025
alpha1 = -2.0
beta1 = 1.0
p = 0.5
Omega = 1.0
alpha2 = -1.0
beta2 = 1.0

# ---------------------------
# Pontos fixos analíticos + classificação
# ---------------------------
fixed_points = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]  # Corrigido: (-np.pi, 0.0) duplicado
fixed_labels = ["Estável", "Instável", "Estável"]

# ---------------------------
# Integrador RK4
# ---------------------------
def rk4(func, t_span, y0, t_eval, args=()):
    n = len(t_eval)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    h = t_eval[1] - t_eval[0]

    for i in range(n - 1):
        t = t_eval[i]
        k1 = np.array(func(t, y[i], *args))
        k2 = np.array(func(t + h/2, y[i] + h/2 * k1, *args))
        k3 = np.array(func(t + h/2, y[i] + h/2 * k2, *args))
        k4 = np.array(func(t + h, y[i] + h * k3, *args))
        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_eval, y

# ---------------------------
# Função para classificar atrator final
# ---------------------------
def classify_state(final_state, tol=0.3):
    x1, x1_dot, x2, x2_dot = final_state
    label_x1x2  = (round(x1/tol), round(x2/tol))
    label_x1dx1 = (round(x1/tol), round(x1_dot/tol))
    label_x2dx2 = (round(x2/tol), round(x2_dot/tol))
    return label_x1x2, label_x1dx1, label_x2dx2

# ---------------------------
# Construção das bacias de atração
# ---------------------------
grid_points = 50
x_range = np.linspace(-2, 2, grid_points)
y_range = np.linspace(-2, 2, grid_points)

t_span = [0, 300]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

basin_x1x2  = np.zeros((grid_points, grid_points))
basin_x1dx1 = np.zeros((grid_points, grid_points))
basin_x2dx2 = np.zeros((grid_points, grid_points))

# ---------------------------
# Loop 1: x1 vs x2
# ---------------------------
for i, x1 in enumerate(x_range):
    for j, x2 in enumerate(y_range):
        y0 = [x1, 0.0, x2, 0.0]
        _, sol = rk4(multi_stable_system, t_span, y0, t_eval,
                     args=(W, G, alpha1, beta1, p, Omega, alpha2, beta2))
        final_state = np.mean(sol[-200:], axis=0)
        label_x1x2, _, _ = classify_state(final_state)
        basin_x1x2[j, i] = hash(label_x1x2) % 10

# ---------------------------
# Loop 2: x1 vs dx1
# ---------------------------
for i, x1 in enumerate(x_range):
    for j, dx1 in enumerate(y_range):
        y0 = [x1, dx1, 0.0, 0.0]
        _, sol = rk4(multi_stable_system, t_span, y0, t_eval,
                     args=(W, G, alpha1, beta1, p, Omega, alpha2, beta2))
        final_state = np.mean(sol[-200:], axis=0)
        _, label_x1dx1, _ = classify_state(final_state)
        basin_x1dx1[j, i] = hash(label_x1dx1) % 10

# ---------------------------
# Loop 3: x2 vs dx2
# ---------------------------
for i, x2 in enumerate(x_range):
    for j, dx2 in enumerate(y_range):
        y0 = [0.0, 0.0, x2, dx2]
        _, sol = rk4(multi_stable_system, t_span, y0, t_eval,
                     args=(W, G, alpha1, beta1, p, Omega, alpha2, beta2))
        final_state = np.mean(sol[-200:], axis=0)
        _, _, label_x2dx2 = classify_state(final_state)
        basin_x2dx2[j, i] = hash(label_x2dx2) % 10

# ---------------------------
# Plots
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# x1 vs x2
axes[0].imshow(basin_x1x2, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
               origin='lower', cmap='tab10', interpolation="nearest")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].set_title("Bacia de Atração (x1 vs x2)")

# Destacar pontos fixos
stable_plotted = False
unstable_plotted = False

for idx, (xf, xd) in enumerate(fixed_points):
    if fixed_labels[idx] == "Estável":
        if not stable_plotted:
            axes[0].plot(xf, xd, marker='o', color='red', markersize=8, label="Estável")
            stable_plotted = True
        else:
            axes[0].plot(xf, xd, marker='o', color='red', markersize=8)
    else:
        if not unstable_plotted:
            axes[0].plot(xf, xd, marker='s', color='red', markersize=8, label="Instável")
            unstable_plotted = True
        else:
            axes[0].plot(xf, xd, marker='s', color='red', markersize=8)

# x1 vs dx1
axes[1].imshow(basin_x1dx1, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
               origin='lower', cmap='tab10', interpolation="nearest")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("dx1/dt")
axes[1].set_title("Bacia de Atração (x1 vs dx1/dt)")

# x2 vs dx2
axes[2].imshow(basin_x2dx2, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
               origin='lower', cmap='tab10', interpolation="nearest")
axes[2].set_xlabel("x2")
axes[2].set_ylabel("dx2/dt")
axes[2].set_title("Bacia de Atração (x2 vs dx2/dt)")

plt.legend()
plt.show()