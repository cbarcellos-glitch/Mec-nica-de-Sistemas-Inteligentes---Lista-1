#Questão 4 (item a)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------
# Sistema de Duffing
# ---------------------------
def duffing_system(t, X, zeta, alpha, beta):
    x, x_dot = X
    dxdt = x_dot
    dx_dotdt = -2*zeta*x_dot - alpha*x - beta*x**3
    return [dxdt, dx_dotdt]

# ---------------------------
# Parâmetros do sistema
# ---------------------------
zeta = 0.1     # amortecimento
alpha = -1.0   # parâmetro linear
beta = 1.0     # parâmetro não linear

# ---------------------------
# Pontos fixos analíticos + classificação
# ---------------------------
fixed_points = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]
fixed_labels = ["Instável", "Estável", "Estável"]

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
# Classificação do atrator
# ---------------------------
def classify_state(final_state, fixed_points, tol=0.3):
    x, x_dot = final_state
    for idx, (xf, xd) in enumerate(fixed_points):
        if np.sqrt((x-xf)**2 + (x_dot-xd)**2) < tol:
            return idx
    return -1  # sem convergência clara

# ---------------------------
# Construção da bacia de atração
# ---------------------------
grid_points = 50
x_range = np.linspace(-2, 2, grid_points)
dx_range = np.linspace(-2, 2, grid_points)

t_span = [0, 300]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

basin = np.full((grid_points, grid_points), -1)

for i, x0 in enumerate(x_range):
    for j, dx0 in enumerate(dx_range):
        y0 = [x0, dx0]
        _, sol = rk4(duffing_system, t_span, y0, t_eval,
                     args=(zeta, alpha, beta))
        final_state = np.mean(sol[-200:], axis=0)  # estado médio final
        label = classify_state(final_state, fixed_points, tol=0.2)
        basin[j, i] = label

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(7,6))
cmap = plt.cm.Blues
colors = [cmap(0.3), cmap(0.6), cmap(0.9)]  # tons fixos de azul
custom_cmap = ListedColormap(colors[:len(fixed_points)])

plt.imshow(basin, extent=[x_range[0], x_range[-1], dx_range[0], dx_range[-1]],
           origin='lower', cmap=custom_cmap, interpolation="nearest")

plt.xlabel("x")
plt.ylabel("dx/dt")
plt.title("Bacia de Atração do Oscilador de Duffing")

# Destacar pontos fixos com símbolos diferentes
for idx, (xf, xd) in enumerate(fixed_points):
    if fixed_labels[idx] == "Estável":
        plt.plot(xf, xd, marker='o', color='red', markersize=8, label="Estável")
    else:
        plt.plot(xf, xd, marker='s', color='red', markersize=8, label="Instável")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()