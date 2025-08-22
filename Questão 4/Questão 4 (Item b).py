#Questão 4 (item b)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------
# Sistema pendulo simples com amortecimento
# ---------------------------
def pendulum_system(t, X, zeta, w_n):
    phi, v = X
    dphi_dt = v
    dv_dt = -zeta * v - w_n**2 * np.sin(phi)
    return [dphi_dt, dv_dt]

# ---------------------------
# Parâmetros
# ---------------------------
zeta = 0.1     # coeficiente de amortecimento
w_n = 1.0      # frequência natural

# ---------------------------
# Pontos fixos analíticos + classificação
# ---------------------------
fixed_points = [(-np.pi, 0.0), (0.0, 0.0), (np.pi, 0.0)]  # Corrigido: (-np.pi, 0.0) duplicado
fixed_labels = ["Instável", "Estável", "Instável"]

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
    phi, v = final_state
    label = (round(phi/tol), round(v/tol))
    return label

# ---------------------------
# Construção das bacias de atração
# ---------------------------
grid_points = 50
phi_range = np.linspace(-np.pi, np.pi, grid_points)
v_range = np.linspace(-3, 3, grid_points)

t_span = [0, 300]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

basin_phi_v = np.zeros((grid_points, grid_points))

for i, phi in enumerate(phi_range):
    for j, v in enumerate(v_range):
        y0 = [phi, v]
        _, sol = rk4(pendulum_system, t_span, y0, t_eval, args=(zeta, w_n))
        final_state = np.mean(sol[-200:], axis=0)
        label = classify_state(final_state)
        basin_phi_v[j, i] = hash(label) % 10

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(7,6))
cmap = plt.cm.Blues
colors = [cmap(0.3), cmap(0.6), cmap(0.9)]  # tons fixos de azul
custom_cmap = ListedColormap(colors[:len(fixed_points)])

plt.imshow(basin_phi_v, extent=[phi_range[0], phi_range[-1], v_range[0], v_range[-1]],
           origin='lower', cmap=custom_cmap, interpolation="nearest", aspect='auto')
plt.xlabel("φ (rad)")
plt.ylabel("dφ/dt")
plt.title("Bacia de Atração do Pêndulo Amortecido")

# Destacar pontos fixos
stable_plotted = False
unstable_plotted = False

for idx, (xf, xd) in enumerate(fixed_points):
    if fixed_labels[idx] == "Estável":
        if not stable_plotted:
            plt.plot(xf, xd, marker='o', color='red', markersize=8, label="Estável")
            stable_plotted = True
        else:
            plt.plot(xf, xd, marker='o', color='red', markersize=8)
    else:
        if not unstable_plotted:
            plt.plot(xf, xd, marker='s', color='red', markersize=8, label="Instável")
            unstable_plotted = True
        else:
            plt.plot(xf, xd, marker='s', color='red', markersize=8)

plt.legend()
plt.show()