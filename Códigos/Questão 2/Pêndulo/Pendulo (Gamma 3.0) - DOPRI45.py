#Questão 2 (Pêndulo Simples)

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Método de integração DOPRI45
# ==============================
def dopri45_step(f, t, y, h):
    """
    Um passo do método Dormand-Prince 4(5).
    Retorna (y_next, erro_estimado).
    """
    # Coeficientes Butcher (Dormand-Prince)
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = (35/384, 0, 500/1113, 125/192, -2187/6784, 11/84)

    # Coeficientes para solução de ordem 5
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    # Coeficientes para solução de ordem 4 (para erro)
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640,
                       -92097/339200, 187/2100, 1/40])

    # Estágios
    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*a21*k1)
    k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
    k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
    k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
    k7 = f(t + h,    y + h*(a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6))

    ks = np.array([k1, k2, k3, k4, k5, k6, k7])

    # Ordem 5
    y_next = y + h * np.dot(b, ks)
    # Ordem 4
    y_next_4 = y + h * np.dot(b_star, ks)

    # Estimativa de erro
    err = np.linalg.norm(y_next - y_next_4, ord=np.inf)

    return y_next, err

def dopri45(f, y0, t_span, h_init=0.01, tol=1e-6):
    """Integrador DOPRI45 adaptativo."""
    t0, tf = t_span
    t = t0
    y = y0.copy()
    h = h_init

    T = [t]
    Y = [y]

    while t < tf:
        if t + h > tf:
            h = tf - t

        y_next, err = dopri45_step(f, t, y, h)

        # Controle adaptativo de passo
        if err < tol:
            t += h
            y = y_next
            T.append(t)
            Y.append(y)

        # Ajuste do passo
        if err == 0:
            s = 2
        else:
            s = 0.9 * (tol / err) ** 0.25
        h = h * min(max(s, 0.2), 5.0)

    return np.array(T), np.array(Y).T

# ==============================
# Sistema: Pêndulo Forçado
# ==============================
def sistema_pendulo(t, y, omega_n, Z, gamma, Omega):
    """Pêndulo simples amortecido e forçado."""
    u, v = y
    du_dt = v
    dv_dt = -Z * v - omega_n**2 * np.sin(u) + gamma * np.sin(Omega * t)
    return np.array([du_dt, dv_dt])

# ==============================
# Mapa de Poincaré
# ==============================
def mapa_de_poincare(t, y, Omega, n_transientes=5):
    """Calcula os pontos do mapa de Poincaré com interpolação linear."""
    T = 2 * np.pi / Omega
    poincare_points = []

    # momentos exatos após descartar transientes
    t_poincare = np.arange(n_transientes * T, t[-1], T)

    for tp in t_poincare:
        # encontra intervalo onde t[i] <= tp < t[i+1]
        idx = np.searchsorted(t, tp) - 1
        if idx < 0 or idx >= len(t) - 1:
            continue

        # interpolação linear
        t1, t2 = t[idx], t[idx + 1]
        y1, y2 = y[:, idx], y[:, idx + 1]
        alpha = (tp - t1) / (t2 - t1)
        y_interp = y1 + alpha * (y2 - y1)

        poincare_points.append(y_interp)

    return np.array(poincare_points).T if poincare_points else np.empty((2, 0))

# ==============================
# Parâmetros
# ==============================
omega_n = 0.5   # frequência natural (sqrt(g/L))
Z = 0.5         # amortecimento
gamma = 3.0     # amplitude da força externa
Omega_values = [0.5, 1.0, 1.5]

# Condições iniciais
u0 = 0.0   # posição inicial
v0 = 0.0   # velocidade inicial
y0 = np.array([u0, v0])

# Tempo de simulação
t_final = 200.0

# ==============================
# Simulação e plots
# ==============================
for Omega in Omega_values:
    f = lambda t, y: sistema_pendulo(t, y, omega_n, Z, gamma, Omega)

    t_span = (0.0, t_final)
    t, Y = dopri45(f, y0, t_span, h_init=0.01, tol=1e-6)

    # Criar figura
    plt.figure(figsize=(10, 8))

    # Resposta temporal
    plt.subplot(2, 1, 1)
    plt.plot(t, Y[0, :])
    plt.title(f'Resposta da Posição ao Longo do Tempo (Pêndulo, Ω = {Omega})')
    plt.xlabel('Tempo (s)')
    plt.ylabel('u (rad)')
    plt.grid(True)

    # Espaço de fase e Poincaré
    plt.subplot(2, 1, 2)
    plt.plot(Y[0, :], Y[1, :], 'b', label='Espaço de Fase')
    poincare_points = mapa_de_poincare(t, Y, Omega)

    if poincare_points.size > 0:
        plt.scatter(poincare_points[0, :], poincare_points[1, :],
                    color='red', marker='o', label='Mapa de Poincaré')

    plt.title(f'Espaço de Fase e Mapa de Poincaré (Pêndulo, Ω = {Omega})')
    plt.xlabel('u (rad)')
    plt.ylabel('v (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

plt.show()