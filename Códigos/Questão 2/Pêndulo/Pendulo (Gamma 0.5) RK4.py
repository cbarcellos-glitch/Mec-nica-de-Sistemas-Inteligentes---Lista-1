#Questão 2 (Pêndulo simples)
import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, y, t, h):
    """Realiza um passo do método Runge-Kutta de quarta ordem."""
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def sistema_pendulo(t, y, omega_n, Z, gamma, Omega):
    """Define o sistema de equações diferenciais."""
    u, v = y
    du_dt = v
    dv_dt = -Z * v - omega_n**2 * np.sin(u) + gamma * np.sin(Omega * t)
    return np.array([du_dt, dv_dt])

def mapa_de_poincare(t, y, Omega, n_transientes=50):
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

# Parâmetros do sistema
omega_n = 1.0   # frequência natural (sqrt(g/L))
Z = 0.5         # amortecimento
gamma = 0.5     # amplitude da força externa
Omega_values = [0.5, 1.0, 1.5]  # frequências de excitação para análise

# Condições iniciais
u0 = 0.0   # deslocamento inicial (rad)
v0 = 0.0   # velocidade inicial
y0 = np.array([u0, v0])

# Tempo de simulação e passo
t_final = 200.0
h = 0.01  # passo RK4
t_span = np.arange(0, t_final + h, h)

# Resolver e plotar os resultados
for idx, Omega in enumerate(Omega_values):
    # Inicializar variáveis
    y = np.zeros((2, len(t_span)))
    y[:, 0] = y0

    # Método de Runge-Kutta de quarta ordem
    for i in range(len(t_span) - 1):
        y[:, i + 1] = rk4_step(lambda t, y: sistema_pendulo(t, y, omega_n, Z, gamma, Omega),
                               y[:, i], t_span[i], h)

    # Criar nova figura para cada Omega
    plt.figure(figsize=(10, 8))

    # Resposta da posição ao longo do tempo
    plt.subplot(2, 1, 1)
    plt.plot(t_span, y[0, :])
    plt.title(f'Resposta da Posição ao Longo do Tempo (Ω = {Omega})')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (u em rad)')
    plt.grid(True)

    # Espaço de fase e Mapa de Poincaré
    plt.subplot(2, 1, 2)
    plt.plot(y[0, :], y[1, :], 'b', label='Espaço de Fase')
    poincare_points = mapa_de_poincare(t_span, y, Omega)

    if poincare_points.size > 0:
        plt.scatter(poincare_points[0, :], poincare_points[1, :],
                    color='red', marker='o', label='Mapa de Poincaré')

    plt.title(f'Espaço de Fase e Mapa de Poincaré (Ω = {Omega})')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

plt.show()