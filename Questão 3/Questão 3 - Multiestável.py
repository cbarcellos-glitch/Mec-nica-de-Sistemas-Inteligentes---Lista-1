import numpy as np

def numerical_jacobian(f, x, eps=1e-6):
    """
    Jacobiano numérico por diferenças progressivas.
    f: função R^n -> R^n, recebe x (array) e retorna array de mesma dimensão.
    x: ponto onde avaliar o jacobiano.
    eps: passo de diferenciação.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)
    fx = np.asarray(f(x), dtype=float)
    for j in range(n):
        x_step = x.copy()
        x_step[j] += eps
        fx_step = np.asarray(f(x_step), dtype=float)
        J[:, j] = (fx_step - fx) / eps
    return J

def newton_solve(f, x0, jac=None, tol=1e-10, max_iter=100, line_search=True):
    """
    Newton-Raphson multivariado para f(x)=0.
    - f: R^n -> R^n
    - x0: chute inicial (array-like)
    - jac: função que retorna J(x). Se None, usa numerical_jacobian.
    Retorna (x, convergiu, iterações).
    """
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, max_iter+1):
        fx = np.asarray(f(x), dtype=float)
        norm_fx = np.linalg.norm(fx, ord=2)
        if norm_fx < tol:
            return x, True, k
        J = numerical_jacobian(f, x) if jac is None else np.asarray(jac(x), dtype=float)
        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(J) @ fx
        if line_search:
            alpha = 1.0
            x_new = x + alpha*dx
            fx_new = np.asarray(f(x_new), dtype=float)
            while np.linalg.norm(fx_new) > (1 - 1e-4*alpha)*norm_fx and alpha > 1e-4:
                alpha *= 0.5
                x_new = x + alpha*dx
                fx_new = np.asarray(f(x_new), dtype=float)
            x = x_new
        else:
            x = x + dx
    return x, False, max_iter

def dedup_points(points, tol=1e-8):
    """
    Remove pontos repetidos/duplicados (clustering por distância).
    """
    unique = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not any(np.linalg.norm(p - np.asarray(q, dtype=float)) <= tol for q in unique):
            unique.append(p)
    return unique

def classify_2d(evals, tol_real=1e-9, tol_imag=1e-9):
    """
    Classificação 2D baseada nos autovalores.
    """
    lam = np.asarray(evals)
    re = np.real(lam).copy()
    im = np.imag(lam).copy()   # <-- copy() aqui!

    re[np.abs(re) < tol_real] = 0.0
    im[np.abs(im) < tol_imag] = 0.0

    idx = np.argsort(re + 1j*im)
    re = re[idx]; im = im[idx]

    purely_real = np.all(im == 0.0)
    if purely_real:
        if np.all(re < 0.0):
            return "poço estável", {"tipo": "no real estável", "autovalores": lam}
        if np.all(re > 0.0):
            return "fonte instável", {"tipo": "no real instável", "autovalores": lam}
        if np.any(re > 0.0) and np.any(re < 0.0):
            return "sela instável", {"tipo": "sela (autovalores reais de sinais opostos)", "autovalores": lam}
        if np.all(re == 0.0):
            return "indeterminado (linearmente neutro)", {"motivo": "autovalores reais nulos", "autovalores": lam}
        return "indeterminado", {"motivo": "caso degenerado com zeros", "autovalores": lam}

    conj_like = np.isclose(re[0], re[1], atol=tol_real) and np.isclose(im[0], -im[1], atol=tol_imag)
    if conj_like:
        if np.all(re < 0.0):
            return "espiral estável", {"tipo": "foco espiral estável (parte real negativa)", "autovalores": lam}
        if np.all(re > 0.0):
            return "foco instável", {"tipo": "foco espiral instável (parte real positiva)", "autovalores": lam}
        if np.all(re == 0.0):
            return "centro estável", {"tipo": "centro (parte real zero, imaginário ±)", "autovalores": lam}
        return "indeterminado", {"motivo": "caso limítrofe (parte real ~0)", "autovalores": lam}

    if np.all(re < 0.0):
        return "espiral estável (numérica)", {"autovalores": lam}
    if np.all(re > 0.0):
        return "foco instável (numérica)", {"autovalores": lam}
    if np.any(re > 0.0) and np.any(re < 0.0):
        return "sela instável (numérica)", {"autovalores": lam}
    return "indeterminado", {"autovalores": lam}

def analyze_equilibria(
    f,
    guesses,
    jac=None,
    newton_tol=1e-10,
    newton_max_iter=100,
    dedup_tol=1e-8,
    jac_eps=1e-6,
    eig_tol_real=1e-9,
    eig_tol_imag=1e-9,
):
    """
    Pipeline principal para análise de equilíbrios.
    """
    candidatos = []
    for x0 in guesses:
        root, ok, it = newton_solve(f, x0, jac=jac, tol=newton_tol, max_iter=newton_max_iter)
        if ok and np.all(np.isfinite(root)):
            candidatos.append(np.asarray(root, dtype=float))
    equil = dedup_points(candidatos, tol=dedup_tol)

    resultados = []
    for p in equil:
        Jp = numerical_jacobian(f, p, eps=jac_eps) if jac is None else np.asarray(jac(p), dtype=float)
        evals = np.linalg.eigvals(Jp)
        if len(p) == 2:
            label, detalhe = classify_2d(evals, tol_real=eig_tol_real, tol_imag=eig_tol_imag)
        else:
            re = np.real(evals)
            if np.all(re < -eig_tol_real):
                label = "linearmente estável (todas as partes reais negativas)"
            elif np.all(re > eig_tol_real):
                label = "linearmente instável (todas as partes reais positivas)"
            elif np.any(re > eig_tol_real) and np.any(re < -eig_tol_real):
                label = "sela (mista)"
            else:
                label = "indeterminado/limítrofe (autovalores próximos de zero)"
            detalhe = {"autovalores": evals}

        resultados.append({
            "equilibrio": p,
            "jacobiano": Jp,
            "autovalores": evals,
            "classificacao": label,
            "detalhe": detalhe
        })
    return resultados

if __name__ == "__main__":
    # ----------------- Definição do sistema ----------------- #
    def f_lin(x):
        x1, x2, x3, x4 = x

        # parâmetros
        zeta1 = 0.025
        zeta2 = 0.025
        alpha1 = -2.0
        alpha2 = -1.0
        beta1  = 1.0
        beta2  = 1.0
        rho    = 0.5
        Omega_s = 1.0

        dx1 = x3
        dx2 = x4
        dx3 = -2*zeta1*(x3 + zeta2*(x4 - x3)) - (1+alpha1)*x1 - beta1*x1**3 + rho*Omega_s**2*(x2 - x1)
        dx4 = -(2*zeta2/rho)*(x4 - x3) - (alpha2/rho)*x2 - (beta2/rho)*x2**3 - Omega_s**2*(x2 - x1)

        return np.array([dx1, dx2, dx3, dx4], dtype=float)

    # ----------------- Chutes iniciais para equilíbrio ----------------- #
    guesses = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0],
        [-1.0, -0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0]
    ]
    resultados = analyze_equilibria(f_lin, guesses)
    for r in resultados:
        print("Equilíbrio:", r["equilibrio"])
        print("Jacobiano:\n", r["jacobiano"])
        print("Autovalores:", r["autovalores"])
        print("Classificação:", r["classificacao"])
        print("Detalhe:", r["detalhe"])
        print("-"*40)