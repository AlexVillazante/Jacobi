import numpy as np

def gauss_seidel(A, b, x_init, tol=1e-4, max_iterations=100):
    n = len(b)
    x = x_init.copy()
    errores = []

    for k in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]

        error = np.max(np.abs(x - x_old))
        errores.append(error)

        # Formatear con 4 decimales
        x_formatted = [f'{val:.4f}' for val in x]
        error_formatted = f'{error:.4f}'

        print(f"Iteración {k + 1}: x = {x_formatted}, error = {error_formatted}")

        if error < tol:
            print("Convergencia alcanzada")
            break

    print("\nSolución final:")
    print([f'{val:.4f}' for val in x])
    print(f"Número de iteraciones: {k + 1}")
    print(f"Errores por iteración: {[f'{e:.4f}' for e in errores]}")

# Definir la matriz A y el vector b del sistema
A = np.array([[3, -0.1, -0.2], 
              [0.1, 7, -0.3], 
              [0.3, -0.2, 10]], dtype=float)

b = np.array([7.85, -19.3, 71.4], dtype=float)

# Vector inicial
x_init = np.array([0, 0, 0], dtype=float)

# Ejecutar el método de Gauss-Seidel
gauss_seidel(A, b, x_init)
