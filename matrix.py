import math
from collections import defaultdict
import numpy as np
from itertools import count

# Funciones

def calcular_contenido_informacion(p_i, base=2):
    """
    Calcula el contenido de información de una probabilidad p_i dada.
    """
    if not (0 < p_i <= 1):
        raise ValueError("La probabilidad p_i debe estar entre 0 y 1")
    I_i = -math.log(p_i, base)
    return I_i

def calcular_entropia(probabilidades):
    """
    Calcula la entropía de una fuente de información dada sus probabilidades.
    """
    entropia = sum(-p * math.log(p, 2) for p in probabilidades.values() if p > 0)
    return entropia

def calcular_probabilidades_salida(p_x, p_y_dado_x):
    """
    Calcula las probabilidades de los símbolos de salida p(y_j) dadas p(x_i) y P(y_j | x_i).
    """
    p_y = {}
    for y in next(iter(p_y_dado_x.values())):
        p_y[y] = sum(p_x[x] * p_y_dado_x[x][y] for x in p_x)
    return p_y

def calcular_probabilidad_error(p_x, p_y_dado_x, expected_y):
    """
    Calcula la probabilidad de error de símbolo del canal.
    """
    P_e = 0
    for x in p_x:
        P_e_x = 1 - p_y_dado_x[x][expected_y[x]]
        P_e += p_x[x] * P_e_x
    return P_e

def calcular_entropia_condicional(p_x, p_y_dado_x):
    """
    Calcula la entropía condicional H(Y|X).
    """
    H_Y_dado_X = 0
    for x in p_x:
        H_Y_given_x = -sum(p_y_dado_x[x][y] * math.log(p_y_dado_x[x][y], 2) for y in p_y_dado_x[x] if p_y_dado_x[x][y] > 0)
        H_Y_dado_X += p_x[x] * H_Y_given_x
    return H_Y_dado_X

def calcular_entropia_salida(p_y):
    """
    Calcula la entropía de los símbolos de salida H(Y).
    """
    H_Y = -sum(p * math.log(p, 2) for p in p_y.values() if p > 0)
    return H_Y

def calcular_informacion_mutua(p_x, p_y_dado_x, p_y):
    """
    Calcula la información mutua I(X;Y).
    """
    I_XY = 0
    for x in p_x:
        for y in p_y_dado_x[x]:
            if p_y_dado_x[x][y] > 0 and p_y[y] > 0:
                I_XY += p_x[x] * p_y_dado_x[x][y] * math.log(p_y_dado_x[x][y] / p_y[y], 2)
    return I_XY

def main():
    # Probabilidades de entrada p(x_i)
    p_x = {
        'x1': 0.35,
        'x2': 0.15,
        'x3': 0.50
    }

    # Matriz de transición P(y_j | x_i)
    p_y_dado_x = {
        'x1': {'y1': 0.72, 'y2': 0.04, 'y3': 0.15, 'y4': 0.09},
        'x2': {'y1': 0.12, 'y2': 0.75, 'y3': 0.00, 'y4': 0.13},
        'x3': {'y1': 0.07, 'y2': 0.08, 'y3': 0.82, 'y4': 0.03}
    }

    # Símbolos de salida esperados para cada símbolo de entrada
    expected_y = {
        'x1': 'y1',
        'x2': 'y2',
        'x3': 'y3'
    }

    # Cálculo de probabilidades de salida
    p_y = calcular_probabilidades_salida(p_x, p_y_dado_x)

    # Cálculo de la probabilidad de error
    P_e = calcular_probabilidad_error(p_x, p_y_dado_x, expected_y)

    # Cálculo de la entropía condicional H(Y|X)
    H_Y_given_X = calcular_entropia_condicional(p_x, p_y_dado_x)

    # Cálculo de la entropía de salida H(Y)
    H_Y = calcular_entropia_salida(p_y)

    # Cálculo de la información mutua I(X;Y)
    I_XY = H_Y - H_Y_given_X

    # Imprimir resultados
    print_results(p_x, p_y_dado_x, p_y, P_e, H_Y, H_Y_given_X, I_XY)

def print_results(p_x, p_y_dado_x, p_y, P_e, H_Y, H_Y_given_X, I_XY):
    # Imprimir probabilidades de entrada
    print("Probabilidades de entrada p(x_i):")
    for x in p_x:
        print(f"p({x}) = {p_x[x]:.4f}")
    print()

    # Imprimir matriz de transición
    print("Matriz de transición P(y_j | x_i):")
    print("          ", end='')
    for y in next(iter(p_y_dado_x.values())):
        print(f"{y:>6}", end='')
    print()
    for x in p_y_dado_x:
        print(f"{x:>10}", end='')
        for y in p_y_dado_x[x]:
            print(f"{p_y_dado_x[x][y]:6.2f}", end='')
        print()
    print()

    # Imprimir probabilidades de salida
    print("Probabilidades de salida p(y_j):")
    for y in p_y:
        print(f"p({y}) = {p_y[y]:.4f}")
    print()

    # Imprimir probabilidad de error
    print(f"Probabilidad de error de símbolo del canal: {P_e:.4f}")

    # Imprimir entropías e información mutua
    print(f"Entropía de la salida H(Y): {H_Y:.4f} bits")
    print(f"Entropía condicional H(Y|X): {H_Y_given_X:.4f} bits")
    print(f"Información mutua I(X;Y): {I_XY:.4f} bits")

if __name__ == "__main__":
    main()
