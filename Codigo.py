import math
import heapq
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog


# Funciones originales proporcionadas por el usuario

def calcular_contenido_informacion(p_i, base=2):
    """
    Calcula el contenido de información de una probabilidad p_i dada.

    :param p_i: Probabilidad p_i entre 0 y 1.
    :param base: Base del logaritmo para el cálculo.
    :return: Contenido de información I_i.
    """
    if not (0 < p_i <= 1):
        raise ValueError("La probabilidad p_i debe estar entre 0 y 1")
    I_i = -math.log(p_i, base)
    return I_i


def calcular_entropia(probabilidades):
    """
    Calcula la entropía de una fuente de información dada sus probabilidades.

    :param probabilidades: Diccionario de probabilidades de símbolos.
    :return: Valor de entropía.
    """
    entropia = sum(-p * math.log(p, 2) for p in probabilidades.values() if p > 0)
    return entropia


def construir_arbol_huffman(probabilidades):
    """
    Construye un árbol de codificación Huffman.

    :param probabilidades: Diccionario de probabilidades de símbolos.
    :return: Raíz del árbol de Huffman.
    """
    pq = [(prob, symbol) for symbol, prob in probabilidades.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        prob1, node1 = heapq.heappop(pq)
        prob2, node2 = heapq.heappop(pq)
        nodo_fusionado = (prob1 + prob2, [node1, node2])
        heapq.heappush(pq, nodo_fusionado)
    return pq[0][1]


def calcular_codigos_huffman(nodo, codigo="", mapeo=None):
    """
    Calcula los códigos Huffman para los símbolos.

    :param nodo: Nodo del árbol Huffman.
    :param codigo: Código parcial actual.
    :param mapeo: Diccionario para mapear símbolos a códigos.
    :return: Diccionario de códigos Huffman.
    """
    if mapeo is None:
        mapeo = {}
    if isinstance(nodo, str):
        mapeo[nodo] = codigo
    else:
        calcular_codigos_huffman(nodo[0], codigo + "0", mapeo)
        calcular_codigos_huffman(nodo[1], codigo + "1", mapeo)
    return mapeo


def longitud_promedio_codigos(codigos, probabilidades):
    """
    Calcula la longitud promedio de los códigos Huffman.

    :param codigos: Diccionario de códigos Huffman.
    :param probabilidades: Diccionario de probabilidades de símbolos.
    :return: Longitud promedio de los códigos.
    """
    longitud_promedio = sum(len(codigos[simbolo]) * prob for simbolo, prob in probabilidades.items())
    return longitud_promedio


def simular_error_transmision(mensaje, tasa_error):
    """
    Simula errores de transmisión en un mensaje.

    :param mensaje: Mensaje de entrada.
    :param tasa_error: Tasa de error de transmisión.
    :return: Mensaje con errores de transmisión.
    """
    mensaje_corrupto = ''
    for caracter in mensaje:
        representacion_binaria = format(ord(caracter), '08b')
        binario_corrupto = ''.join(
            '1' if bit == '0' else '0' if np.random.random() < tasa_error else bit
            for bit in representacion_binaria
        )
        caracter_corrupto = chr(int(binario_corrupto, 2))
        mensaje_corrupto += caracter_corrupto
    return mensaje_corrupto


# Nueva función para simular el canal con errores

def simular_canal_con_errores(probabilidades, tasa_error):
    """
    Simula el comportamiento de un canal con errores.

    :param probabilidades: Diccionario de probabilidades de símbolos.
    :param tasa_error: Tasa de error de transmisión.
    :return: Probabilidades condicionales p(y|x).
    """
    p_y_dado_x = {x: {y: 0 for y in probabilidades} for x in probabilidades}
    for x in probabilidades:
        for y in probabilidades:
            if x == y:
                p_y_dado_x[x][y] = 1 - tasa_error
            else:
                p_y_dado_x[x][y] = tasa_error / (len(probabilidades) - 1)
    return p_y_dado_x


# Integración de la información mutua y cálculo de la entropía condicional

def calcular_informacion_mutua(p_x, p_y_dado_x):
    """
    Calcula la información mutua entre X e Y.

    :param p_x: Probabilidades marginales p(x).
    :param p_y_dado_x: Probabilidades condicionales p(y|x).
    :return: Información mutua I(X; Y).
    """
    p_y = defaultdict(float)
    for x in p_x:
        for y in p_y_dado_x[x]:
            p_y[y] += p_x[x] * p_y_dado_x[x][y]
    I_XY = 0
    for x in p_x:
        for y in p_y_dado_x[x]:
            if p_y_dado_x[x][y] > 0 and p_y[y] > 0:
                I_XY += p_x[x] * p_y_dado_x[x][y] * math.log(p_y_dado_x[x][y] / p_y[y], 2)
    return I_XY


def calcular_entropia_condicional(p_x, p_y_dado_x):
    """
    Calcula la entropía condicional H(Y|X).

    :param p_x: Probabilidades marginales p(x).
    :param p_y_dado_x: Probabilidades condicionales p(y|x).
    :return: Entropía condicional H(Y|X).
    """
    H_Y_dado_X = 0
    for x in p_x:
        for y in p_y_dado_x[x]:
            if p_y_dado_x[x][y] > 0:
                H_Y_dado_X += p_x[x] * p_y_dado_x[x][y] * math.log(p_y_dado_x[x][y], 2)
    return -H_Y_dado_X


# Cálculo de la capacidad del canal

def capacidad_canal(p_y_dado_x):
    """
    Calcula la capacidad del canal.

    :param p_y_dado_x: Probabilidades condicionales p(y|x).
    :return: Capacidad del canal en bits por símbolo.
    """
    c = [-1 for _ in p_y_dado_x]
    A_eq = [[1] * len(p_y_dado_x)]
    b_eq = [1]
    A_ub = [[-1 if j == i else 0 for j in range(len(p_y_dado_x))] for i in range(len(p_y_dado_x))]
    b_ub = [0 for _ in p_y_dado_x]
    bounds = [(0, 1) for _ in p_y_dado_x]

    resultado = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if resultado.success:
        p_x_optimizado = resultado.x
        capacidad_canal_valor = -resultado.fun
        print(f"P(x) optimizado: {p_x_optimizado}")
        print(f"Capacidad del canal: {capacidad_canal_valor:.10f} bits por símbolo")
        return capacidad_canal_valor
    else:
        raise ValueError("La optimización falló. Mensaje de error: {}".format(resultado.message))


def main():
    probabilidades = {
        "a": 11.7868890,
        "b": 1.5914440,
        "c": 4.0138747,
        "d": 4.8795963,
        "e": 13.0540221,
        "f": 0.9484367,
        "g": 1.3089327,
        "h": 1.3611867,
        "i": 6.1462618,
        "j": 0.4708559,
        "k": 0.0967092,
        "l": 5.5396843,
        "m": 2.8490104,
        "n": 6.9935255,
        "o": 9.0869112,
        "p": 2.5317129,
        "q": 1.0239995,
        "r": 6.6389544,
        "s": 7.6876143,
        "t": 4.8818026,
        "u": 4.1347402,
        "v": 1.0878623,
        "w": 0.2114415,
        "x": 0.1612294,
        "y": 1.1529195,
        "z": 0.3603828
    }

    # Normalizar probabilidades para que sumen 1
    total_probabilidad = sum(probabilidades.values())
    probabilidades_normalizadas = {k: v / total_probabilidad for k, v in probabilidades.items()}

    # Calcular el contenido de información para cada letra
    contenido_informacion = {letra: calcular_contenido_informacion(prob) for letra, prob in
                             probabilidades_normalizadas.items()}

    # Calcular la entropía de la fuente
    entropia_fuente = calcular_entropia(probabilidades_normalizadas)

    # Construir el árbol de codificación Huffman
    arbol_huffman = construir_arbol_huffman(probabilidades_normalizadas)

    # Calcular los códigos Huffman
    codigos_huffman = calcular_codigos_huffman(arbol_huffman)

    # Imprimir el contenido de información para cada letra
    for letra, info_contenido in contenido_informacion.items():
        print(f"Letra '{letra}' tiene un contenido de información de: {info_contenido:.2f} bits")

    # Imprimir la entropía de la fuente
    print(f"\nEntropía de la fuente: {entropia_fuente:.2f} bits")

    # Imprimir los códigos Huffman
    print("\nCódigos Huffman:")
    for simbolo, codigo in codigos_huffman.items():
        print(f"Símbolo '{simbolo}': {codigo}")

    # Calcular la longitud promedio de los códigos
    longitud_promedio_codigos_huffman = longitud_promedio_codigos(codigos_huffman, probabilidades_normalizadas)
    print(f"Longitud promedio de los códigos: {longitud_promedio_codigos_huffman:.2f} bits")

    # Dado que los códigos Huffman son CUD por diseño
    print("Los códigos Huffman son unívocamente decodificables (CUD) por diseño.")

    # Calcular epsilon (ε) como la diferencia entre la longitud promedio de los códigos y la entropía
    epsilon = longitud_promedio_codigos_huffman - entropia_fuente
    print(f"Epsilon (ε): {epsilon:.2f} bits")

    # Tasa de error de bits para Ethernet Cat-8
    ber_cat8 = 1E-10

    # Simular el comportamiento del canal
    p_y_dado_x = simular_canal_con_errores(probabilidades_normalizadas, ber_cat8)

    # Calcular e imprimir la información mutua y la entropía condicional
    informacion_mutua = calcular_informacion_mutua(probabilidades_normalizadas, p_y_dado_x)
    entropia_condicional = calcular_entropia_condicional(probabilidades_normalizadas, p_y_dado_x)
    print(f"Información Mutua I(X; Y): {informacion_mutua:.10f} bits")
    print(f"Entropía Condicional H(Y|X): {entropia_condicional:.10f} bits")

    # Calcular e imprimir la capacidad del canal calculada
    capacidad = capacidad_canal(p_y_dado_x)
    print(f"Capacidad del Canal: {capacidad:.10f} bits por símbolo")


if __name__ == "__main__":
    main()
