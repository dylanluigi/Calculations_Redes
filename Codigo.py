import math
import heapq
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog


# Funciones

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
        print("P(x) optimizado:")
        for prob in p_x_optimizado:
            print(f"{prob:.10f}")  # Display each probability with 10 decimal places
        print(f"Capacidad del canal: {capacidad_canal_valor:.10f} bits por símbolo")
        return capacidad_canal_valor
    else:
        raise ValueError("La optimización falló. Mensaje de error: {}".format(resultado.message))


def calcular_tasa_datos_nyquist(banda, niveles):
    """
    Calcula la tasa máxima de datos según el criterio de Nyquist.

    :param banda: Ancho de banda del canal (en Hz).
    :param niveles: Número de niveles de señal discretos.
    :return: Tasa máxima de datos (en bits por segundo).
    """
    return 2 * banda


def calcular_capacidad_shannon(banda, snr):
    """
    Calcula la capacidad máxima del canal según la fórmula de Shannon.

    :param banda: Ancho de banda del canal (en Hz).
    :param snr: Relación señal-ruido (SNR).
    :return: Capacidad del canal (en bits por segundo).
    """
    return banda * math.log2(1 + snr)


def Q_function(x):
    """
    Approximate the Q-function, which is often used in BER calculations.

    :param x: Input to the Q-function.
    :return: Approximated Q-function result.
    """
    return 0.5 - 0.5 * math.erf(x / (2 ** 0.5))


def calculate_received_signal_power(transmitted_power_dbm, attenuation_db, distance_km):
    """
    Calculate the received signal power.

    :param transmitted_power_dbm: Transmitted power in dBm.
    :param attenuation_db: Attenuation coefficient in dB/km.
    :param distance_km: Distance in kilometers.
    :return: Received power in dBm.
    """
    # Calculate total attenuation
    total_attenuation_db = attenuation_db * distance_km

    # Calculate received power
    received_power_dbm = transmitted_power_dbm - total_attenuation_db

    return received_power_dbm


def calculate_noise_power(thermal_noise_dbm, noise_figure_db):
    noise_power_dbm = thermal_noise_dbm + noise_figure_db
    return noise_power_dbm


def calculate_snr(received_power_dbm, noise_power_dbm):
    snr_db = received_power_dbm - noise_power_dbm
    return snr_db


def calculate_spectral_efficiency(data_rate_bps, bandwidth_hz):
    return data_rate_bps / bandwidth_hz


def calculate_ber(snr_linear, modulation_type):
    if modulation_type == 'PSK':
        return Q_function(np.sqrt(2 * snr_linear))
    else:
        # Placeholder for other modulation types
        return None


def verify_shannons_formula(capacity_bps, bandwidth_hz, snr_linear):
    return capacity_bps < bandwidth_hz * math.log2(1 + snr_linear)


def determine_channel_bandwidth(symbol_rate_bps, rolloff_factor):
    return symbol_rate_bps * (1 + rolloff_factor)


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

    total_probabilidad = sum(probabilidades.values())
    probabilidades_normalizadas = {k: v / total_probabilidad for k, v in probabilidades.items()}

    # Huffman coding calculations
    contenido_informacion = {letra: calcular_contenido_informacion(prob) for letra, prob in
                             probabilidades_normalizadas.items()}
    entropia_fuente = calcular_entropia(probabilidades_normalizadas)
    arbol_huffman = construir_arbol_huffman(probabilidades_normalizadas)
    codigos_huffman = calcular_codigos_huffman(arbol_huffman)
    longitud_promedio_codigos_huffman = longitud_promedio_codigos(codigos_huffman, probabilidades_normalizadas)
    epsilon = longitud_promedio_codigos_huffman - entropia_fuente

    # Channel parameters and calculations
    transmitted_power_dbm = 3
    attenuation_db = 0.35
    thermal_noise_dbm = -174
    noise_figure_db = 4
    data_rate_bps = 1e9
    bandwidth_hz = 5e8
    rolloff_factor = 0.25

    received_power_dbm = calculate_received_signal_power(transmitted_power_dbm, attenuation_db, 1)
    noise_power_dbm = calculate_noise_power(thermal_noise_dbm, noise_figure_db)
    snr_db = calculate_snr(received_power_dbm, noise_power_dbm)
    snr_linear = 10 ** (snr_db / 10)
    spectral_efficiency = calculate_spectral_efficiency(data_rate_bps, bandwidth_hz)
    ber = calculate_ber(snr_linear, 'PSK')
    channel_bandwidth = determine_channel_bandwidth(data_rate_bps, rolloff_factor)
    shannons_compliance = verify_shannons_formula(data_rate_bps, bandwidth_hz, snr_linear)

    p_y_dado_x = simular_canal_con_errores(probabilidades_normalizadas, ber)
    informacion_mutua = calcular_informacion_mutua(probabilidades_normalizadas, p_y_dado_x)
    entropia_condicional = calcular_entropia_condicional(probabilidades_normalizadas, p_y_dado_x)
    capacidad = capacidad_canal(p_y_dado_x)

    # Print results
    print_results(contenido_informacion, entropia_fuente, codigos_huffman, longitud_promedio_codigos_huffman, epsilon,
                  received_power_dbm, noise_power_dbm, snr_db, spectral_efficiency, ber, channel_bandwidth,
                  shannons_compliance, informacion_mutua, entropia_condicional, capacidad)


def print_results(contenido_informacion, entropia_fuente, codigos_huffman, longitud_promedio_codigos_huffman, epsilon,
                  received_power_dbm, noise_power_dbm, snr_db, spectral_efficiency, ber, channel_bandwidth,
                  shannons_compliance, informacion_mutua, entropia_condicional, capacidad):
    # Print Huffman coding results
    print("Contenido de información:")
    for letra, info in contenido_informacion.items():
        print(f"Letra '{letra}': {info:.2f} bits")
    print(f"\nEntropía de la fuente: {entropia_fuente:.2f} bits")
    print("\nCódigos Huffman:")
    for simbolo, codigo in codigos_huffman.items():
        print(f"Símbolo '{simbolo}': {codigo}")
    print(f"Longitud promedio de los códigos Huffman: {longitud_promedio_codigos_huffman:.2f} bits")
    print(f"Epsilon (ε): {epsilon:.2f} bits")

    # Print channel analysis results
    print(f"\nReceived Power (dBm): {received_power_dbm}")
    print(f"Noise Power (dBm): {noise_power_dbm}")
    print(f"SNR (dB): {snr_db}")
    print(f"Spectral Efficiency (bps/Hz): {spectral_efficiency}")
    print(f"BER: {ber:.15f}")
    print(f"Channel Bandwidth (Hz): {channel_bandwidth}")
    print(f"Shannon's Compliance: {'Yes' if shannons_compliance else 'No'}")
    print(f"Información Mutua I(X; Y): {informacion_mutua:.10f} bits")
    print(f"Entropía Condicional H(Y|X): {entropia_condicional:.10f} bits")
    print(f"Capacidad del Canal: {capacidad:.10f} bits por símbolo")


if __name__ == "__main__":
    main()
