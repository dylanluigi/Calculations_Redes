import math
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linprog


# Original functions provided by the user
def calculate_information_content(p_i, base=2):
    if not (0 < p_i <= 1):
        raise ValueError("Probability p_i must be between 0 and 1")
    I_i = -math.log(p_i, base)
    return I_i


def calculate_entropy(probabilities):
    # Directly use the probabilities without converting to Decimal
    entropy = sum(-p * math.log(p, 2) for p in probabilities.values() if p > 0)
    return entropy


def build_huffman_tree(probabilities):
    pq = [(prob, symbol) for symbol, prob in probabilities.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        prob1, node1 = heapq.heappop(pq)
        prob2, node2 = heapq.heappop(pq)
        merged_node = (prob1 + prob2, [node1, node2])
        heapq.heappush(pq, merged_node)
    return pq[0][1]


def calculate_huffman_codes(node, code="", mapping=None):
    if mapping is None:
        mapping = {}
    if isinstance(node, str):
        mapping[node] = code
    else:
        calculate_huffman_codes(node[0], code + "0", mapping)
        calculate_huffman_codes(node[1], code + "1", mapping)
    return mapping


def average_code_length(codes, probabilities):
    # Use probabilities directly as floats
    avg_length = sum(len(codes[symbol]) * prob for symbol, prob in probabilities.items())
    return avg_length


def simulate_transmission_error(message, error_rate):
    corrupted_message = ''
    for char in message:
        binary_representation = format(ord(char), '08b')
        corrupted_binary = ''.join(
            '1' if bit == '0' else '0' if np.random.random() < error_rate else bit
            for bit in binary_representation
        )
        corrupted_char = chr(int(corrupted_binary, 2))
        corrupted_message += corrupted_char
    return corrupted_message


# New function for channel simulation with errors
def simulate_channel_with_errors(probabilities, error_rate):
    p_y_given_x = {x: {y: 0 for y in probabilities} for x in probabilities}
    for x in probabilities:
        for y in probabilities:
            if x == y:
                p_y_given_x[x][y] = 1 - error_rate
            else:
                p_y_given_x[x][y] = error_rate / (len(probabilities) - 1)
    return p_y_given_x


# Integration of mutual information and conditional entropy calculation
def calculate_mutual_information(p_x, p_y_given_x):
    p_y = defaultdict(float)
    for x in p_x:
        for y in p_y_given_x[x]:
            p_y[y] += p_x[x] * p_y_given_x[x][y]
    I_XY = 0
    for x in p_x:
        for y in p_y_given_x[x]:
            if p_y_given_x[x][y] > 0 and p_y[y] > 0:
                I_XY += p_x[x] * p_y_given_x[x][y] * math.log(p_y_given_x[x][y] / p_y[y], 2)
    return I_XY


def calculate_conditional_entropy(p_x, p_y_given_x):
    H_Y_given_X = 0
    for x in p_x:
        for y in p_y_given_x[x]:
            if p_y_given_x[x][y] > 0:
                H_Y_given_X += p_x[x] * p_y_given_x[x][y] * math.log(p_y_given_x[x][y], 2)
    return -H_Y_given_X


def channel_capacity(p_y_given_x):
    # Objective function coefficients for minimization
    c = [-1 for _ in p_y_given_x]

    # Equality constraint: sum of probabilities must equal 1
    A_eq = [[1] * len(p_y_given_x)]
    b_eq = [1]

    # Inequality constraints: probabilities must be non-negative
    A_ub = [[-1 if j == i else 0 for j in range(len(p_y_given_x))] for i in range(len(p_y_given_x))]
    b_ub = [0 for _ in p_y_given_x]

    # Bounds for probabilities (0 <= p_x <= 1)
    bounds = [(0, 1) for _ in p_y_given_x]

    # Solve the linear programming problem
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        optimized_p_x = result.x
        channel_capacity_value = -result.fun  # Negate to get the actual value
        print(f"Optimized p(x): {optimized_p_x}")
        print(f"Channel Capacity: {channel_capacity_value:.10f} bits per symbol")
        return channel_capacity_value
    else:
        raise ValueError("Optimization failed. The error message: {}".format(result.message))


def main():
    probabilities = {
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

    # Normalize probabilities to sum to 1
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate information content for each letter
    information_content = {letter: calculate_information_content(prob) for letter, prob in
                           normalized_probabilities.items()}

    # Calculate entropy of the source
    entropy_of_source = calculate_entropy(normalized_probabilities)

    # Build Huffman coding tree
    huffman_tree = build_huffman_tree(normalized_probabilities)

    # Calculate Huffman codes
    huffman_codes = calculate_huffman_codes(huffman_tree)

    # Print information content for each letter
    for letter, info_content in information_content.items():
        print(f"Letter '{letter}' has an information content of: {info_content:.2f} bits")

    # Print the entropy of the source
    print(f"\nEntropy of the source: {entropy_of_source:.2f} bits")

    # Print Huffman codes
    print("\nHuffman Codes:")
    for symbol, code in huffman_codes.items():
        print(f"Symbol '{symbol}': {code}")

    # Calculate the average code length
    avg_code_length = average_code_length(huffman_codes, normalized_probabilities)
    print(f"Average code length: {avg_code_length:.2f} bits")

    # Since Huffman codes are CUD by design
    print("Huffman codes are uniquely decodable (CUD) by design.")

    # Calculate epsilon (ε) as the difference between the average code length and the entropy
    epsilon = avg_code_length - entropy_of_source
    print(f"Epsilon (ε): {epsilon:.2f} bits")

    # Bit Error Rate for Ethernet Cat-8
    ber_cat8 = 1E-10

    # Simulate the channel behavior
    p_y_given_x = simulate_channel_with_errors(normalized_probabilities, ber_cat8)

    # Calculate and print mutual information and conditional entropy
    mutual_info = calculate_mutual_information(normalized_probabilities, p_y_given_x)
    conditional_entropy = calculate_conditional_entropy(normalized_probabilities, p_y_given_x)
    print(f"Mutual Information I(X; Y): {mutual_info:.10f} bits")
    print(f"Conditional Entropy H(Y|X): {conditional_entropy:.10f} bits")

    Cs = channel_capacity(p_y_given_x)

    # Print the calculated channel capacity
    print(f"Channel Capacity Cs: {Cs:.10f} bits per symbol")


if __name__ == "__main__":
    main()
