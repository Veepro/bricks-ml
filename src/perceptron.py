from artificial_neuron import make_dataset
from interference import add_noise

WEIGHTS_MATRIX_START = [[2 for i in range(10)] for j in range(10)]


def get_max(weights, num):
    column_sum = []  # format: [(digit, summ)]
    c = 0
    for el in weights:
        summ = 0
        for i in range(9):
            summ += el[i] * num[1][i]

        column_sum.append((c, summ))
        c += 1

    column_sum = sorted(column_sum, key=lambda x: x[1], reverse=True)
    maximum = []
    prev = column_sum[0][1]
    for el in column_sum:
        if el[1] == prev:
            maximum.append(el[0])
        else:
            break

    return maximum


if __name__ == "__main__":
    ds = make_dataset(20)
