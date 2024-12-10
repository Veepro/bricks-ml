import random
import copy
import matplotlib.pyplot as plt

from artificial_neuron import make_dataset, DIGITS
from interference import add_noise

WEIGHTS_MATRIX_START = [[random.randrange(1, 10) for i in range(9)] for j in range(10)]
SIZE_DF = 1500

representation_digits = []
for d in DIGITS:
    representation_digits.append([int(i + 1 in d) for i in range(9)])


def get_max(weights, num):
    # This func can be used for recognition by a prepared matrix
    summator = []
    for i in range(10):
        summator.append((i, sum([weights[i][j] * num[1][j] for j in range(9)])))

    return sorted(summator, key=lambda x: x[1], reverse=True)[0][0]


def train_epoch(dataset, weights_start):
    weights = copy.deepcopy(weights_start)
    c = 0
    for el in dataset:
        m = get_max(weights, el)
        if m != el[0]:
            c += 1
            weights[el[0]] = [weights[el[0]][j] + el[1][j] for j in range(9)]

            weights[m] = [weights[m][j] - el[1][j] for j in range(9)]

    accuracy = 1 - (c / len(dataset))
    return weights, accuracy


if __name__ == "__main__":
    dataset = make_dataset(SIZE_DF)
    weights = copy.deepcopy(WEIGHTS_MATRIX_START)

    num_of_noise = [i for i in range(0, 10)]  # 0...9
    inaccuracy = []

    for i in num_of_noise:
        ds_with_noise = add_noise(dataset, i)
        accuracy = -1
        cur_accuracy = 0
        cur_weights = copy.deepcopy(weights)
        c = 0
        while cur_accuracy > accuracy:
            c += 1
            accuracy = cur_accuracy
            weights = cur_weights
            cur_weights, cur_accuracy = train_epoch(ds_with_noise, weights)

            if c > 15:
                break

        inaccuracy.append((1 - max(accuracy, cur_accuracy)) * 100)

    plt.plot(num_of_noise, inaccuracy, '.-')
    plt.xlabel('Number of noise')
    plt.ylabel('Probability of error, %')

    plt.title('Perceptron with noise')

    plt.show()

    print(f'{[round(el, 2) for el in inaccuracy]}')
