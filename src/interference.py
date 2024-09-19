import copy
import matplotlib.pyplot as plt
from random import shuffle
import artificial_neuron as art


def add_noise(dataset, n):
    ds = copy.deepcopy(dataset)
    if n == 0:
        return ds

    for el in ds:
        indexes_of_noise = [i for i in range(9)]
        shuffle(indexes_of_noise)
        for i in indexes_of_noise[:n]:
            el[1][i] = 1 - el[1][i]

    return ds


if __name__ == "__main__":
    WEIGHTS_START = [2, 4, 1, 1, 8, 6, 1, 6, 7]
    TETTA = 25

    dataset = art.make_dataset(300)

    num_of_noise = [i for i in range(0, 10)]  # 0..9
    inaccuracy = []

    for i in num_of_noise:
        ds_with_noise = add_noise(dataset, i)
        accuracy = -1
        cur_accuracy = 0
        cur_weights = WEIGHTS_START
        c = 0
        while cur_accuracy > accuracy:
            c += 1
            accuracy = cur_accuracy
            cur_weights, cur_accuracy = art.train_epoch(ds_with_noise, cur_weights, TETTA)

            if c > 15:
                break

        inaccuracy.append((1 - max(accuracy, cur_accuracy)) * 100)

    plt.plot(num_of_noise, inaccuracy, '.-')
    plt.xlabel('Number of noise')
    plt.ylabel('Inaccuracy, %')

    plt.title('Artificial neuron with noise')

    plt.show()

    print(f'{[round(el, 2) for el in inaccuracy]}')
