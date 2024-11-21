from artificial_neuron import make_dataset, DIGITS
from interference import add_noise

WEIGHTS_MATRIX_START = [[2 for i in range(9)] for j in range(10)]
SIZE_DF = 300

representation_digits = []
for d in DIGITS:
    representation_digits.append([int(i + 1 in d) for i in range(9)])


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


def train_epoch(dataset, weights_start):
    weights = weights_start
    count_error = 0
    for el in dataset:
        maximim = get_max(weights, el)
        if len(maximim) == 1 and maximim[0] == el[0]:
            pass

        else:
            count_error += 1

            for i in range(9):
                weights[el[0]][i] += el[1][i]

            if el[0] in maximim:
                maximim.remove(el[0])

            for m in maximim:
                for i in range(9):
                    weights[m][i] -= representation_digits[m][i]

    if count_error == 0:
        accuracy = 1
    else:
        accuracy = 1 - (count_error / SIZE_DF)

    return weights, accuracy, count_error


if __name__ == "__main__":
    ds = make_dataset(SIZE_DF)
    cur_weights = WEIGHTS_MATRIX_START
    print(cur_weights)
    for i in range(15):
        cur_weights, cur_accuracy, c = train_epoch(ds, cur_weights)
        print(cur_accuracy * 100, c)
        if cur_accuracy == 1:
            print("Success!")
            break

    print("after train:")
    print(cur_weights)
