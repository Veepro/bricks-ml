import random

WEIGHTS_START = [2, 4, 1, 1, 8, 6, 1, 6, 7]
TETTA = 25

DIGIT_ONE = [3, 4, 8]
DIGIT_TWO = [2, 4, 7, 9]
DIGIT_THREE = [2, 3, 5, 7]
DIGIT_FOUR = [1, 5, 4, 8]
DIGIT_FIVE = [2, 1, 5, 8, 9]
DIGIT_SIX = [3, 6, 9, 8, 5]
DIGIT_SEVEN = [2, 3, 6]
DIGIT_EIGHT = [1, 2, 4, 5, 6, 9, 8]
DIGIT_NINE = [4, 2, 1, 5, 7]
DIGIT_ZERO = [1, 2, 4, 8, 9, 6]

DIGITS = [DIGIT_ZERO, DIGIT_ONE, DIGIT_TWO, DIGIT_THREE, DIGIT_FOUR, DIGIT_FIVE, DIGIT_SIX, DIGIT_SEVEN, DIGIT_EIGHT, DIGIT_NINE]


def make_dataset(n):
    dataset = []  # [int Digit, list Represent]
    for i in range(n):
        cur_digit = random.randrange(0, 10)
        dataset.append([cur_digit, [int(d + 1 in DIGITS[cur_digit]) for d in range(9)]])

    return dataset


def train_epoch(dataset, weights_start, tetta):
    digit_for_train = 0
    # (sum > tetta) => digit is ZERO
    # (sum <= tetta) => digit in NOT ZERO
    count_error = 0
    weights = weights_start
    for el in dataset:
        summ = 0
        for i in range(0, 9):
            summ += el[1][i] * weights[i]

        if (summ > tetta) and (el[0] != digit_for_train):
            count_error += 1
            for j in range(0, 9):
                weights[j] -= el[1][j]

        if (summ <= tetta) and (el[0] == digit_for_train):
            count_error += 1
            for j in range(0, 9):
                weights[j] += el[1][j]

    if count_error == 0:
        accuracy = 1
    else:
        accuracy = 1 - (count_error / 300)

    return weights, accuracy


if __name__ == "__main__":
    dataset = make_dataset(300)
    weights = WEIGHTS_START

    for i in range(12):
        cur_weight, cur_accuracy = train_epoch(dataset, weights, TETTA)
        if cur_accuracy == 1:
            break

    print(f'Vector of weights: {cur_weight}')
