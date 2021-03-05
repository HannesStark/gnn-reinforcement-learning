import itertools

a = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9], [10, 11, 24], [12, 13], [14, 15, 25], [16, 17], [18, 19, 26], [20, 21],
     [22, 23, 27]]

b = itertools.chain.from_iterable(a)
print(max(b) + 1)
