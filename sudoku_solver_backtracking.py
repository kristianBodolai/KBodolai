# <codecell>
'''Sudoku solver with backtracking algorithm. Following computerphile'''

import numpy as np
import time

sudoku = [[1, 0, 0, 0, 0, 0, 0, 0, 5],
          [0, 0, 0, 1, 2, 3, 0, 0, 0],
          [0, 0, 9, 0, 0, 0, 7, 0, 0],
          [0, 3, 0, 0, 1, 0, 0, 7, 0],
          [0, 2, 0, 5, 0, 6, 0, 8, 0],
          [0, 1, 0, 0, 7, 0, 0, 9, 0],
          [0, 0, 6, 0, 0, 0, 3, 0, 0],
          [0, 0, 0, 4, 6, 8, 0, 0, 0],
          [5, 0, 0, 0, 0, 0, 0, 0, 4]]


print(np.array(sudoku))

# <codecell>
def is_legal(y, x, n, sudoku):
    for i in range(9):
        if sudoku[y][i] == n:  #checks if n number is in row y
            return False

    for i in range(9):
        if sudoku[i][x] == n:
            return False

    x_0 = (x//3)*3
    y_0 = (y//3)*3

    for i in range(3):
        for j in range(3):
            if sudoku[y_0+i][x_0+j] == n:
                return False

    return True


def solve_sudoku(sudoku):
    for y in range(9):
        for x in range(9):
            if sudoku[y][x] == 0: #if we have an empty square:
                for n in range (1, 10):

                    if is_legal(y, x, n, sudoku):
                        sudoku[y][x] = n
                        solve_sudoku(sudoku)
                        sudoku[y][x] = 0 #backtracking

                return

    print(np.array(sudoku))

# <codecell>
solve_sudoku(sudoku)
