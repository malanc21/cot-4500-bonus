import numpy as np
from decimal import Decimal
import sympy
from sympy import sympify

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def make_diagonally_dominant(matrix, b_vector):
    n = len(matrix)

    for i in range(n):
        pivot = matrix[i][i]
        sum_of_other_elements = sum(abs(matrix[i][i+1:]))

        # we can guarantee this pivot is the largest in the row
        if abs(pivot) > abs(sum_of_other_elements):
            continue

        # if we reach this point, this means we need to swap AT LEAST ONCE
        max_value_of_row = 0
        max_index_in_row = 0

        for j in range(n):
            current_value_in_row = abs(matrix[i][j])
            if current_value_in_row > max_value_of_row:
                max_value_of_row = current_value_in_row
                max_index_in_row = j

        # now that we have a new "pivot", we swap cur_row with the expected index
        matrix[[i, max_index_in_row]] = matrix[[max_index_in_row, i]]
        b_vector[[i, max_index_in_row]] = b_vector[[max_index_in_row, i]]

    return matrix, b_vector


def gauss_seidel(A, b, guess, accuracy):
    error1 = 1
    error2 = 1
    error3 = 1

    old_x1 = guess[0]
    old_x2 = guess[1]
    old_x3 = guess[2]

    iterations = 0

    while error1 > accuracy or error2 > accuracy or error3 > accuracy:
        x1 = (b[0] - A[0, 1] * old_x2 - A[0, 2] * old_x3) / A[0, 0]
        x2 = (b[1] - A[1, 0] * x1 - A[1, 2] * old_x3) / A[1, 1]
        x3 = (b[2] - A[2, 0] * x1 - A[2, 1] * x2) / A[2, 2]

        error1 = abs(x1 - old_x1)
        error2 = abs(x2 - old_x2)
        error3 = abs(x3 - old_x3)

        iterations = iterations + 1

        old_x1 = x1
        old_x2 = x2
        old_x3 = x3

    print(str(iterations) + "\n")


def jacobi(A, b, guess, accuracy):
    error1 = 1
    error2 = 1
    error3 = 1

    old_x1 = guess[0]
    old_x2 = guess[1]
    old_x3 = guess[2]

    iterations = 1

    while error1 > accuracy or error2 > accuracy or error3 > accuracy:
        x1 = (b[0] - A[0, 1] * old_x2 - A[0, 2] * old_x3) / A[0, 0]
        x2 = (b[1] - A[1, 0] * old_x1 - A[1, 2] * old_x3) / A[1, 1]
        x3 = (b[2] - A[2, 0] * old_x1 - A[2, 1] * old_x2) / A[2, 2]

        error1 = abs(x1 - old_x1)
        error2 = abs(x2 - old_x2)
        error3 = abs(x3 - old_x3)

        iterations = iterations + 1

        old_x1 = x1
        old_x2 = x2
        old_x3 = x3

    print(str(iterations) + "\n")


def evaluate_f(expression, initial_approximation):
    x = sympy.Symbol('x')
    expression_eval = expression.subs(x, initial_approximation)
    return expression_eval


def evaluate_derivative(expression, initial_approximation):
    x = sympy.Symbol('x')
    f_prime_eval = sympy.diff(expression, x, 1).evalf(subs={x: initial_approximation})
    return f_prime_eval


def newton_raphson(initial_approximation, tolerance, given_factor):
    iteration_counter = 0

    # finds f
    string_as_expression = sympify(given_factor, evaluate=False)
    f = evaluate_f(string_as_expression, initial_approximation)

    # finds f'
    f_prime = evaluate_derivative(string_as_expression, initial_approximation)

    approximation: float = f / f_prime
    while abs(approximation) >= tolerance:
        # finds f
        f = evaluate_f(string_as_expression, initial_approximation)

        # finds f'
        f_prime = evaluate_derivative(string_as_expression, initial_approximation)

        approximation = f / f_prime
        initial_approximation -= approximation
        iteration_counter += 1

    print(str(iteration_counter) + "\n")


def hermite_divided_diff(matrix):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if cell is already filled
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # difference = (left - upper left) / (span of x-values)
            # numerator = left - diagonal left cell
            numerator = Decimal(matrix[i][j-1] - matrix[i-1][j-1])

            denominator = Decimal(matrix[i][0]) - Decimal(matrix[i-(j-1)][0])

            operation = numerator / denominator

            matrix[i][j] = operation

    return matrix


def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))

    # populate x values
    for x in range(0, num_of_points):
        matrix[2*x][0] = x_points[x]
        matrix[2*x+1][0] = x_points[x]

    # populate y values
    for y in range(0, num_of_points):
        matrix[2*y][1] = y_points[y]
        matrix[2*y+1][1] = y_points[y]

    # populate with derivates
    for z in range(0, num_of_points):
        matrix[2 * z + 1][2] = slopes[z]

    filled_matrix = hermite_divided_diff(matrix)

    print(str(filled_matrix) + "\n")


def function(t, y):
    return y - (t**3)


def modified_eulers(original_w, start_of_t, end_of_t, iterations):
    h = (end_of_t - start_of_t) / iterations

    for i in range(0, iterations):
        t = start_of_t
        w = original_w
        h = h

        # do calculations
        incremented_t = t + h

        # w_(i+1) = w + (h/2)*[f(t,w) + f(t+h, w + h*f(t,w))]
        incremented_w = w + (h/2) * (function(t, w) + function(t + h, w + h*function(t, w)))

        start_of_t = incremented_t
        original_w = incremented_w

    print("%.5f" % original_w + "\n")


def main():
    # Basic info used in ?s 1 and 2
    matrix = np.array([[3, 1, 1],
                       [1, 4, 1],
                       [2, 3, 7]])
    b_vector = np.array([1, 3, 0])
    d_matrix, new_b = make_diagonally_dominant(matrix, b_vector)
    accuracy = 10 ** (-6)
    initial_guess = np.array([0, 0, 0])

    # QUESTION 1 / Number of iterations for Gauss-Seidel to converge
    gauss_seidel(d_matrix, new_b, initial_guess, accuracy)

    # QUESTION 2 / Number of iterations for Jacobi method to converge
    jacobi(d_matrix, new_b, initial_guess, accuracy)

    # QUESTION 3 / Number of iterations to solve f(x) using newton-raphson
    function_string = "x**3 - x**2 + 2"
    accuracy = 10**(-6)
    initial_approximation = 0.5
    newton_raphson(initial_approximation, accuracy, function_string)

    # QUESTION 4 / Print Hermite polynomial approximation matrix using divided difference
    x_points = [0.0, 1, 2]
    y_points = [1.0, 2, 4]
    slopes = [1.06, 1.23, 1.55]

    hermite_interpolation(x_points, y_points, slopes)

    # QUESTION 5 / Final value of modified eulers method
    # function is y - t^3; if not, MUST update def function
    original_w = 0.5
    start_of_t = 0
    end_of_t = 3
    iterations = 100

    modified_eulers(original_w, start_of_t, end_of_t, iterations)


if __name__ == '__main__':
    main()
