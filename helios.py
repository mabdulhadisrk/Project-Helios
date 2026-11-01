
import numpy as np


def validate_matrix(matrix, name):
    """Check that a matrix is non-empty, rectangular and numeric."""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise ValueError(f"{name} must be a non-empty list of rows.")

    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"{name} must contain lists as rows.")

    columns = len(matrix[0])

    if columns == 0:
        raise ValueError(f"{name} cannot have empty rows.")

    for row in matrix:
        if len(row) != columns:
            raise ValueError(f"{name} must be rectangular.")

        for value in row:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must contain numeric values.")


def matrix_multiply(matrix_a, matrix_b):
    """
    Multiply two matrices using the standard
    row-by-column multiplication algorithm.

    Returns:
        result: Product matrix
        operations: Arithmetic operation counts
    """

    validate_matrix(matrix_a, "Matrix A")
    validate_matrix(matrix_b, "Matrix B")

    rows_a = len(matrix_a)
    columns_a = len(matrix_a[0])

    rows_b = len(matrix_b)
    columns_b = len(matrix_b[0])

    if columns_a != rows_b:
        raise ValueError(
            "Incompatible dimensions: "
            "columns of A must equal rows of B."
        )

    result = []

    for i in range(rows_a):
        result_row = []

        for j in range(columns_b):
            total = 0

            for k in range(columns_a):
                total += matrix_a[i][k] * matrix_b[k][j]

            result_row.append(total)

        result.append(result_row)

    multiplications = rows_a * columns_b * columns_a
    additions = rows_a * columns_b * (columns_a - 1)

    operations = {
        "multiplications": multiplications,
        "additions": additions,
        "total": multiplications + additions
    }

    return result, operations


def verify_with_numpy(matrix_a, matrix_b, result):
    """Independently verify our result using NumPy."""

    numpy_result = np.matmul(
        np.array(matrix_a),
        np.array(matrix_b)
    )

    return np.allclose(numpy_result, np.array(result))


def run_tests():
    """Run basic correctness and validation tests."""

    # Test 1: Standard matrix multiplication
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    matrix_b = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]

    result, operations = matrix_multiply(matrix_a, matrix_b)

    expected_result = [
        [58, 64],
        [139, 154]
    ]

    assert result == expected_result
    assert verify_with_numpy(matrix_a, matrix_b, result)

    # Test 2: Identity matrix
    identity = [
        [1, 0],
        [0, 1]
    ]

    test_matrix = [
        [3, 5],
        [7, 9]
    ]

    identity_result, _ = matrix_multiply(test_matrix, identity)

    assert identity_result == test_matrix
    assert verify_with_numpy(test_matrix, identity, identity_result)

    # Test 3: Incompatible dimensions
    try:
        matrix_multiply(
            [[1, 2]],
            [[1, 2]]
        )
        raise AssertionError("Incompatible dimensions were not rejected.")

    except ValueError:
        pass

    print("All tests passed.")


def main():
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    matrix_b = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]

    result, operations = matrix_multiply(matrix_a, matrix_b)

    print("PROJECT HELIOS")
    print("------------------------------")

    print("\nMatrix A:")
    print(np.array(matrix_a))

    print("\nMatrix B:")
    print(np.array(matrix_b))

    print("\nResult:")
    print(np.array(result))

    print("\nArithmetic operation count:")
    print(f"Multiplications: {operations['multiplications']}")
    print(f"Additions:       {operations['additions']}")
    print(f"Total:           {operations['total']}")

    print("\nNumPy verification:")
    print("PASSED" if verify_with_numpy(matrix_a, matrix_b, result) else "FAILED")

    print("\nRunning tests:")
    run_tests()


if __name__ == "__main__":
    main()
