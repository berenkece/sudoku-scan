

def validate_solution(grid: list[list[int]] | None) -> bool:
    """Validate rows, columns, and 3x3 boxes for a completed Sudoku grid."""
    if grid is None:
        return False
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    expected = set(range(1, 10))
    if any(set(row) != expected for row in grid):
        return False
    if any({grid[row][col] for row in range(9)} != expected for col in range(9)):
        return False
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = {grid[row][col] for row in range(box_row, box_row + 3) for col in range(box_col, box_col + 3)}
            if box != expected:
                return False
    return True
