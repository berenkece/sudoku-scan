from pipeline.cells import CELLS

BOX = 3


def groups():
    rows = [[(r, c) for c in range(CELLS)] for r in range(CELLS)]
    cols = [[(r, c) for r in range(CELLS)] for c in range(CELLS)]
    boxes = [[(r, c) for r in range(CELLS) for c in range(CELLS) if (r // BOX, c // BOX) == (br, bc)]
             for br in range(BOX) for bc in range(BOX)]
    return rows + cols + boxes


def find_conflicts(grid):
    conflicts = set()
    for group in groups():
        positions = {}
        for row, col in group:
            value = grid[row][col]
            if value:
                positions.setdefault(value, []).append((row, col))
        for cells in positions.values():
            if len(cells) > 1:
                conflicts.update(cells)
    return sorted(conflicts)


def verdict(conflicts):
    return "invalid" if conflicts else "valid"
