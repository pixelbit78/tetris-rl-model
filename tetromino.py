# tetromino.py
import random

class Tetromino:
    SHAPES = [
        [[1, 1, 1, 1]],  # I shape
        [[1, 1, 1],
         [0, 1, 0]],     # T shape
        [[0, 1, 1],
         [1, 1, 0]],     # S shape
        [[1, 1, 0],
         [0, 1, 1]],     # Z shape
        [[1, 1],
         [1, 1]],        # O shape
        [[1, 0, 0],
         [1, 1, 1]],     # J shape
        [[0, 0, 1],
         [1, 1, 1]]      # L shape
    ]
    COLORS = [
        (0, 255, 255),  # Cyan for I
        (128, 0, 128),  # Purple for T
        (0, 255, 0),    # Green for S
        (255, 0, 0),    # Red for Z
        (255, 255, 0),  # Yellow for O
        (0, 0, 255),    # Blue for J
        (255, 165, 0),   # Orange for L
        (0, 0, 0),       # black for clear
        (255, 255, 255)       # white for clear
    ]
    # Define max rotations for each shape
    MAX_ROTATIONS = [2, 4, 2, 2, 1, 4, 4]

    def __init__(self, x, y, id=None):
        self.id = id if id else random.randint(1, len(Tetromino.SHAPES))  # Assign a unique type ID
        self.shape = Tetromino.SHAPES[self.id-1]
        self.color = Tetromino.COLORS[self.id-1]
        self.x = x
        self.y = y
        self.orientation = 0  # Default orientation
        self.max_rotations = Tetromino.MAX_ROTATIONS[self.id-1]

    def clone(self):
        clone = Tetromino(self.x, self.y)
        clone.id = self.id
        clone.shape = [row[:] for row in self.shape]
        clone.color = self.color
        clone.orientation = self.orientation
        clone.max_rotations = self.max_rotations
        return clone

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self):
        # Update orientation on rotation
        self.orientation = (self.orientation + 1) % self.max_rotations  # Assuming 4 possible orientations
        # Rotating the matrix (shape) of the Tetromino
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

    def can_move(self, grid, dx, dy):
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell:
                    new_x = self.x + j + dx
                    new_y = self.y + i + dy
                    if new_x < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
                        return False
                    if new_y >= 0 and grid[new_y][new_x]:
                        return False
        return True

    def rotated_shape(self):
        return [list(row) for row in zip(*self.shape[::-1])]

    def can_rotate(self, grid):
        new_shape = self.rotated_shape()
        for i, row in enumerate(new_shape):
            for j, cell in enumerate(row):
                if cell:
                    new_x = self.x + j
                    new_y = self.y + i
                    if new_x < 0 or new_x >= len(grid[0]) or new_y < 0 or new_y >= len(grid):
                        return False
                    if grid[new_y][new_x]:
                        return False
        return True

    def get_positions(self):
        positions = []
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell != 0:  # Assuming a non-zero value indicates a filled part of the Tetromino
                    grid_x = self.x + j
                    grid_y = self.y + i
                    positions.append((grid_y, grid_x))
        return positions
