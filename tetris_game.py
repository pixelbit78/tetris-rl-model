# tetris_game.py
from tetromino import Tetromino
import random
from enum import Enum, auto
import math
import numpy as np

class Actions(Enum):
    ROTATE = 0
    LEFT = auto()
    RIGHT = auto()
    PLACE = auto()
    DROP = auto()
    IDLE = auto()

class TetrisGame:
    def __init__(self):
        self.grid_size = (10, 20)  # Dimensions for the game grid
        self.current_tetromino = None
        self.next_tetromino = None
        self.grid = None
        self.nb_actions = 1
        self.high_score = self.read_high_score()
        self.action = Actions.IDLE
        self.restart_game()

    def restart_game(self):
        self.current_tetromino = Tetromino(self.grid_size[0] // 2, 0)
        self.next_tetromino = Tetromino(self.grid_size[0] // 2, 0)
        self.grid = [[0 for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]
        self.score = 0
        self.level = 0
        self.lines_cleared = 0
        self.total_lines_cleared = 0
        self.simulated_lines_cleared = 0
        self.paused = False
        self.steps_taken = 0
        self.action = Actions.IDLE
        self.update_drop_speed()
        self.spawn_new_tetromino()
        return self.get_state()

    def get_state(self, grid=None):
        """
        Extracts and returns features from the current game state.

        Returns:
        - tuple: A tuple of state features (total_height, holes, bumpiness, complete_lines).
        """
        simulated = grid is not None
        lines_cleared = self.simulated_lines_cleared if simulated else self.lines_cleared
        grid = grid if simulated else self.grid

        # Calculate the total height of the stack
        total_height = sum(self.column_heights(grid))

        # Calculate the number of holes
        n_holes = sum(self.count_holes(grid))

        # Calculate the bumpiness
        bumpiness = self.calculate_bumpiness(grid)

        return (total_height, n_holes, bumpiness)

    def get_next_states(self):
        simulated_game = self
        shape_x = simulated_game.current_tetromino.x
        shape_y = simulated_game.current_tetromino.y
        shape_orientation = simulated_game.current_tetromino.orientation
        shape = simulated_game.current_tetromino.shape
        possible_moves = {}

        # find positions
        for r in range(0, simulated_game.current_tetromino.max_rotations):
            for x in range(0, simulated_game.grid_size[0]):
                # move all the way left
                while simulated_game.current_tetromino.can_move(simulated_game.grid, -1, 0):
                    simulated_game.perform_action(Actions.LEFT)

                # move right
                for mx in range(x):
                    simulated_game.perform_action(Actions.RIGHT)

                # place peice
                simulated_game.perform_action(Actions.PLACE)

                # check cleared lines
                grid = simulated_game.grid_clone()
                simulated_game.place_tetromino(grid)
                simulated_game.clear_lines(grid)

                # Evaluate the simulated game state
                state_features = simulated_game.get_state(grid)

                # append possible move
                possible_moves[(r, shape_x - x if x < shape_x else 0, x - shape_x if x > shape_x else 0, 1)] = state_features

                # reset position
                simulated_game.current_tetromino.x = shape_x
                simulated_game.current_tetromino.y = shape_y

            simulated_game.perform_action(Actions.ROTATE)


        # reset position
        simulated_game.current_tetromino.x = shape_x
        simulated_game.current_tetromino.y = shape_y
        simulated_game.current_tetromino.orientation = shape_orientation
        simulated_game.current_tetromino.shape = shape

        return possible_moves

    def train_step(self, moves):
        moves = moves or {Actions.IDLE: 0}
        new_moves = {}
        if isinstance(moves, list):
            for i, move in enumerate(moves):
                new_moves[Actions(i)] = move
            moves = new_moves

        self.steps_taken += 1
        self.lines_cleared = 0
        truncated = False
        reward = 0
        done = False

        for action, value in moves.items():
            self.perform_action(action, value)

        if not self.current_tetromino.can_move(self.grid, 0, 1) and not self.paused:
            self.place_tetromino()
            lines_cleared = self.clear_lines()
            self.total_lines_cleared += lines_cleared
            done = self.check_game_over()

            reward = 1 + (lines_cleared ** 2) * self.grid_size[0]

        if not done:
            self.spawn_new_tetromino()
        else:
            self.update_high_score()
            reward = -10

        # return game over and score
        return self.get_state(), reward, done, truncated

    def step(self, moves):
        moves = moves or {Actions.IDLE: 0}
        self.steps_taken += 1

        if self.steps_taken % self.drop_speed == 0 and not self.paused:
            self.perform_action(Actions.DROP)

        for action, value in moves.items():
            self.perform_action(action, value)

        # return game over and score
        return self.get_state()

    def get_score(self):
        return self.score

    def perform_action(self, action, increment=1):
        if action == Actions.ROTATE:
            self.rotate(increment)
        elif action != Actions.IDLE:
            self.move(action, increment)

        self.action = action

    def read_high_score(self):
        try:
            with open("high_score.txt", "r") as file:
                return int(file.read())
        except (FileNotFoundError, ValueError):
            return 0

    def write_high_score(self, high_score):
        with open("high_score.txt", "w") as file:
            file.write(str(high_score))

    def update_high_score(self):
        if self.score > self.high_score:
            self.high_score = self.score
            self.write_high_score(self.high_score)
            print(f"New high score: {self.high_score}")

    def spawn_new_tetromino(self):
        if not self.current_tetromino.can_move(self.grid, 0, 1):
            self.current_tetromino = self.next_tetromino
            self.current_tetromino.x = self.grid_size[0] // 2
            self.current_tetromino.y = 0
            self.next_tetromino = Tetromino(self.grid_size[0] // 2, 0)

    def move(self, action, increment=1):
        dx, dy = 0, 0
        if action == Actions.LEFT:
            dx = -1 * increment
        elif action == Actions.RIGHT:
            dx = 1 * increment
        elif action == Actions.DROP:
            dy = 1 * increment

        if action == Actions.PLACE and self.current_tetromino.can_move(self.grid, 0, 1):
            piece = self.current_tetromino.clone()
            while piece.can_move(self.grid, 0, 1):
                piece.y += 1
            self.current_tetromino.y = piece.y
        elif self.current_tetromino.can_move(self.grid, dx, dy):
            self.current_tetromino.move(dx, dy)

    def rotate(self, increment=1):
        for _ in range(increment):
            if not self.paused and self.current_tetromino.can_rotate(self.grid):
                self.current_tetromino.rotate()

    def place_tetromino(self, grid=None):
        if not self.current_tetromino.can_move(self.grid, 0, 1) and not self.paused:
            grid = grid or self.grid
            for i, row in enumerate(self.current_tetromino.shape):
                for j, cell in enumerate(row):
                    if cell and self.current_tetromino.y + i >= 0:  # Check if part of Tetromino is within the grid
                        grid[self.current_tetromino.y + i][self.current_tetromino.x + j] = self.current_tetromino.id #self.current_tetromino.color

    def clear_lines(self, grid=None):
        lines_cleared = 0

        if not self.current_tetromino.can_move(self.grid, 0, 1) and not self.paused:
            simulated = grid is not None
            grid = grid or self.grid

            # Remove completed lines and update score
            new_grid = [row for row in grid if not all(row)]
            lines_cleared = len(grid) - len(new_grid)
            for _ in range(lines_cleared):
                new_grid.insert(0, [0 for _ in range(self.grid_size[0])])

            if not simulated:
                self.grid = new_grid
                self.lines_cleared = lines_cleared
                self.total_lines_cleared += lines_cleared
                self.score += self.calculate_score(lines_cleared)
                self.update_level()
            else:
                self.simulated_lines_cleared = lines_cleared
                grid = new_grid

        return lines_cleared

    def calculate_score(self, lines_cleared):
        scores = {1: 40, 2: 100, 3: 300, 4: 1200}
        return scores.get(lines_cleared, 0) * (self.level + 1)

    def update_level(self):
        self.level = self.total_lines_cleared // 10
        self.update_drop_speed()

    def update_drop_speed(self):
        # Adjust speed based on level; these values can be tweaked
        speeds = [60, 50, 45, 40, 35, 30, 25, 20, 10, 5, 1]  # Example speeds
        self.drop_speed = speeds[min(self.level, len(speeds) - 1)]

    def check_game_over(self):
        # Check if the top row of the grid has any blocks
        return any(cell for cell in self.grid[0])

    def toggle_pause(self):
        self.paused = not self.paused

    def column_heights(self, grid):
        heights = [0] * len(grid[0])  # Initialize heights for each column
        for col in range(len(grid[0])):
            for row in range(len(grid)):
                if grid[row][col] != 0:  # Assuming 0 represents an empty cell
                    heights[col] = len(grid) - row
                    break
        return heights

    def calculate_bumpiness(self, grid):
        heights = self.column_heights(grid)
        bumpiness = sum(abs(heights[col] - heights[col + 1]) for col in range(len(heights) - 1))
        return bumpiness

    def count_holes(self, grid):
        holes = [0] * len(grid[0])  # Initialize holes count for each column
        for col in range(len(grid[0])):
            column = [grid[row][col] for row in range(len(grid))]
            filled_found = False
            for cell in column:
                if cell != 0:
                    filled_found = True
                elif filled_found and cell == 0:
                    holes[col] += 1

        return holes

    def grid_clone(self):
        return [row[:] for row in self.grid]
