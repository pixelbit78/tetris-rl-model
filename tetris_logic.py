# tetris_logic.py
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
    CELL_SIZE = 30  # Pixel size of a grid cell
    def __init__(self):
        self.grid_size = (10, 20)  # Dimensions for the game grid
        self.current_tetromino = Tetromino(self.grid_size[0] // 2, 0)
        self.next_tetromino = Tetromino(self.grid_size[0] // 2, 0)
        self.grid = [[0 for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]
        self.drop_speed = 1000  # Tetromino drop speed in milliseconds
        self.nb_actions = 1
        self.last_drop_time = 0
        self.score = 0  # Initialize score
        self.total_lines_cleared = 0
        self.lines_cleared = 0
        self.simulated_lines_cleared = 0
        self.paused = False
        self.high_score = self.read_high_score()
        self.level = 0
        self.cells_filled = 0
        self.steps_taken = 0  # Initialize step counter
        self.action = Actions.IDLE
        self.update_drop_speed()  # Update this method to adjust speed based on level
        self.spawn_new_tetromino()
        self.restart_game()

    def get_row_transition(self, area, highest_peak):
        sum = 0
        # From highest peak to bottom
        for row in range(int(len(area) - highest_peak), len(area)):
            for col in range(1, len(area[0])):
                if area[row][col] != area[row][col - 1]:
                    sum += 1
        return sum


    def get_col_transition(self, area, peaks):
        sum = 0
        for col in range(len(area[0])):
            if peaks[col] <= 1:
                continue
            for row in range(int(len(area) - peaks[col]), len(area) - 1):
                if area[row][col] != area[row + 1][col]:
                    sum += 1
        return sum

    def get_wells(self, peaks):
        wells = []
        for i in range(len(peaks)):
            if i == 0:
                w = peaks[1] - peaks[0]
                w = w if w > 0 else 0
                wells.append(w)
            elif i == len(peaks) - 1:
                w = peaks[-2] - peaks[-1]
                w = w if w > 0 else 0
                wells.append(w)
            else:
                w1 = peaks[i - 1] - peaks[i]
                w2 = peaks[i + 1] - peaks[i]
                w1 = w1 if w1 > 0 else 0
                w2 = w2 if w2 > 0 else 0
                w = w1 if w1 >= w2 else w2
                wells.append(w)
        return wells

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
                state_features = simulated_game.get_state_features(grid)

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

    def explore(self, weights, model):
        simulated_game = self
        best_score = -float('inf')
        best_action = {Actions.ROTATE: 0, Actions.LEFT: 0, Actions.RIGHT: 0, Actions.PLACE: 0}
        shape_x = simulated_game.current_tetromino.x
        shape_y = simulated_game.current_tetromino.y
        shape_orientation = simulated_game.current_tetromino.orientation
        shape = simulated_game.current_tetromino.shape
        cnt_moves = 0
        possible_moves = []

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

                # append possible move
                possible_moves.append({Actions.ROTATE: r,
                                           Actions.LEFT: shape_x - x if x < shape_x else 0,
                                           Actions.RIGHT: x - shape_x if x > shape_x else 0,
                                           Actions.PLACE: 1})

                # reset position
                simulated_game.current_tetromino.x = shape_x
                simulated_game.current_tetromino.y = shape_y

            simulated_game.perform_action(Actions.ROTATE)

        # evaluate possible moves
        for move in possible_moves:
            for action, value in move.items():
                for _ in range(value):
                    simulated_game.perform_action(action)
                    cnt_moves += 1
                    if action == Actions.PLACE:
                        break

            # check cleared lines
            grid = simulated_game.grid_clone()
            simulated_game.place_tetromino(grid)
            simulated_game.clear_lines(grid)

            # Evaluate the simulated game state
            state_features = simulated_game.get_state_features(grid)
            if weights is None:
                score = model.activate(state_features)[0]
            else:
                score = sum(weight * feature for weight, feature in zip(weights, state_features))

            if score > best_score:
                best_score = score
                best_action = move

            # reset position
            simulated_game.current_tetromino.x = shape_x
            simulated_game.current_tetromino.y = shape_y
            simulated_game.current_tetromino.orientation = shape_orientation
            simulated_game.current_tetromino.shape = shape

        return best_action, cnt_moves, [value for _, value in best_action.items()]

    def get_state_features(self, grid=None):
        """
        Extracts and returns features from the current game state.

        Returns:
        - tuple: A tuple of state features (total_height, complete_lines, holes, bumpiness).
        """
        simulated = grid is not None
        lines_cleared = self.simulated_lines_cleared if simulated else self.lines_cleared
        grid = grid if simulated else self.grid

        # Calculate the total height of the stack
        heights = self.column_heights(grid)
        highest_peak = max(heights)
        total_height = sum(heights)

        # Calculate the number of holes
        holes = self.count_holes(grid)
        n_holes = sum(holes)
        n_cols_with_holes = np.count_nonzero(np.array(holes) > 0)

        # Calculate the bumpiness
        bumpiness = self.calculate_bumpiness(grid)

        # Row transitions
        row_transitions = self.get_row_transition(grid, highest_peak)

        # Columns transitions
        col_transitions = self.get_col_transition(grid, heights)

        # Number of cols with zero blocks
        num_pits = np.count_nonzero(np.count_nonzero(grid, axis=0) == 0)

        wells = self.get_wells(heights)
        # Deepest well
        max_wells = np.max(wells)

        # You might also want to include the number of complete lines as a feature,
        # but since that's only updated when lines are cleared, it may not be as
        # dynamic or useful for decision-making in every step. Alternatively, you
        # could simulate the next move and assess its impact on line clearance,
        # but that would be more complex and computationally expensive.

        # For simplicity, we'll skip the complete lines in this example and focus
        # on the metrics that can be directly calculated from the current state.
        return (total_height, n_holes, bumpiness, lines_cleared)
    #, num_pits, max_wells, \
     #       n_cols_with_holes, self.current_tetromino.orientation, self.current_tetromino.x, self.current_tetromino.y)

    def get_state(self):
        return self.get_state_features()

    def train_step(self, moves):
        moves = moves or {Actions.IDLE: 0}
        new_moves = {}
        if isinstance(moves, list):
            for i, move in enumerate(moves):
                new_moves[Actions(i)] = move
            moves = new_moves

        self.steps_taken += 1
        self.lines_cleared = 0
        truncated = (self.score > 10_000)
        reward = 0 #self.steps_taken * increment_reward
        done = False

        holes = sum(self.count_holes(self.grid))
        bumpiness = self.calculate_bumpiness(self.grid)

        #if self.steps_taken % 10 == 0 and not self.paused:
        #    reward = 0
        #    self.perform_action(Actions.DROP)
        #elif not self.paused:
        reward = 0
        for action, value in moves.items():
            self.perform_action(action, value)

        if not self.current_tetromino.can_move(self.grid, 0, 1) and not self.paused:
            self.place_tetromino()
            lines_cleared = self.clear_lines()
            self.total_lines_cleared += lines_cleared
            done = self.check_game_over()

            #if self.is_tetromino_adjacent_to_shape():
            #    reward = 10
            if self.calculate_bumpiness(self.grid) <= 2:
                reward = 10
            if sum(self.count_holes(self.grid)) <= holes and self.calculate_bumpiness(self.grid) <= bumpiness:
                reward += 10
            elif self.calculate_bumpiness(self.grid) <= bumpiness:
                reward += 5
            elif sum(self.count_holes(self.grid)) <= holes:
                reward += 5
            elif self.calculate_bumpiness(self.grid) > bumpiness:
                reward -= 5
            elif sum(self.count_holes(self.grid)) > holes:
                reward -= 5
            #elif self.calculate_well_setup_reward() > 0:
            #    reward = 1

            reward += 1 + (lines_cleared ** 2) * self.grid_size[0]
            #if self.is_tetromino_adjacent_to_shape():
            #    reward += self.calculate_partial_line_clear_reward()
            #reward += self.steps_taken * increment_reward
            #reward += int(self.is_tetromino_adjacent_to_shape()) * 2
            #if self.current_tetromino.y >= 16 or sum(self.count_holes(self.grid)) < 2:
            #    reward += 1
            #reward += sum(self.count_holes(self.grid)) * -increment_reward
            #reward += self.calculate_bumpiness(self.grid) * -increment_reward

        if not done:
            self.spawn_new_tetromino()
        else:
            self.update_high_score()

        # return game over and score
        return self.get_state_features(), reward, done, truncated

    def step(self, moves):
        moves = moves or {Actions.IDLE: 0}
        self.steps_taken += 1

        if self.steps_taken % 10 == 0 and not self.paused:
            self.perform_action(Actions.DROP)

        for action, value in moves.items():
            self.perform_action(action, value)

        # return game over and score
        return self.get_state_features()

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

    def update_game_state(self):
        # Drop Tetromino
        if not self.current_tetromino.can_move(self.grid, 0, 1):
            self.place_tetromino()
            self.clear_lines()
        else:
            self.perform_action(Actions.DROP)

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
        self.level = self.get_score() // 500
        self.update_drop_speed()

    def update_drop_speed(self):
        # Adjust speed based on level; these values can be tweaked
        speeds = [800, 720, 630, 550, 470, 380, 300, 220, 130, 100, 80]  # Example speeds
        self.drop_speed = speeds[min(self.level, len(speeds) - 1)]

    def check_game_over(self):
        # Check if the top row of the grid has any blocks
        return any(cell for cell in self.grid[0])

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
        self.cells_filled = 0
        self.action = Actions.IDLE
        self.update_drop_speed()
        return self.get_state()

    def toggle_pause(self):
        self.paused = not self.paused

    def tetromino_to_vector(self, tetromino):
        # Use the type ID to create a type vector directly
        type_vector = [0] * len(Tetromino.SHAPES)
        type_vector[tetromino.id-1] = 1

        # Encode the orientation
        orientation_vector = [0] * 4  # Assuming 4 orientations
        orientation_vector[tetromino.orientation] = 1

        # Normalize positions
        position_vector = [tetromino.x, tetromino.y]

        return type_vector + orientation_vector + position_vector

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

    def count_holes_diff(self, old_holes, new_holes):
        old_count = sum(old_holes)
        new_count = sum(new_holes)

        return new_count - old_count

    def calculate_bumpiness_reward(self, new_bumpiness):
        bumpiness_reward = 0
        if new_bumpiness > 0:
            bumpiness_reward = max(0, 10 - new_bumpiness)  # Example scaling factor
        return bumpiness_reward
