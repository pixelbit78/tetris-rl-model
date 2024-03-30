import pygame
import numpy as np
from tetris_game import Actions, TetrisGame
from tetromino import Tetromino
import torch, pickle
from agent import DeepQNetwork
import cv2, argparse

SPEED = 60  # fps
CELL_SIZE = 30  # Pixel size of a grid cell

class TetrisRender:
    def __init__(self, screen_width=700, screen_height=600, model=None, output=None):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.game = TetrisGame()
        self.action = {Actions.IDLE: 1}
        self.model = model
        self.video_output = output

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.action = self.handle_input(event)

            if not self.game.paused:
                # predict action if ai is playing
                self.action = self._predict_action()

                # loop through set of actions
                for action, value in self.action.items():
                    # game step
                    self.game.step({action: value})

                    # call game logic after game step.  set state
                    self.game.place_tetromino()
                    self.flash_lines()
                    self.game.clear_lines()
                    done = self.game.check_game_over()
                    self.game.spawn_new_tetromino()

                    # render game to screen
                    self.draw()

                    # update screen
                    pygame.display.update()
                    self.clock.tick(SPEED)

                # reset action
                self.action = {Actions.IDLE: 1}

            # check for game over
            if done:
                self.game.update_high_score()
                break;

        # Release everything if job is finished
        if self.video_output:
            self.video_output.release()

        pygame.quit()

    def _predict_action(self):
        predict_action = self.action

        if self.model is not None:
            next_actions, next_states = zip(*self.game.get_next_states().items())
            next_states = tuple(torch.tensor(t, dtype=torch.float32) for t in next_states)
            next_states = torch.stack(next_states)
            predict_map = model(next_states)[:, 0]
            max_action =  torch.argmax(predict_map).item()
            predict_action.clear()
            for i, act in enumerate(list(next_actions[max_action])):
                predict_action[Actions(i)] = act

        return predict_action


    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.toggle_pause()
            if event.key == pygame.K_LEFT:
                return {Actions.LEFT: 1}
            elif event.key == pygame.K_RIGHT:
                return {Actions.RIGHT: 1}
            elif event.key == pygame.K_DOWN:
                return {Actions.DROP: 1}
            elif event.key == pygame.K_UP:
                return {Actions.ROTATE: 1}
            elif event.key == pygame.K_LSHIFT:
                return {Actions.PLACE: 1}

        return self.action

    def write_video(self):
        # Read pixels from screen and write to video
        if self.video_output is not None:
            # Create an array from the pixel buffer
            raw_data = pygame.image.tostring(self.screen, "RGB")
            frame = np.frombuffer(raw_data, dtype=np.uint8)
            frame = frame.reshape((self.screen.get_height(), self.screen.get_width(), 3))

            # OpenCV uses BGR format, whereas Pygame uses RGB, so we need to convert the colors
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write the frame
            self.video_output.write(frame)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.draw_tetrino()
        self.draw_score()
        self.draw_next_tetromino()
        self.draw_high_score()
        self.draw_level()
        self.draw_reward()

        if self.game.check_game_over() or self.game.paused:
            self.draw_pause_message()

        self.write_video()

    def draw_tetrino(self):
        tetromino = self.game.current_tetromino
        for i, row in enumerate(tetromino.shape):
            for j, cell in enumerate(row):
                if cell:  # If the cell is not 0, draw it
                    pygame.draw.rect(
                        self.screen,
                        tetromino.color,
                        ((tetromino.x + j) * CELL_SIZE, (tetromino.y + i) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )

    def draw_board(self):
        # Draw the grid and other game state elements
        for y, row in enumerate(self.game.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.screen,
                        Tetromino.COLORS[cell-1],
                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )

        # draw grid border
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (0, 0, self.game.grid_size[0] * CELL_SIZE, self.game.grid_size[1] * CELL_SIZE),
            2
        )

    def draw_pause_message(self):
        font = pygame.font.Font(None, 72)
        if self.game.check_game_over():
            text = font.render("Game Over", True, (255, 255, 255))
        else:
            text = font.render("Paused", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen.get_width()/2, self.screen.get_height()/2))
        self.screen.blit(text, text_rect)

    def draw_next_tetromino(self):
        # Adjust these coordinates to position the next Tetromino preview
        base_x = self.screen.get_width() - 300
        base_y = 50

        for i, row in enumerate(self.game.next_tetromino.shape):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.screen,
                        self.game.next_tetromino.color,
                        ((base_x + j * CELL_SIZE), (base_y + i * CELL_SIZE), CELL_SIZE, CELL_SIZE)
                    )

        # Optional: Add text label for the next Tetromino
        font = pygame.font.Font(None, 36)
        text = font.render("Next:", True, (255, 255, 255))
        self.screen.blit(text, (base_x, base_y - 30))

    def draw_score(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.game.score}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 150))  # Adjust position as needed

    def draw_high_score(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"High Score: {self.game.high_score}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 180))  # Adjust position as needed

    def draw_level(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Level: {self.game.level}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 210))  # Adjust position as needed

    def draw_reward(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Holes: {sum(self.game.count_holes(self.game.grid))}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 240))  # Adjust position as needed

        text = font.render(f"Bumpiness: {self.game.calculate_bumpiness(self.game.grid)}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 270))  # Adjust position as needed

        text = font.render(f"Steps: {self.game.steps_taken}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 300))  # Adjust position as needed

        text = font.render(f"Lines: {self.game.total_lines_cleared}", True, (255, 255, 255))
        self.screen.blit(text, (self.screen.get_width() - 300, 330))  # Adjust position as needed

    def flash_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.game.grid) if all(row)]
        if lines_to_clear:
            for _ in range(3):  # Number of flashes
                for line in lines_to_clear:
                    for j in range(self.game.grid_size[0]):
                        self.game.grid[line][j] = len(Tetromino.COLORS)  # Set to white
                self.draw_board()
                pygame.display.update()
                pygame.time.delay(50)  # Flash delay
                for line in lines_to_clear:
                    for j in range(self.game.grid_size[0]):
                        self.game.grid[line][j] = len(Tetromino.COLORS)-1  # Set back to black
                self.draw_board()
                pygame.display.update()
                pygame.time.delay(50)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--model", type=str, default=None, help="Path to the model file.")
    parser.add_argument("--output", type=str, default=None, help="Output file for the video of the game.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    game = TetrisRender()
    model = None
    video_out = None

    # setup model
    if args.model:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        model.eval()

    # setup video output
    if args.output:
        video_out = cv2.VideoWriter(args.output, 0x00000021, 20, (game.screen.get_width(), game.screen.get_height()))

    # set props and run game
    game.model = model
    game.video_output = video_out
    game.run()