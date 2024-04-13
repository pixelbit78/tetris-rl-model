import pytest
from tetris_game import TetrisGame, Actions
from tetromino import Tetromino

def test_count_wells_middle():
    #setup grid of shapes
    game = TetrisGame()
    game.current_tetromino = Tetromino(0, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()
    game.spawn_new_tetromino()
    game.current_tetromino = Tetromino(0, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()
    game.spawn_new_tetromino()
    game.current_tetromino = Tetromino(3, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()

    # test sut
    wells = game.count_wells(game.grid)

    # assert nbr of wells
    assert wells == 2

def test_count_wells_left():
    #setup grid of shapes
    game = TetrisGame()
    game.current_tetromino = Tetromino(1, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()
    game.spawn_new_tetromino()
    game.current_tetromino = Tetromino(4, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()

    # test sut
    wells = game.count_wells(game.grid)

    # assert nbr of wells
    assert wells == 4

def test_count_wells_right():
    #setup grid of shapes
    game = TetrisGame()
    game.current_tetromino = Tetromino(7, 0, 5)
    game.perform_action(Actions.PLACE, 1)
    game.place_tetromino()

    # test sut
    wells = game.count_wells(game.grid)

    # assert nbr of wells
    assert wells == 2