import torch

from ChessGeneration import prepare_input, generate, MaskedDiffusion, Config


def test_prepare_input():
    config = Config()
    config.n_layers = 1
    config.embed_dim = 64
    config.n_heads = 2
    
    model = MaskedDiffusion(config)
    
    themes = ["fork", "pin"]
    rating = 1500.0
    n_attempts = 4
    
    themes_one_hot, scaled_ratings = prepare_input(themes, rating, model, n_attempts=n_attempts)
    
    assert themes_one_hot.shape == (n_attempts, config.n_themes)
    assert scaled_ratings.shape == (n_attempts,)
    assert themes_one_hot.device == next(model.parameters()).device
    assert scaled_ratings.device == next(model.parameters()).device
    
    # Check that it encoded some valid classes
    assert torch.sum(themes_one_hot) == n_attempts * len(themes)


def test_generate():
    config = Config()
    config.n_layers = 1
    config.embed_dim = 64
    config.n_heads = 2
    config.predict_moves = True
    
    model = MaskedDiffusion(config)
    
    themes = ['fork']
    rating = 1600.0
    n_attempts = 2
    steps = 2
    
    # Verify the entire real sampling executes without throwing errors
    positions = generate(themes, rating, model, n_attempts=n_attempts, steps=steps)
    assert isinstance(positions, list)


def test_generate_no_moves():
    config = Config()
    config.n_layers = 1
    config.embed_dim = 64
    config.n_heads = 2
    config.predict_moves = False
    config.__post_init__()
    
    model = MaskedDiffusion(config)
    
    themes = ['pin']
    rating = 1400.0
    n_attempts = 2
    steps = 2
    
    # Verify the entire real sampling executes without throwing errors
    positions = generate(themes, rating, model, n_attempts=n_attempts, steps=steps)
    assert isinstance(positions, list)
