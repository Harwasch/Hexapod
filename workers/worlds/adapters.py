"""One explicitly selected model per worker; GPU packages stay in child processes."""
import os


def selected_adapter(model_id=None):
    model_id = model_id or os.environ.get('WORLD_MODEL_ID', 'astronex-world')
    if model_id == 'astronex-world':
        from astronex.adapter import AstronexAdapter
        return AstronexAdapter()
    if model_id == 'forge-wm':
        from forgewm.adapter import Adapter
    elif model_id == 'matrix-game-3':
        from matrix_game.adapter import Adapter
    elif model_id == 'helix-world':
        from helixworld.adapter import Adapter
    elif model_id == 'sana-wm':
        from sana_wm.adapter import Adapter
    elif model_id == 'ltx-2.5':
        from ltx25.adapter import Adapter
    else:
        raise ValueError('Unknown WORLD_MODEL_ID')
    return Adapter()
