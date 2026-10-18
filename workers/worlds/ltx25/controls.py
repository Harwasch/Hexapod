"""Prompt-adapted navigation; no calibrated camera transform is implied."""
from astronex.controls import NATIVE_ACTIONS, CameraControls, finite

MODES = ('walk', 'direct', 'cruise')
CADENCES = ('responsive', 'balanced', 'smooth')
DEFAULT_EXPLORATION = {'mode': 'walk', 'cadence': 'balanced', 'speed': 0.5}
ACTIONS = [*NATIVE_ACTIONS, 'exploration']


def validate_exploration(values):
    if not isinstance(values, dict) or not values or set(values) - {'mode', 'cadence', 'speed'}:
        raise ValueError('Exploration accepts mode, cadence and speed only')
    result = dict(values)
    if 'mode' in result and result['mode'] not in MODES:
        raise ValueError('Exploration mode must be walk, direct or cruise')
    if 'cadence' in result and result['cadence'] not in CADENCES:
        raise ValueError('Exploration cadence must be responsive, balanced or smooth')
    if 'speed' in result:
        result['speed'] = finite(result['speed'], 0, 1)
    return result


class ExplorationControls:
    def __init__(self):
        self.camera = CameraControls()
        self.settings = dict(DEFAULT_EXPLORATION)
        self.cruise_suspended = False
        self.paused = False

    def semantic_state(self):
        return ({key: value[0] for key, value in self.camera.held.items()},
                {key: value[0] for key, value in self.camera.analog.items()},
                dict(self.camera.mouse), dict(self.settings), self.cruise_suspended, self.paused)

    def update(self, action, values):
        before = self.semantic_state()
        if action == 'exploration':
            self.settings.update(validate_exploration(values))
            if values.get('mode') == 'cruise':
                self.cruise_suspended = False
        else:
            self.camera.update(action, values)
            if action == 'stop':
                self.cruise_suspended = True
        return self.semantic_state() != before

    def snapshot(self, revision):
        motion = self.camera.snapshot()
        if self.settings['mode'] == 'cruise' and not self.cruise_suspended and not self.paused and not motion.get('forward', 0):
            motion['forward'] = 1.0
        return {'motion': motion, 'exploration': dict(self.settings), 'revision': revision}

    def pause(self):
        before = self.semantic_state()
        self.camera.clear()
        self.paused = True
        return self.semantic_state() != before

    def resume(self):
        changed = self.paused
        self.paused = False
        return changed

    def clear(self):
        self.camera.clear()
        self.cruise_suspended = True
