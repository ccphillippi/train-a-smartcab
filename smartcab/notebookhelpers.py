from contextlib import contextmanager
from .environment import Environment
from .agent import LearningAgent
from .simulator import Simulator


# Alex Martelli's stdout suppression hack
# http://stackoverflow.com/questions/2828953
class Ignore:
    def write(self, x):
        pass


@contextmanager
def silence():
    import sys
    stdout = sys.stdout
    sys.stdout = Ignore()
    try:
        yield
    finally:
        sys.stdout = stdout


def _get_agent_env(num_dummies=20, update_delay=0., display=False,
                   **agent_params):
    from functools import partial
    e = Environment(num_dummies=num_dummies)
    a = e.create_agent(partial(LearningAgent, **agent_params))
    e.set_primary_agent(a, enforce_deadline=True)

    return a, e


def generated_sim_stats(agent_env=None, n_trials=100, **agent_params):

    if agent_env is None:
        a, env = _get_agent_env(update_delay=0, display=False,
                                **agent_params)
    else:
        a, env = agent_env
        a.set_best_only()
        a.reset_stats()

    with silence():
        Simulator(env, update_delay=0., display=False).run(
            n_trials=n_trials
        )

    stats = a.get_stats()
    stats['missed_destination'] = 1. - stats['reached_destination']
    stats['always_reached_destination'] = 1.

    return (a, env), stats
