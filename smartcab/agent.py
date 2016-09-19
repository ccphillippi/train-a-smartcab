import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from contextlib import contextmanager

State = namedtuple('State', 'waypoint light oncoming left right'.split())

# Heading -> linear map from absolute to relative direction
ROTATIONS = {
    # North
    (0, -1): np.array([
        [1., 0.],
        [0., 1.],
    ]),

    # East
    (1, 0): np.array([
        [0., -1.],
        [1., 0.],
    ]),

    # South
    (0, 1): np.array([
        [-1., 0.],
        [0., -1.],
    ]),

    # West
    (-1, 0): np.array([
        [0., 1.],
        [-1., 0.],
    ])
}


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, initial_Q=0., initial_alpha=1.0,
                 min_alpha=0.05, best_only=False, gamma=0.05,
                 success_span=None, alpha_span=2., seed=0):
        from collections import defaultdict
        # sets self.env = env, state = None, next_waypoint = None,
        # and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here

        self.random_state = np.random.RandomState(seed)
        self.gamma = gamma
        self.Q = defaultdict(lambda: initial_Q)
        assert initial_alpha >= min_alpha
        self.alpha = defaultdict(lambda: initial_alpha - min_alpha)
        self.min_alpha = min_alpha
        self.best_only = best_only

        self.success_sum = 0.
        self.success_denom = 0.
        if success_span is None:
            self.success_retain = 1.
        else:
            self.success_retain = 1. - 2. / (float(success_span) + 1.)
        self.alpha_decay = 1. - 2. / (float(alpha_span) + 1.)

        self.reset_stats()

    def set_best_only(self, best_only=True):
        self.best_only = best_only

    def reset(self, destination=None):
        self.planner.route_to(destination)

        if self.success_denom:
            print 'Success Rate: %.2f' % (
                self.success_sum / self.success_denom
            )

        self.success_denom = self.success_retain * self.success_denom + 1.
        self.success_sum = self.success_retain * self.success_sum
        self.trial += 1

    def reset_stats(self):
        from collections import defaultdict

        def zerodict():
            return defaultdict(lambda: 0)

        self.trial = 0
        self.success_denom = 0.
        self.success_sum = 0.
        self.suboptimals = zerodict()
        self.crimes = zerodict()
        self.optimal_stops = zerodict()
        self.successes = zerodict()
        self.reached_destination = zerodict()
        self.n_turns = zerodict()

    def get_stats(self):
        import pandas as pd
        return pd.concat({
            'crimes': pd.Series(self.crimes),
            'successes': pd.Series(self.successes),
            'optimal_stops': pd.Series(self.optimal_stops),
            'suboptimals': pd.Series(self.suboptimals),
            'reached_destination': pd.Series(self.reached_destination),
            'n_turns': pd.Series(self.n_turns),
        }, axis=1).fillna(0.)

    def choose_action(self, state, best_only=False):
        import pandas as pd

        valid_actions = self.env.valid_actions

        action_qs = np.array(
            [self.Q[(state, action)]
             for action in valid_actions]
        )

        if best_only:
            return valid_actions[np.argmax(action_qs)], None

        # Temperature decreases as we learn (affine of Q learning rate)
        temperatures = np.array(
            [self.alpha[(state, action)]
             for action in valid_actions]
        ) + self.min_alpha
        temp = np.mean(temperatures)

        action_softmax = np.exp(action_qs / temp)
        action_probs = action_softmax / np.sum(action_softmax)

        debug_info = pd.DataFrame(
            dict(
                Qs=action_qs,
                Temps=temperatures,
                T=temp,
                action_probs=action_probs,
            ),
            index=[a if a else 'None' for a in valid_actions]
        )

        # Randomly select action i based on action_probs
        u = self.random_state.rand()
        cumulativeProb = 0.
        for i, prob in enumerate(action_probs):
            cumulativeProb += prob
            if u <= cumulativeProb:
                break

        return valid_actions[i], debug_info

    def _get_state(self, waypoint):
        env = dict(self.env.sense(self))
        env['left'] = env['left'] is not None
        env['right'] = env['right'] is not None
        return State(waypoint=waypoint, **env)

    def update(self, t):
        # Gather inputs

        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self._get_state(self.next_waypoint)

        # Select action according to policy
        action, debug_info = self.choose_action(self.state, self.best_only)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # fail to award completion when taking suboptimal/illegal route
        if reward < 12 and reward > 2:
            print 'Remapping Reward: %s, %s, %s' % (
                reward, self.next_waypoint, action)
            reward -= 10

        # Learn policy based on state, action, reward
        key = (self.state, action)
        Q_prev = self.Q[key]

        new_state = self._get_state(self.planner.next_waypoint())

        if not self.best_only:
            alpha = self.alpha[key] + self.min_alpha
            best_new_action = self.choose_action(new_state, best_only=True)[0]
            self.Q[key] = (
                Q_prev * (1. - alpha) +
                alpha * (
                    reward +
                    self.gamma * self.Q[(new_state, best_new_action)]
                )
            )
            self.alpha[key] = self.alpha[key] * self.alpha_decay

        # Check if success
        agent_state = self.env.agent_states[self]
        location = np.array(agent_state['location'])
        destination = np.array(agent_state['destination'])
        if np.all(location == destination):
            self.success_sum += 1.



        # Log events
        self.n_turns[self.trial] += 1
        if reward < 0.:

            if reward == -0.5:
                self.suboptimals[self.trial] += 1
            elif reward <= -1.:
                self.crimes[self.trial] += 1

            if debug_info is not None:
                print debug_info

            print (
                "LearningAgent.update(): deadline = {}, state = {}, "
                "action = {}, reward = {}, Q_change = {}, "
            ).format(
                deadline, self.state, action, reward,
                self.Q[key] - Q_prev)  # [debug]
        elif reward == 0.:
            # if stopped when should have right turned on red
            if (
                (self.state.light == 'red') and
                (self.state.waypoint == 'right')
            ):
                self.suboptimals[self.trial] += 1
            # should have left turned on green
            elif (
                (self.state.light == 'green') and
                (self.state.waypoint == 'left') and
                (self.state.oncoming is None)
            ):
                self.suboptimals[self.trial] += 1
            else:
                self.optimal_stops[self.trial] += 1
        else:
            self.successes[self.trial] += 1
            if reward > 5:
                self.reached_destination[self.trial] += 1


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    Simulator(e, update_delay=0., display=False).run(n_trials=100)

    a.set_best_only()
    a.reset_stats()

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
