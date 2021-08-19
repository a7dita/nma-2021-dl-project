#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT_DIR = Path(__file__).parent


class logbook:
    def __init__(self, agent):

        self._agent = agent
        self._action_log = []
        self._episode_log = []

    def write_actions(self, episode_num, cum_return):
        """Writes action-wise log."""
        self._action_log.append(
            {
                "episode": episode_num,
                "state": self._agent.get_state(),
                "action": self._agent._action,
                "agent_rule": self._agent._rule,
                "env_rule": self._agent._env.rule,
                "cum_return": cum_return,
            }
        )

    def write_episodes(self, episode_num, episode_steps, episode_return):
        """Writes episode-wise log."""

        # Collect the results and combine with counts.
        # steps_per_second = episode_steps / (time.time() - start_time)
        # if log_loss:
        # result['loss_avg'] = episode_loss/episode_steps
        self._episode_log.append(
            {
                "episode": episode_num,
                "episode_length": episode_steps,
                "episode_return": episode_return,
            }
        )

    def to_csv(self):
        """Writes a dataframe to root/gen_data/YY-MM-DD_HHMMSS_{metadata}.csv"""
        df_a = pd.DataFrame(self._action_log)
        df_a.set_index("episode", inplace=True)

        df_e = pd.DataFrame(self._episode_log)
        df_e.set_index("episode", inplace=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        datadir = ROOT_DIR.joinpath("gen_data")

        filepath_a = datadir.joinpath(f"{timestamp}_{self._metadata}.ACTIONS.csv")
        filepath_e = datadir.joinpath(f"{timestamp}_{self._metadata}.EPISODES.csv")

        # Create directory if needed
        os.makedirs(datadir, exist_ok=True)
        df_a.to_csv(filepath_a)
        df_e.to_csv(filepath_e)
