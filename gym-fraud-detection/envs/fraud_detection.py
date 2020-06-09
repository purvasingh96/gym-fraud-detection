import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import random
import json

"""
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

"""


class FraudDetectionEnv(gym.Env):
    def __init__(self):
        """
        action space -

        0 -> non_fraud
        1 -> fraud

        Observation space -

        1 -> correctly predicted
        2 -> wrongly predicted

        Rewards -

        +1 : correct prediction
        -1 : wrong prediction
        0 : initial value of reward

        Episode termination -

        An episode will terminate if agent guesses correctly (cummulative_reward > 0) or 200 steps have been completed

        Experience Replay -

        Agent will use experience replay for guidance

        """
        self.credit_card_dataset = './dataset/creditcard.csv'
        self.df_credit_card = pd.DataFrame(pd.read_csv(self.credit_card_dataset))

        self.ACTION_LOOKUP = {0: 'not_fraud', 1: 'fraud'}

        self.observation_space = spaces.Discrete(self.df_credit_card.shape[0])
        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))


        self.observation = 0

        self.initial_state = 0

        self.episode_over = False
        self.turns = 0
        self.turns_max = 200
        self.cummulative_rewards = 0
        self.action = 0
        self.state_index = 0

        self.true_positives = 0
        self.true_negatives = 0
        self.false_postives = 0
        self.false_negatives = 0

        self.total_positive_cases, self.total_negative_cases = self.total_positives_and_negatives()


    def total_positives_and_negatives(self):
        n_fraud = 0
        n_non_fraud = 0
        for state_idx in range(self.df_credit_card.shape[0]):
            if self.label_for(state_idx) == 1:
                n_fraud += 1
            else:
                n_non_fraud += 1

        return (n_fraud, n_non_fraud)



    def create_info_json_data(self, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate):
        data_set = {"true_positive_rate" : true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "true_negative_rate": true_negative_rate,
                    "false_negative_rate": false_negative_rate
                    }
        return json.dumps(data_set)


    def label_for(self, state_idx):
        return self.df_credit_card.iloc[state_idx]['Class']

    # this step will return (next_state, reward, episode_over, more_info (tpr, tnr, fpr, fnr))
    def step(self, action):
        """
                Parameters
                ----------
                action_index :
                Returns
                -------
                ob, reward, episode_over, info : tuple
                    ob (object) :
                        an environment-specific object representing your observation of
                        the environment.
                    reward (float) :
                        amount of reward achieved by the previous action. The scale
                        varies between environments, but the goal is always to increase
                        your total reward.
                    episode_over (bool) :
                        whether it's time to reset the environment again. Most (but not
                        all) tasks are divided up into well-defined episodes, and done
                        being True indicates the episode has terminated. (For example,
                        perhaps the pole tipped too far, or you lost your last life.)
                    info (dict) :
                         diagnostic information useful for debugging. It can sometimes
                         be useful for learning (for exam   ple, it might contain the raw
                         probabilities behind the environment's last state change).
                         However, official evaluations of your agent are not allowed to
                         use this for learning.
                """

        # extracting next state
        assert self.action_space.contains(action)

        label_for_current_state = self.label_for(self.state_index)

        # reward
        reward = 0


        # agent predicted fraud
        if action == 1:
            if label_for_current_state == 1:
                self.true_positives += 1
                reward += 1
            else:
                self.false_postives += 1
                reward -= 1

        # agent predicted non_fraud
        elif action == 0:
            if label_for_current_state == 0:
                self.true_negatives += 1
                reward += 1
            else:
                self.false_negatives += 1
                reward -=1

        if self.state_index <= self.df_credit_card.shape[0] - 2:
            self.state_index += 1

        self.turns += 1

        info = self.create_info_json_data(self.true_positives, self.false_postives, self.true_positives, self.false_negatives)

        if self.turns > self.turns_max or self.state_index == (self.df_credit_card.shape[0]-1):
            self.episode_over = True

        return self.state_index, reward, self.episode_over, info

        

    def reset(self):
        self.turns = 0
        self.episode_over = False
        self.sum_rewards = 0.0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
