import random
import gym
from gym import spaces
import numpy as np
import pandas as pd


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Discrete(21)

        self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,
            0 if pd.isnull(self.df.loc[self.current_step, 'turn'] / 10) else pd.isnull(self.df.loc[self.current_step, 'turn'] / 10),
            self.df.loc[self.current_step, 'tradestatus'] / 1,
            self.df.loc[self.current_step, 'pctChg'] / 100,
            self.df.loc[self.current_step, 'peTTM'] / 1e4,
            self.df.loc[self.current_step, 'psTTM'] / 100,
            self.df.loc[self.current_step, 'pcfNcfTTM'] / 1e3,
            self.df.loc[self.current_step, 'pbMRQ'] / 100,
        ])
        return obs

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "open"]

        amount = (action - 10) * 50

        possible_buy_amount = int(self.balance / current_price)
        trade_amount = amount if (
                    possible_buy_amount > amount > - self.shares_held) else possible_buy_amount if amount > possible_buy_amount else - self.shares_held

        trade_value = trade_amount * current_price

        self.balance += trade_value
        self.shares_held += trade_amount

    def step(self, action):
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'open'].values) - 2:
            done = True

        # profits
        price_before_trade = self.df.loc[self.current_step - 1, "open"]
        price_after_trade = self.df.loc[self.current_step, "open"]

        reward = 0 if action == 0 else 1 if price_after_trade > price_before_trade and action > 10 or price_after_trade < price_before_trade and action < 10 else -100

        self.total_worth = self.balance + self.shares_held * price_after_trade

        if self.total_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.total_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        print('-'*50)
        profit = self.total_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total worth: {self.total_worth}')
        print(f'Profit: {profit}')
        return profit