import os
import pickle
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from env import StockTradingEnv
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False


def stock_trade(stock_file):
    day_profits = []

    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    policy_kwargs = dict(activation_fn=th.nn.Sigmoid,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = DQN("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    model.policy.activation_fn = th.nn.Sigmoid
    model.learn(total_timesteps=int(len(df) - 1))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)

        current_price = env.envs[0].df.loc[env.envs[0].current_step, "open"]
        if action > 10:
            action_name = "buy"

            amount = (action - 10) * 50
            possible_buy_amount = int(env.envs[0].balance / current_price)
            trade_amount = amount if (
                    possible_buy_amount > amount > - env.envs[0].shares_held) else possible_buy_amount if amount > possible_buy_amount else - env.envs[0].shares_held
        elif action < 10:
            action_name = "sell"
            amount = (action - 10) * 50
            possible_buy_amount = int(env.envs[0].balance / current_price)
            trade_amount = amount if (
                    possible_buy_amount > amount > - env.envs[0].shares_held) else possible_buy_amount if amount > possible_buy_amount else - env.envs[0].shares_held
        else:
            action_name = "hold"
            trade_amount = ''

        print('*'*50)
        print(f"action: {action_name} amount: {trade_amount}")
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def find_file(path, name):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    stock_file = find_file('D:/stockdata/train', str(stock_code))

    daily_profits = stock_trade(stock_file)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('d:/stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    test_a_stock_trade('sh.601808')
    # multi_stock_trade()
