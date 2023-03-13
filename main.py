import os
import pickle
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from env import StockTradingEnv

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

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)

        current_price = env.envs[0].df.loc[env.envs[0].current_step, "open"]
        if action[0][0] < 1:
            action_name = "buy"
            total_possible = int(env.envs[0].balance / current_price)
            shares_bought = int(total_possible * action[0][1])
            additional_cost = shares_bought * current_price
        elif action[0][0] < 2:
            action_name = "sell"
            additional_cost = int(env.envs[0].shares_held * action[0][1]) * current_price
        else:
            action_name = "hold"
            additional_cost = ''

        print('*'*50)
        print(f"action: {action_name} amount: {additional_cost}")
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
