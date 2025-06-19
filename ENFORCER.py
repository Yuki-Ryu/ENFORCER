import os
import ccxt
import numpy as np
import gym
import ccxt
import talib
import requests
from dotenv import load_dotenv
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
load_dotenv()

# ======================
# Custom Trading Environment
# ======================
class CryptoEnv(gym.Env):
    def __init__(self):
        super(CryptoEnv, self).__init__()

        # Initialize Bybit
        self.exchange = ccxt.bybit({
            'apiKey': os.getenv("BYBIT_API_KEY"),
            'secret': os.getenv("BYBIT_API_SECRET"),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'testnet': os.getenv("BYBIT_TESTNET") == 'true'
        })

        # Gym spaces
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(8,), # Close, MA5, Low5, High5, Volume, RSI, MACD, Signal
            dtype=np.float32
        )

        # Tracking
        self.scaler = MinMaxScaler()
        self.portfolio = {'USDT': 1000, 'BTC': 0}
        self.current_step = 0

    def _get_technical_indicators(self, closes):
        """Calculate RSI and MACD"""
        rsi = talib.RSI(np.array(closes), timeperiod=14)[-1]
        macd, signal, _ = talib.MACD(np.array(closes), fastperiod=12, slowperiod=26, signalperiod=9)
        return rsi, macd[-1], signal[-1]

    # Get initial data
    def reset(self):
        self.portfolio = {'USDT': 1000, 'BTC': 0}
        self.current_step = 0
        ohlcv = bybit.fetch_ohlcv('BTC/USDT', '1h', limit=30)
        self.state = np.array([candle[4] for candle in ohlcv])  # Use Close prices
        return self._next_observation()

    def _next_observation(self):
        rsi, macd, signal = self._get_technical_indicators(self.prices)
        obs = np.array([
            self.prices[-1],             # Close
            np.mean(self.prices[-5:]),   # MA5
            np.min(self.prices[-5:]),    # Low5
            np.max(self.prices[-5:]),    # High5
            len(self.prices),            # Volume
            rsi,                        # RSI
            macd,                       # MACD
            signal                      # MACD Signal
        ])
        return self.scaler.fit_transform(obs.reshape(1, -1)).flatten()

    def step(self, action):
        # Execute action (Buy/Sell/Hold)
        current_price = self.fetch_ticker('BTC/USDT')['last']
        reward = 0
        done = False
        info = {'action': 'hold'}
        
        # Execute action
        if action == 1 and self.portfolio['USDT'] > 10:  # Buy
            btc_amount = 10 / current_price
            self.portfolio['BTC'] += btc_amount
            self.portfolio['USDT'] -= 10
            info['action'] = '✅ BUY BTC at'
        elif action == 2 and self.portfolio['BTC'] > 0:  # Sell
            usdt_amount = self.portfolio['BTC'] * current_price
            self.portfolio['USDT'] += usdt_amount
            self.portfolio['BTC'] = 0
            info['action'] = '✅ SELL BTC at'
        
        # Get new data / Update state
        new_price = self.fetch_ticker('BTC/USDT')['last']
        self.prices = np.append(self.prices, new_price)[-30:]
        
        # Calculate reward (profit/loss)
        portfolio_value = self.portfolio['USDT'] + (self.portfolio['BTC'] * new_price)
        reward = portfolio_value - 1000  # Profit since reset
        
        # Stop conditions
        if portfolio_value < 50 or self.current_step >= 1000:
            done = True
        
        self.current_step += 1
        return self._next_observation(), reward, done, info

# ======================
# 2. Telegram Notifier
# ======================
class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def send(self, message):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, json={'chat_id': self.chat_id, 'text': message})

# ======================
# 3. Main Training Loop
# ======================
if __name__ == "__main__":
    # Initialize
    env = CryptoEnv()
    notifier = TelegramNotifier()
    check_env(env)  # Validate gym environment    

    # Train the PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )
    model.learn(total_timesteps=50_000)     # Train for 50k steps
    model.save("ppo_crypto_trader")         # Save the model

    # Test the model
    obs = env.reset()
    for i in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # Send Telegram alerts
        if info['action'] != 'hold':
            notifier.send(
                f"{info['action'].upper()} | "
                f"Step {i} | "
                f"Reward: {reward:.2f} | "
                f"BTC: {env.portfolio['BTC']:.4f} | "
                f"USDT: {env.portfolio['USDT']:.2f}"
            )

        if done:
            break

# Run the bot live
info['action'] = "🚀 Trading Bot is LIVE!"
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f}")