import ccxt
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# Connect to Bybit
bybit = ccxt.bybit({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
})

# Custom Trading Environment
class CryptoEnv(gym.Env):
    def __init__(self):
        super(CryptoEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))  # OHLCV data

    def reset(self):
        # Get initial data (Open, High, Low, Close, Volume)
        ohlcv = bybit.fetch_ohlcv('BTC/USDT', '1m', limit=5)
        self.state = np.array([candle[4] for candle in ohlcv])  # Use Close prices
        return self.state

    def step(self, action):
        # Execute action (Buy/Sell/Hold)
        current_price = bybit.fetch_ticker('BTC/USDT')['last']
        reward = 0
        
        if action == 1:  # Buy
            bybit.create_market_buy_order('BTC/USDT', 0.001)  # Buy 0.001 BTC
            print("✅ BUY BTC at", current_price)
        elif action == 2:  # Sell
            bybit.create_market_sell_order('BTC/USDT', 0.001)
            print("✅ SELL BTC at", current_price)
        
        # Get new data
        new_ohlcv = bybit.fetch_ohlcv('BTC/USDT', '1m', limit=5)
        new_state = np.array([candle[4] for candle in new_ohlcv])
        
        # Calculate reward (profit/loss)
        reward = new_state[-1] - self.state[-1]  # Reward = price change
        
        self.state = new_state
        return self.state, reward, False, {}

# Train the PPO Agent
env = CryptoEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)  # Train for 10k steps

# Save the model
model.save("ppo_crypto_trader")

# Run the bot live
print("🚀 Trading Bot is LIVE!")
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f}")