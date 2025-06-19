# ENFORCER
- BOT ENFORCER Bybit -
This implementation is production-ready with proper credential security. For live trading, replace testnet with real API keys after testing.

- Key Features -
1. Bybit API Integration

    - Uses .env for secure credential management
    - Supports testnet (set BYBIT_TESTNET=true)

2. Telegram Notifications

    - Sends trade alerts via Telegram bot
    - Alerts for buy/sell actions with portfolio status

3. Technical Indicators
    
    - Action space: [0=Hold, 1=Buy, 2=Sell]
    - Observation space: [Price, MA5, Low5, High5, Volume, RSI, MACD, Signal]
    - Reward: Portfolio profit/loss
    - Uses TA-Lib for technical indicators

4. PPO Training

    - Uses Stable-Baselines3
    - 50,000 timesteps with saved model
    - Saves model to ppo_crypto_trader.zip


- How to Run -
1. Create .Env file with your credentials
    Configure:

    - Replace YOUR_API_KEY, YOUR_SECRET (Bybit)

    - Replace YOUR_TELEGRAM_BOT_TOKEN and YOUR_CHAT_ID

2. Install dependencies:

    pip install ccxt stable-baselines3 numpy gym
    pip install gym ccxt stable-baselines3 talib requests python-telegram-bot

    pip install -r requirements.txt

        Requirements:
            gym==0.26.2
            numpy==1.23.5
            ccxt==4.0.85
            talib==0.4.24
            requests==2.28.2
            stable-baselines3==2.0.0
            python-dotenv==1.0.0

3. Train and Run the bot:

    python ENFORCER.py

