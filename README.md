# Trading Signal Bot

A Telegram bot for trading signals using UT Bot Alerts and Heikin Ashi indicators for Binance Futures.

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

The bot now uses python-dotenv to load environment variables from a `.env` file.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your actual values:
   ```env
   # Telegram Bot Configuration
   TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
   TELEGRAM_CHAT_ID="your_telegram_chat_id_here"

   # Binance API Configuration  
   BINANCE_API_KEY="your_binance_api_key_here"
   BINANCE_API_SECRET="your_binance_api_secret_here"

   # Trading Configuration
   ENABLE_TRADING="false"            # "true" để đặt lệnh thật
   USE_TESTNET="true"                # khuyến nghị: testnet trước
   ```

### 3. Getting Required Credentials

#### Telegram Bot Token:
1. Message @BotFather on Telegram
2. Create a new bot using `/newbot`
3. Get your bot token

#### Telegram Chat ID:
1. Start a chat with your bot
2. Send a message
3. Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Find your chat ID in the response

#### Binance API Keys:
1. Log into Binance
2. Go to API Management
3. Create new API key with Futures trading permissions
4. **Important**: For testing, use Binance Testnet instead

### 4. Run the Bot

```bash
# Activate virtual environment
source venv/bin/activate

# Run the bot
python bot.py
```

## Features

- **UT Bot Alerts**: Technical analysis indicator for trend detection
- **Heikin Ashi**: Alternative candlestick representation for trend analysis
- **Risk Management**: Position sizing based on percentage risk
- **Telegram Integration**: Real-time signal notifications
- **Auto Trading**: Optional automatic order placement (when enabled)
- **Testnet Support**: Safe testing environment

## Commands

- `/start` - Get started and see available commands
- `/symbol <SYMBOL>` - Change trading symbol (default: BTCUSDT)
- `/signal` - Get latest signal analysis
- `/auto_on` - Enable automatic signal sending every 15 minutes
- `/auto_off` - Disable automatic signals
- `/status` - Check current bot status
- `/use_ha on|off` - Toggle Heikin Ashi as UT Bot source

## Security Notes

- Never commit your `.env` file to version control
- The `.env` file is already in `.gitignore`
- Use testnet for initial testing
- Keep your API keys secure
- Use IP restrictions on your Binance API keys

## Configuration

The bot configuration can be modified in the script:
- `SYMBOL_DEFAULT`: Default trading pair
- `INTERVAL`: Candlestick interval (15 minutes)
- `HA_CONSECUTIVE`: Heikin Ashi consecutive candles for signal
- `ATR_PERIOD`: ATR period for UT Bot
- `UT_MULT`: UT Bot multiplier
- `RISK_PCT`: Risk percentage per trade
- `TP_RR`: Take profit risk-reward ratio
- `MAX_LEVERAGE`: Maximum leverage for positions
