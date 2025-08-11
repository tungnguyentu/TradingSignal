# Trading Signal Bot

A trading bot that uses UT Bot and Heikin Ashi indicators for cryptocurrency trading signals.

## Setup

### 1. Clone the repository and navigate to the project directory

```bash
cd /path/to/trading-signal
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure it with your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file and replace the placeholder values:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_actual_telegram_bot_token
TELEGRAM_CHAT_ID=your_actual_chat_id

# Binance API Configuration
BINANCE_API_KEY=your_actual_binance_api_key
BINANCE_API_SECRET=your_actual_binance_api_secret

# Trading Configuration
ENABLE_TRADING=false
USE_TESTNET=true
```

### 5. Run the bot

```bash
python bot.py
```

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID for receiving notifications
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `ENABLE_TRADING`: Set to `true` to enable actual trading (default: `false`)
- `USE_TESTNET`: Set to `true` to use Binance testnet (default: `true`)

## Features

- Heikin Ashi and UT Bot indicators
- Telegram notifications
- Risk management with stop loss and take profit
- Backtest functionality
- Paper trading mode for testing

## Security

- Never commit your `.env` file to version control
- The `.env` file is already added to `.gitignore`
- Use the `.env.example` file as a template for other developers
