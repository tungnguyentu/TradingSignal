import os
from dotenv import load_dotenv
import math
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
# ---------- TA (no external TA libs) ----------
load_dotenv()

# ---------- TA (no external TA libs) ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / (loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ---------- Decision logic ----------
def make_decision(df: pd.DataFrame):
    close = df['close']
    df['ema9'] = ema(close, 9)
    df['ema21'] = ema(close, 21)
    df['rsi'] = rsi(close, 14)
    df['atr'] = atr(df, 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_bull = last.ema9 > last.ema21
    ema_bear = last.ema9 < last.ema21
    rsi_rising = last.rsi > prev.rsi
    rsi_falling = last.rsi < prev.rsi

    decision = "NO TRADE"
    reasons = []
    if ema_bull and last.rsi > 50 and rsi_rising:
        decision = "BUY"
        reasons = ["EMA9>EMA21", f"RSI {last.rsi:.1f}>50 & rising"]
    elif ema_bear and last.rsi < 50 and rsi_falling:
        decision = "SELL"
        reasons = ["EMA9<EMA21", f"RSI {last.rsi:.1f}<50 & falling"]
    else:
        reasons.append("Mixed conditions")

    entry = float(last.close)
    vol = float(last.atr) if not math.isnan(last.atr) else float((df['high'].tail(14).max() - df['low'].tail(14).min()) / 14)

    if decision == "BUY":
        sl = entry - 1.0 * vol
        tp = entry + 2.0 * vol
    elif decision == "SELL":
        sl = entry + 1.0 * vol
        tp = entry - 2.0 * vol
    else:
        sl = None
        tp = None

    return {
        "decision": decision,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "ema9": float(last.ema9),
        "ema21": float(last.ema21),
        "rsi": float(last.rsi),
        "atr": float(last.atr),
        "reasons": reasons
    }

# ---------- Binance data (python-binance) ----------
INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}

def build_client():
    # Keys not required for public klines
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    return Client(api_key=api_key, api_secret=api_secret)

def fetch_klines(symbol: str, timeframe: str, limit: int = 300, market: str = "spot") -> pd.DataFrame:
    """market: 'spot' or 'futures' (USDT-M)"""
    if timeframe not in INTERVAL_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    interval = INTERVAL_MAP[timeframe]
    client = build_client()

    if market == "futures":
        raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    else:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df = df.astype({
        "open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
    })
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["ts","open","high","low","close","volume"]]
    return df

# ---------- Telegram bot ----------
HELP = (
    "Use /signal <SYMBOL> <TIMEFRAME> [spot|futures]\n"
    "Examples:\n"
    "  /signal ETHUSDT 15m\n"
    "  /signal BTCUSDT 1h futures\n"
    "Logic: EMA(9/21) + RSI(14). TP/SL via ATR (RR 1:2)."
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I’ll help you quickly decide.\n" + HELP)

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text("Format: /signal SYMBOL [TIMEFRAME] [spot|futures]")
            return

        symbol = context.args[0].upper()              # e.g., ETHUSDT
        timeframe = (context.args[1].lower() if len(context.args) >= 2 else "15m")
        market = (context.args[2].lower() if len(context.args) >= 3 else "spot")
        if market not in ("spot", "futures"):
            market = "spot"

        df = fetch_klines(symbol, timeframe, limit=300, market=market)
        if df.empty or len(df) < 50:
            await update.message.reply_text("Not enough data.")
            return

        sig = make_decision(df)
        ts = df['ts'].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"📊 *{symbol}*  ⏱ *{timeframe}*  🏦 {market.upper()}",
            f"Time: {ts}",
            f"Decision: *{sig['decision']}*",
            f"Price: `{sig['entry']:.2f}`",
            f"EMA9/21: `{sig['ema9']:.2f}` / `{sig['ema21']:.2f}`",
            f"RSI14: `{sig['rsi']:.1f}`   ATR14: `{sig['atr']:.2f}`",
        ]
        if sig["decision"] != "NO TRADE":
            lines += [f"SL: `{sig['sl']:.2f}`", f"TP: `{sig['tp']:.2f}` (RR 1:2 via ATR)"]
        if sig["reasons"]:
            lines.append("Why: " + ", ".join(sig["reasons"]))
        lines.append("\n⚠️ Not financial advice.")

        await update.message.reply_markdown_v2("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

def main():
    token = os.getenv("TG_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TG_BOT_TOKEN env var.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    print("Bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
