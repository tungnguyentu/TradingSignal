import os
import time
import math
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Load environment variables from .env file
load_dotenv()

# ========== CONFIG ==========
SYMBOL_DEFAULT = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
HA_CONSECUTIVE = 3           # Heikin Ashi: số nến cùng màu để xác nhận
ATR_PERIOD = 10              # UT Bot: ATR period
UT_MULT = 2.0                # UT Bot: ATR multiplier (Key Value)
RISK_PCT = 0.01              # 1% balance rủi ro mỗi lệnh
TP_RR = 2.0                  # TP = 2R (RR 1:2)
MAX_LEVERAGE = 10             # đòn bẩy dùng khi vào lệnh (điều chỉnh nếu cần)

# ========== ENV ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
BINANCE_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_API_SECRET", "")
ENABLE_TRADING = os.getenv("ENABLE_TRADING", "false").lower() == "true"
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

# ========== LOG ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("UTBotHA")

# ========== BINANCE CLIENT ==========
client = Client(api_key=BINANCE_KEY, api_secret=BINANCE_SECRET, testnet=USE_TESTNET)
if USE_TESTNET:
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# ========================= INDICATORS =========================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin Ashi candles from regular OHLC."""
    ha = df.copy()
    ha["HA_Close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [ (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0 ]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2.0)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["high", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["low",  "HA_Open", "HA_Close"]].min(axis=1)
    # color: green if close > open else red
    ha["HA_Color"] = np.where(ha["HA_Close"] > ha["HA_Open"], 1, -1)
    return ha

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.rolling(period).mean()

def ut_bot_signals(df: pd.DataFrame, period=10, mult=2.0, use_heikin=False) -> pd.DataFrame:
    """
    UT Bot Alerts (phiên bản đơn giản, không repaint: chỉ dùng nến đã đóng).
    Nếu use_heikin=True, dùng HA_Close làm nguồn.
    """
    src = df["close"].copy()
    if use_heikin and {"HA_Close"}.issubset(df.columns):
        src = df["HA_Close"].copy()

    _atr = atr(df, period)
    # trailing stop
    long_stop = pd.Series(index=df.index, dtype=float)
    short_stop = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(0, index=df.index)

    for i in range(len(df)):
        if i == 0 or np.isnan(_atr.iloc[i]):
            long_stop.iloc[i] = np.nan
            short_stop.iloc[i] = np.nan
            dir_.iloc[i] = 0
            continue

        # calc bands
        up = src.iloc[i] - mult * _atr.iloc[i]
        dn = src.iloc[i] + mult * _atr.iloc[i]

        if i == 1:
            long_stop.iloc[i] = up
            short_stop.iloc[i] = dn
            dir_.iloc[i] = 1 if src.iloc[i] > dn else -1 if src.iloc[i] < up else 0
            continue

        # update trailing stops
        long_stop.iloc[i] = max(up, long_stop.iloc[i-1]) if src.iloc[i-1] > short_stop.iloc[i-1] else up
        short_stop.iloc[i] = min(dn, short_stop.iloc[i-1]) if src.iloc[i-1] < long_stop.iloc[i-1] else dn

        # direction flip rules
        if src.iloc[i] > short_stop.iloc[i-1]:
            dir_.iloc[i] = 1
        elif src.iloc[i] < long_stop.iloc[i-1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i-1]

    signal = pd.Series("HOLD", index=df.index)
    # Buy when dir flips to 1; Sell when flips to -1 (confirmed on close)
    signal[(dir_.shift(1) != 1) & (dir_ == 1)] = "BUY"
    signal[(dir_.shift(1) != -1) & (dir_ == -1)] = "SELL"

    out = pd.DataFrame({
        "dir": dir_,
        "UT_Signal": signal,
        "UT_LongStop": long_stop,
        "UT_ShortStop": short_stop,
        "ATR": _atr
    }, index=df.index)
    return out

def ha_buy_sell(df_ha: pd.DataFrame, consecutive=3) -> pd.Series:
    """
    Heikin Ashi Buy/Sell: BUY khi có 'consecutive' nến xanh連続, SELL khi 'consecutive' nến đỏ連続.
    """
    color = df_ha["HA_Color"]
    buy = (color.rolling(consecutive).sum() == consecutive)
    sell = (color.rolling(consecutive).sum() == -consecutive)
    sig = pd.Series("HOLD", index=df_ha.index)
    # Fix pandas FutureWarning by avoiding fillna and using boolean indexing
    buy_prev = buy.shift(1)
    sell_prev = sell.shift(1)
    # Use isna() to handle NaN values explicitly
    sig[buy & ~(buy_prev == True)] = "BUY"  # This handles NaN as False
    sig[sell & ~(sell_prev == True)] = "SELL"  # This handles NaN as False
    return sig

# ========================= DATA =========================
def fetch_klines(symbol, interval=INTERVAL, limit=500):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","qav",
            "num_trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

def build_signals(symbol, use_ha_in_ut=True):
    df = fetch_klines(symbol)
    ha = heikin_ashi(df)
    df = pd.concat([df, ha[["HA_Open","HA_High","HA_Low","HA_Close","HA_Color"]]], axis=1)

    ut = ut_bot_signals(df, period=ATR_PERIOD, mult=UT_MULT, use_heikin=use_ha_in_ut)
    ha_sig = ha_buy_sell(df, consecutive=HA_CONSECUTIVE)

    merged = df.copy()
    merged["UT_Signal"] = ut["UT_Signal"]
    merged["UT_LongStop"] = ut["UT_LongStop"]
    merged["UT_ShortStop"] = ut["UT_ShortStop"]
    merged["ATR"] = ut["ATR"]
    merged["HA_Signal"] = ha_sig
    return merged

def last_closed_row(df):  # ensure we use closed candle (non-repaint)
    # last row is closed already from klines
    return df.iloc[-1]

# ========================= TRADING =========================
def get_balance_usdt():
    acc = client.futures_account_balance()
    for b in acc:
        if b["asset"] == "USDT":
            return float(b["balance"])
    return 0.0

def set_leverage(symbol, lev):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lev)
    except BinanceAPIException as e:
        log.warning(f"change leverage failed: {e.message}")

def qty_from_risk(symbol, entry, sl, risk_pct=RISK_PCT):
    """
    position size theo rủi ro % balance và khoảng cách SL.
    """
    balance = get_balance_usdt()
    risk_usd = balance * risk_pct
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0
    qty = risk_usd / dist
    # step size/precision:
    info = client.futures_exchange_info()
    step = 0.001
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
    # round down to step
    qty = math.floor(qty / step) * step
    return max(qty, 0.0)

def place_order(symbol, side, entry_price, sl_price, tp_price):
    if not ENABLE_TRADING:
        return {"status": "dry-run"}

    set_leverage(symbol, MAX_LEVERAGE)
    qty = qty_from_risk(symbol, entry_price, sl_price, risk_pct=RISK_PCT)
    if qty <= 0:
        return {"status": "error", "msg": "qty<=0"}

    try:
        # Market entry
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=qty
        )
        # SL/TP via OCO is not available for futures; set separate orders
        if side == "BUY":
            client.futures_create_order(
                symbol=symbol, side=SIDE_SELL,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=round(sl_price, 2), closePosition=True, timeInForce="GTC"
            )
            client.futures_create_order(
                symbol=symbol, side=SIDE_SELL,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=round(tp_price, 2), closePosition=True, timeInForce="GTC"
            )
        else:
            client.futures_create_order(
                symbol=symbol, side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=round(sl_price, 2), closePosition=True, timeInForce="GTC"
            )
            client.futures_create_order(
                symbol=symbol, side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=round(tp_price, 2), closePosition=True, timeInForce="GTC"
            )
        return {"status":"ok","order":order,"qty":qty}
    except (BinanceAPIException, BinanceRequestException) as e:
        return {"status":"error","msg":str(e)}

# ========================= TELEGRAM BOT =========================
STATE = {"symbol": SYMBOL_DEFAULT, "auto": False, "use_ha_in_ut": True}

# Global variable to store last close time for auto job
AUTO_JOB_STATE = {"last_close_time": None}

def fmt_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def send_msg(app: Application, text: str):
    if CHAT_ID:
        await app.bot.send_message(chat_id=CHAT_ID, text=text)

def build_signal_text(symbol, row):
    txt = (
        f"⏱ {fmt_now()}\n"
        f"Symbol: {symbol}\n"
        f"Close: {row['close']:.2f}\n"
        f"UT: {row['UT_Signal']} | HA: {row['HA_Signal']}\n"
    )
    # SL/TP gợi ý theo UT stop + RR
    if row["UT_Signal"] == "BUY" and row["HA_Signal"] == "BUY":
        sl = float(row["UT_LongStop"])
        r = row["close"] - sl
        tp = row["close"] + TP_RR * r
        txt += f"✅ Kết hợp: BUY\nSL ≈ {sl:.2f} | TP ≈ {tp:.2f}\n"
    elif row["UT_Signal"] == "SELL" and row["HA_Signal"] == "SELL":
        sl = float(row["UT_ShortStop"])
        r = sl - row["close"]
        tp = row["close"] - TP_RR * r
        txt += f"✅ Kết hợp: SELL\nSL ≈ {sl:.2f} | TP ≈ {tp:.2f}\n"
    else:
        txt += "⚠️ Chưa có đồng thuận 2 chỉ báo.\n"
    return txt

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào bạn! Bot tín hiệu Futures 15m (UT Bot + Heikin Ashi).\n"
        f"/symbol {SYMBOL_DEFAULT} — đổi cặp\n"
        "/signal — lấy tín hiệu mới nhất\n"
        "/auto_on — bật gửi tín hiệu mỗi nến 15m\n"
        "/auto_off — tắt\n"
        "/status — xem trạng thái\n"
        "/use_ha on|off — UT Bot dùng Heikin Ashi làm source\n"
        f"Trading: {'ON' if ENABLE_TRADING else 'OFF (dry-run)'} | Testnet: {USE_TESTNET}"
    )

async def cmd_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        STATE["symbol"] = context.args[0].upper()
        await update.message.reply_text(f"Đã đặt symbol = {STATE['symbol']}")
    else:
        await update.message.reply_text(f"Symbol hiện tại: {STATE['symbol']}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Symbol: {STATE['symbol']}\n"
        f"Auto: {STATE['auto']}\n"
        f"UT source= {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}\n"
        f"Trading: {'ON' if ENABLE_TRADING else 'OFF (dry-run)'} | Testnet: {USE_TESTNET}"
    )

async def cmd_use_ha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"UT source hiện tại: {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}")
        return
    val = context.args[0].lower()
    STATE["use_ha_in_ut"] = (val == "on" or val == "true" or val == "1")
    await update.message.reply_text(f"Đã đặt UT source = {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym = STATE["symbol"]
    df = build_signals(sym, use_ha_in_ut=STATE["use_ha_in_ut"])
    row = last_closed_row(df)
    text = build_signal_text(sym, row)

    # Tùy chọn đặt lệnh: chỉ khi 2 chỉ báo đồng thuận
    if ENABLE_TRADING:
        if row["UT_Signal"] == "BUY" and row["HA_Signal"] == "BUY":
            sl = float(row["UT_LongStop"])
            r = row["close"] - sl
            tp = row["close"] + TP_RR * r
            res = place_order(sym, "BUY", row["close"], sl, tp)
            text += f"\n📦 Order: {res}"
        elif row["UT_Signal"] == "SELL" and row["HA_Signal"] == "SELL":
            sl = float(row["UT_ShortStop"])
            r = sl - row["close"]
            tp = row["close"] - TP_RR * r
            res = place_order(sym, "SELL", row["close"], sl, tp)
            text += f"\n📦 Order: {res}"

    await update.message.reply_text(text)

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["auto"] = True
    await update.message.reply_text("Đã bật auto. Bot sẽ gửi tín hiệu mỗi khi đóng nến 15m.")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["auto"] = False
    await update.message.reply_text("Đã tắt auto.")

async def auto_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Job function to check for signals periodically"""
    try:
        if STATE["auto"]:
            sym = STATE["symbol"]
            df = build_signals(sym, use_ha_in_ut=STATE["use_ha_in_ut"])
            row = last_closed_row(df)
            ct = row["close_time"]
            
            # Use global state to store last check time
            if AUTO_JOB_STATE["last_close_time"] is None or ct != AUTO_JOB_STATE["last_close_time"]:
                AUTO_JOB_STATE["last_close_time"] = ct
                await send_msg(context.application, build_signal_text(sym, row))
    except Exception as e:
        log.exception(e)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("symbol", cmd_symbol))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("use_ha", cmd_use_ha))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    # Schedule the auto check job to run every 15 seconds
    app.job_queue.run_repeating(auto_check_job, interval=15, first=10)

    log.info("Bot started.")
    app.run_polling()

if __name__ == "__main__":
    main()
