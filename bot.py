import os
import time
import math
import logging
import asyncio
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
MAX_LEVERAGE = 10            # đòn bẩy dùng khi vào lệnh (điều chỉnh nếu cần)

# ===== Backtest & Trailing Stop config =====
BACKTEST_LIMIT = 1500      # số nến tải để backtest
FEE_BPS = 6                # 0.06% mỗi leg (vào/ra); tổng ~0.12%
TRAIL_USE_ATR = True       # bật trailing theo ATR trong backtest
TRAIL_ATR_MULT = 1.0       # khoảng trailing = ATR * hệ số
TRAIL_CHECK_SEC = 15       # chu kỳ cập nhật trailing stop (nếu triển khai live)

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
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2.0)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["high", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["low",  "HA_Open", "HA_Close"]].min(axis=1)
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
    long_stop = pd.Series(index=df.index, dtype=float)
    short_stop = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(0, index=df.index)

    for i in range(len(df)):
        if i == 0 or np.isnan(_atr.iloc[i]):
            long_stop.iloc[i] = np.nan
            short_stop.iloc[i] = np.nan
            dir_.iloc[i] = 0
            continue

        up = src.iloc[i] - mult * _atr.iloc[i]
        dn = src.iloc[i] + mult * _atr.iloc[i]

        if i == 1:
            long_stop.iloc[i] = up
            short_stop.iloc[i] = dn
            dir_.iloc[i] = 1 if src.iloc[i] > dn else -1 if src.iloc[i] < up else 0
            continue

        long_stop.iloc[i] = max(up, long_stop.iloc[i-1]) if src.iloc[i-1] > short_stop.iloc[i-1] else up
        short_stop.iloc[i] = min(dn, short_stop.iloc[i-1]) if src.iloc[i-1] < long_stop.iloc[i-1] else dn

        if src.iloc[i] > short_stop.iloc[i-1]:
            dir_.iloc[i] = 1
        elif src.iloc[i] < long_stop.iloc[i-1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i-1]

    signal = pd.Series("HOLD", index=df.index)
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
    Heikin Ashi Buy/Sell: BUY khi có 'consecutive' nến xanh liên tiếp, SELL khi 'consecutive' nến đỏ liên tiếp.
    """
    color = df_ha["HA_Color"]
    buy = (color.rolling(consecutive).sum() == consecutive)
    sell = (color.rolling(consecutive).sum() == -consecutive)
    sig = pd.Series("HOLD", index=df_ha.index)
    buy_prev = buy.shift(1)
    sell_prev = sell.shift(1)
    sig[buy & ~(buy_prev == True)] = "BUY"
    sig[sell & ~(sell_prev == True)] = "SELL"
    return sig

# ---------- Trailing Stop helper ----------
def calc_trailing_stop(side, price_src, atr_val, mult):
    """
    side: 'LONG' hoặc 'SHORT'; price_src: giá tham chiếu (close hoặc HA_Close); atr_val: ATR hiện tại; mult: hệ số ATR.
    """
    if np.isnan(atr_val):
        return np.nan
    if side == "LONG":
        return price_src - atr_val * mult
    else:
        return price_src + atr_val * mult

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
    info = client.futures_exchange_info()
    step = 0.001
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
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
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=qty
        )
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

# ========================= BACKTESTING (equity curve style) =========================
class BacktestResult:
    def __init__(self):
        self.trades = []
        self.initial_balance = 10000.0
        self.current_balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
    def add_trade(self, entry_time, exit_time, symbol, side, entry_price, exit_price, 
                  qty, pnl, pnl_pct, reason):
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        self.trades.append(trade)
        self.current_balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        if self.current_balance > self.max_balance:
            self.max_balance = self.current_balance
        if self.current_balance < self.min_balance:
            self.min_balance = self.current_balance
        drawdown = (self.max_balance - self.current_balance) / self.max_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_summary(self):
        if self.total_trades == 0:
            return "No trades executed in backtest."
        win_rate = self.winning_trades / self.total_trades * 100
        avg_win = sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / max(self.winning_trades, 1)
        avg_loss = sum(t['pnl'] for t in self.trades if t['pnl'] < 0) / max(self.losing_trades, 1)
        profit_factor = abs(avg_win * self.winning_trades) / abs(avg_loss * self.losing_trades) if self.losing_trades > 0 else float('inf')
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        return f"""
📊 **BACKTEST RESULTS**
🔸 Period: {self.trades[0]['entry_time'].strftime('%Y-%m-%d')} to {self.trades[-1]['exit_time'].strftime('%Y-%m-%d')}
🔸 Total Trades: {self.total_trades}
🔸 Win Rate: {win_rate:.1f}% ({self.winning_trades}W/{self.losing_trades}L)

💰 **Performance:**
🔸 Initial Balance: ${self.initial_balance:,.2f}
🔸 Final Balance: ${self.current_balance:,.2f}
🔸 Total Return: {total_return:+.2f}%
🔸 Total P&L: ${self.total_pnl:+,.2f}
🔸 Max Drawdown: {self.max_drawdown:.2f}%

📈 **Trade Stats:**
🔸 Avg Win: ${avg_win:.2f}
🔸 Avg Loss: ${avg_loss:.2f}
🔸 Profit Factor: {profit_factor:.2f}
🔸 Best Trade: ${max(t['pnl'] for t in self.trades):.2f}
🔸 Worst Trade: ${min(t['pnl'] for t in self.trades):.2f}
"""

def fetch_historical_data(symbol, interval=INTERVAL, days=30):
    """Fetch historical kline data for backtesting"""
    try:
        if interval == Client.KLINE_INTERVAL_15MINUTE:
            candles_per_day = 96
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            candles_per_day = 24
        elif interval == Client.KLINE_INTERVAL_4HOUR:
            candles_per_day = 6
        elif interval == Client.KLINE_INTERVAL_1DAY:
            candles_per_day = 1
        else:
            candles_per_day = 96
        limit = min(days * candles_per_day, 1000)
        raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        cols = ["open_time","open","high","low","close","volume","close_time","qav",
                "num_trades","tbbav","tbqav","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log.error(f"Error fetching historical data: {e}")
        return None

def run_backtest(symbol, days=30, initial_balance=10000.0, use_ha_in_ut=True):
    """Equity-curve style backtest (giữ nguyên logic TP/SL cố định, không trailing)."""
    df = fetch_historical_data(symbol, days=days)
    if df is None or len(df) < 50:
        return None, "Error: Could not fetch sufficient historical data"
    ha = heikin_ashi(df)
    df = pd.concat([df, ha[["HA_Open","HA_High","HA_Low","HA_Close","HA_Color"]]], axis=1)
    ut = ut_bot_signals(df, period=ATR_PERIOD, mult=UT_MULT, use_heikin=use_ha_in_ut)
    ha_sig = ha_buy_sell(df, consecutive=HA_CONSECUTIVE)
    df["UT_Signal"] = ut["UT_Signal"]
    df["UT_LongStop"] = ut["UT_LongStop"]
    df["UT_ShortStop"] = ut["UT_ShortStop"]
    df["ATR"] = ut["ATR"]
    df["HA_Signal"] = ha_sig
    
    result = BacktestResult()
    result.initial_balance = initial_balance
    result.current_balance = initial_balance
    
    current_position = None
    entry_price = 0
    entry_time = None
    stop_loss = 0
    take_profit = 0
    qty = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        if current_position:
            exit_price = None
            exit_reason = None
            if current_position == 'LONG':
                if row['low'] <= stop_loss:
                    exit_price = stop_loss; exit_reason = 'Stop Loss'
                elif row['high'] >= take_profit:
                    exit_price = take_profit; exit_reason = 'Take Profit'
            else:
                if row['high'] >= stop_loss:
                    exit_price = stop_loss; exit_reason = 'Stop Loss'
                elif row['low'] <= take_profit:
                    exit_price = take_profit; exit_reason = 'Take Profit'
            if not exit_price:
                if current_position == 'LONG' and row['UT_Signal'] == 'SELL' and row['HA_Signal'] == 'SELL':
                    exit_price = row['close']; exit_reason = 'Signal Reversal'
                elif current_position == 'SHORT' and row['UT_Signal'] == 'BUY' and row['HA_Signal'] == 'BUY':
                    exit_price = row['close']; exit_reason = 'Signal Reversal'
            if exit_price:
                pnl = (exit_price - entry_price) * qty if current_position == 'LONG' else (entry_price - exit_price) * qty
                pnl_pct = pnl / (result.current_balance * RISK_PCT) * 100
                result.add_trade(entry_time, row['close_time'], symbol, current_position,
                                 entry_price, exit_price, qty, pnl, pnl_pct, exit_reason)
                current_position = None
        
        if not current_position:
            if row['UT_Signal'] == 'BUY' and row['HA_Signal'] == 'BUY':
                current_position = 'LONG'
                entry_price = row['close']; entry_time = row['close_time']
                stop_loss = float(row['UT_LongStop'])
                risk_amount = result.current_balance * RISK_PCT
                qty = risk_amount / abs(entry_price - stop_loss)
                take_profit = entry_price + (entry_price - stop_loss) * TP_RR
            elif row['UT_Signal'] == 'SELL' and row['HA_Signal'] == 'SELL':
                current_position = 'SHORT'
                entry_price = row['close']; entry_time = row['close_time']
                stop_loss = float(row['UT_ShortStop'])
                risk_amount = result.current_balance * RISK_PCT
                qty = risk_amount / abs(stop_loss - entry_price)
                take_profit = entry_price - (stop_loss - entry_price) * TP_RR
    
    if current_position:
        last_row = df.iloc[-1]
        exit_price = last_row['close']
        pnl = (exit_price - entry_price) * qty if current_position == 'LONG' else (entry_price - exit_price) * qty
        pnl_pct = pnl / (result.current_balance * RISK_PCT) * 100
        result.add_trade(entry_time, last_row['close_time'], symbol, current_position,
                         entry_price, exit_price, qty, pnl, pnl_pct, 'End of Data')
    return result, None

# ========================= BACKTEST RR (R-based, có trailing ATR) =========================
def run_backtest_rr(symbol, use_ha_in_ut=True, rr=2.0, ha_n=HA_CONSECUTIVE,
                    atr_period=ATR_PERIOD, ut_mult=UT_MULT,
                    trail_use_atr=TRAIL_USE_ATR, trail_mult=TRAIL_ATR_MULT,
                    fee_bps=FEE_BPS, limit=BACKTEST_LIMIT):
    """
    Backtest theo R (risk units) + tùy chọn trailing ATR.
    Entry khi UT_Signal == HA_Signal. SL ban đầu = UT stop. TP = entry ± rr * (entry - SL0).
    PnL tính theo R, trừ phí gần đúng.
    """
    df = fetch_klines(symbol, limit=limit)
    ha = heikin_ashi(df)
    df = pd.concat([df, ha[["HA_Open","HA_High","HA_Low","HA_Close","HA_Color"]]], axis=1)
    ut = ut_bot_signals(df, period=atr_period, mult=ut_mult, use_heikin=use_ha_in_ut)
    ha_sig = ha_buy_sell(df, consecutive=ha_n)
    df["UT_Signal"] = ut["UT_Signal"]
    df["UT_LongStop"] = ut["UT_LongStop"]
    df["UT_ShortStop"] = ut["UT_ShortStop"]
    df["ATR"] = ut["ATR"]
    df["HA_Signal"] = ha_sig
    src = df["HA_Close"] if use_ha_in_ut else df["close"]

    trades = []
    in_pos = False
    pos = {}
    for i in range(atr_period + 5, len(df)):
        if not in_pos:
            sig_ut = df.at[df.index[i], "UT_Signal"]
            sig_ha = df.at[df.index[i], "HA_Signal"]
            if sig_ut == "BUY" and sig_ha == "BUY":
                entry = df.at[df.index[i], "close"]
                sl0 = float(df.at[df.index[i], "UT_LongStop"])
                if not np.isnan(sl0) and sl0 < entry:
                    tp = entry + rr * (entry - sl0)
                    pos = {"side":"LONG","entry_idx":i,"entry_price":entry,"sl0":sl0,"tp":tp,"cur_sl":sl0}
                    if trail_use_atr:
                        trail = calc_trailing_stop("LONG", src.iloc[i], df.at[df.index[i], "ATR"], trail_mult)
                        if not np.isnan(trail): pos["cur_sl"] = max(pos["cur_sl"], trail)
                    in_pos = True
            elif sig_ut == "SELL" and sig_ha == "SELL":
                entry = df.at[df.index[i], "close"]
                sl0 = float(df.at[df.index[i], "UT_ShortStop"])
                if not np.isnan(sl0) and sl0 > entry:
                    tp = entry - rr * (sl0 - entry)
                    pos = {"side":"SHORT","entry_idx":i,"entry_price":entry,"sl0":sl0,"tp":tp,"cur_sl":sl0}
                    if trail_use_atr:
                        trail = calc_trailing_stop("SHORT", src.iloc[i], df.at[df.index[i], "ATR"], trail_mult)
                        if not np.isnan(trail): pos["cur_sl"] = min(pos["cur_sl"], trail)
                    in_pos = True
        else:
            if trail_use_atr:
                trail = calc_trailing_stop(pos["side"], src.iloc[i], df.at[df.index[i], "ATR"], trail_mult)
                if not np.isnan(trail):
                    if pos["side"] == "LONG":
                        pos["cur_sl"] = max(pos["cur_sl"], trail)
                    else:
                        pos["cur_sl"] = min(pos["cur_sl"], trail)

            hi = df.at[df.index[i], "high"]
            lo = df.at[df.index[i], "low"]
            exit_price = None
            reason = None
            if pos["side"] == "LONG":
                if lo <= pos["cur_sl"]:
                    exit_price = pos["cur_sl"]; reason = "SL"
                if hi >= pos["tp"] and exit_price is None:
                    exit_price = pos["tp"]; reason = "TP"
            else:
                if hi >= pos["cur_sl"]:
                    exit_price = pos["cur_sl"]; reason = "SL"
                if lo <= pos["tp"] and exit_price is None:
                    exit_price = pos["tp"]; reason = "TP"

            if exit_price is not None:
                R = abs(pos["entry_price"] - pos["sl0"])
                pnl_R = (exit_price - pos["entry_price"]) / R if pos["side"]=="LONG" else (pos["entry_price"] - exit_price) / R
                fee_per_R = (fee_bps / 10000.0) * 2 / (R / pos["entry_price"])
                pnl_adj = pnl_R - fee_per_R
                trades.append(pnl_adj)
                in_pos = False

    wins = sum(1 for r in trades if r > 0)
    losses = len(trades) - wins
    winrate = wins / len(trades) * 100 if trades else 0
    total_R = sum(trades)
    avg_R = total_R / len(trades) if trades else 0
    pos_sum = sum(r for r in trades if r > 0)
    neg_sum = sum(-r for r in trades if r < 0)
    pf = pos_sum / neg_sum if neg_sum > 0 else float('inf')

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "winrate_pct": round(winrate, 2),
        "total_R": round(total_R, 2),
        "avg_R": round(avg_R, 3),
        "profit_factor": round(pf, 2) if np.isfinite(pf) else "∞",
        "params": dict(rr=rr, ha_n=ha_n, atr_period=atr_period,
                       ut_mult=ut_mult, trail_use_atr=trail_use_atr,
                       trail_mult=trail_mult, use_ha=use_ha_in_ut)
    }

# ========================= TELEGRAM BOT =========================
STATE = {
    "symbols": [SYMBOL_DEFAULT],   # list symbols
    "current_symbol": SYMBOL_DEFAULT,
    "auto": False,
    "use_ha_in_ut": True
}
AUTO_JOB_STATE = {}  # {"BTCUSDT": last_close_time, ...}

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
    if row["UT_Signal"] == "BUY" and row["HA_Signal"] == "BUY":
        sl = float(row["UT_LongStop"]); r = row["close"] - sl; tp = row["close"] + TP_RR * r
        txt += f"✅ Kết hợp: BUY\nSL ≈ {sl:.2f} | TP ≈ {tp:.2f}\n"
    elif row["UT_Signal"] == "SELL" and row["HA_Signal"] == "SELL":
        sl = float(row["UT_ShortStop"]); r = sl - row["close"]; tp = row["close"] - TP_RR * r
        txt += f"✅ Kết hợp: SELL\nSL ≈ {sl:.2f} | TP ≈ {tp:.2f}\n"
    else:
        txt += "⚠️ Chưa có đồng thuận 2 chỉ báo.\n"
    return txt

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Chào bạn! Bot tín hiệu Futures 15m (UT Bot + Heikin Ashi)\n\n"
        "📋 **Quản lý Symbols:**\n"
        "/symbol — xem danh sách và trạng thái\n"
        "/symbol add ETHUSDT — thêm symbol mới\n"
        "/symbol remove ETHUSDT — xóa symbol\n"
        "/symbol set ETHUSDT — chọn symbol hiện tại\n"
        "/symbol list — xem tất cả symbols\n\n"
        "📈 **Tín hiệu:**\n"
        "/signal — lấy tín hiệu cho symbol hiện tại\n"
        "/signals_all — tín hiệu cho TẤT CẢ symbols\n"
        "/auto_on — bật auto\n"
        "/auto_off — tắt auto\n\n"
        "🧪 **Backtest:**\n"
        "/backtest [days] — backtest equity-curve (TP/SL cố định)\n"
        "/backtest_rr [symbol] [RR] — backtest theo R + trailing ATR (vd: /backtest_rr BTCUSDT 2)\n\n"
        "⚙️ **Cài đặt:**\n"
        "/status — xem trạng thái\n"
        "/use_ha on|off — UT dùng Heikin Ashi làm source\n\n"
        f"💼 Trading: {'ON' if ENABLE_TRADING else 'OFF (dry-run)'} | "
        f"🌐 Testnet: {'ON' if USE_TESTNET else 'OFF'}"
    )

async def cmd_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        action = context.args[0].lower()
        if action == "add" and len(context.args) > 1:
            new_symbol = context.args[1].upper()
            if new_symbol not in STATE["symbols"]:
                STATE["symbols"].append(new_symbol)
                await update.message.reply_text(f"✅ Đã thêm symbol: {new_symbol}\nDanh sách: {', '.join(STATE['symbols'])}")
            else:
                await update.message.reply_text(f"⚠️ Symbol {new_symbol} đã có trong danh sách")
        elif action == "remove" and len(context.args) > 1:
            symbol_to_remove = context.args[1].upper()
            if symbol_to_remove in STATE["symbols"] and len(STATE["symbols"]) > 1:
                STATE["symbols"].remove(symbol_to_remove)
                if STATE["current_symbol"] == symbol_to_remove:
                    STATE["current_symbol"] = STATE["symbols"][0]
                await update.message.reply_text(f"❌ Đã xóa symbol: {symbol_to_remove}\nDanh sách: {', '.join(STATE['symbols'])}")
            elif symbol_to_remove not in STATE["symbols"]:
                await update.message.reply_text(f"⚠️ Symbol {symbol_to_remove} không có trong danh sách")
            else:
                await update.message.reply_text("⚠️ Không thể xóa symbol cuối cùng")
        elif action == "set" and len(context.args) > 1:
            symbol_to_set = context.args[1].upper()
            if symbol_to_set in STATE["symbols"]:
                STATE["current_symbol"] = symbol_to_set
                await update.message.reply_text(f"🎯 Đã đặt symbol hiện tại: {STATE['current_symbol']}")
            else:
                await update.message.reply_text(f"⚠️ Symbol {symbol_to_set} chưa có trong danh sách. Dùng /symbol add {symbol_to_set} trước")
        elif action == "list":
            current_mark = "👉"
            symbol_list = []
            for sym in STATE["symbols"]:
                mark = current_mark if sym == STATE["current_symbol"] else "   "
                symbol_list.append(f"{mark} {sym}")
            await update.message.reply_text(f"📋 Danh sách symbols:\n" + "\n".join(symbol_list))
        else:
            new_symbol = context.args[0].upper()
            if new_symbol not in STATE["symbols"]:
                STATE["symbols"].append(new_symbol)
            STATE["current_symbol"] = new_symbol
            await update.message.reply_text(f"✅ Đã đặt symbol: {STATE['current_symbol']}")
    else:
        current_mark = "👉"
        symbol_list = []
        for sym in STATE["symbols"]:
            mark = current_mark if sym == STATE["current_symbol"] else "   "
            symbol_list.append(f"{mark} {sym}")
        await update.message.reply_text(
            f"📋 Danh sách symbols:\n" + "\n".join(symbol_list) + 
            f"\n\n🎯 Hiện tại: {STATE['current_symbol']}" +
            f"\n\n💡 Sử dụng:\n" +
            f"/symbol add ETHUSDT - thêm symbol\n" +
            f"/symbol remove ETHUSDT - xóa symbol\n" +
            f"/symbol set ETHUSDT - chọn symbol hiện tại\n" +
            f"/symbol list - xem danh sách"
        )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol_count = len(STATE["symbols"])
    symbols_text = ", ".join(STATE["symbols"]) if symbol_count <= 3 else f"{', '.join(STATE['symbols'][:3])}... ({symbol_count} total)"
    await update.message.reply_text(
        f"📊 Trạng thái Bot:\n"
        f"🎯 Symbol hiện tại: {STATE['current_symbol']}\n"
        f"📋 Tất cả symbols ({symbol_count}): {symbols_text}\n"
        f"🤖 Auto: {'ON' if STATE['auto'] else 'OFF'}\n"
        f"📈 UT source: {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}\n"
        f"💼 Trading: {'ON' if ENABLE_TRADING else 'OFF (dry-run)'}\n"
        f"🌐 Testnet: {'ON' if USE_TESTNET else 'OFF'}"
    )

async def cmd_use_ha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"UT source hiện tại: {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}")
        return
    val = context.args[0].lower()
    STATE["use_ha_in_ut"] = (val == "on" or val == "true" or val == "1")
    await update.message.reply_text(f"Đã đặt UT source = {'HeikinAshi' if STATE['use_ha_in_ut'] else 'Close'}")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym = STATE["current_symbol"]
    df = build_signals(sym, use_ha_in_ut=STATE["use_ha_in_ut"])
    row = last_closed_row(df)
    text = build_signal_text(sym, row)

    if ENABLE_TRADING:
        if row["UT_Signal"] == "BUY" and row["HA_Signal"] == "BUY":
            sl = float(row["UT_LongStop"]); r = row["close"] - sl; tp = row["close"] + TP_RR * r
            res = place_order(sym, "BUY", row["close"], sl, tp)
            text += f"\n📦 Order: {res}"
        elif row["UT_Signal"] == "SELL" and row["HA_Signal"] == "SELL":
            sl = float(row["UT_ShortStop"]); r = sl - row["close"]; tp = row["close"] - TP_RR * r
            res = place_order(sym, "SELL", row["close"], sl, tp)
            text += f"\n📦 Order: {res}"

    await update.message.reply_text(text)

async def cmd_signals_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(STATE["symbols"]) > 5:
        await update.message.reply_text("⚠️ Quá nhiều symbols (>5). Dùng /signal để xem từng cái một.")
        return
    messages = []
    for sym in STATE["symbols"]:
        try:
            df = build_signals(sym, use_ha_in_ut=STATE["use_ha_in_ut"])
            row = last_closed_row(df)
            text = build_signal_text(sym, row)
            messages.append(text)
        except Exception as e:
            messages.append(f"❌ {sym}: Lỗi khi lấy dữ liệu - {str(e)[:50]}")
    full_text = "\n" + "="*30 + "\n".join(messages)
    if len(full_text) > 4000:
        for msg in messages:
            await update.message.reply_text(msg)
    else:
        await update.message.reply_text(full_text)

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["auto"] = True
    symbol_count = len(STATE["symbols"])
    await update.message.reply_text(
        f"✅ Đã bật auto monitoring cho {symbol_count} symbol(s):\n"
        f"{', '.join(STATE['symbols'])}\n\n"
        f"Bot sẽ gửi tín hiệu mỗi khi có nến 15m mới."
    )

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["auto"] = False
    await update.message.reply_text("❌ Đã tắt auto monitoring cho tất cả symbols.")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        days = 30
        symbol = STATE["current_symbol"]
        if context.args:
            try:
                days = int(context.args[0])
                days = max(1, min(days, 90))
            except ValueError:
                await update.message.reply_text("⚠️ Số ngày không hợp lệ. Sử dụng: /backtest [số_ngày]")
                return
        await update.message.reply_text(f"🔄 Đang chạy backtest cho {symbol} - {days} ngày...")
        result, error = run_backtest(symbol, days=days, use_ha_in_ut=STATE["use_ha_in_ut"])
        if error or result is None:
            await update.message.reply_text(f"❌ Lỗi backtest: {error}")
            return
        if result.total_trades == 0:
            await update.message.reply_text(f"⚠️ Không có trade nào được thực hiện trong {days} ngày qua.")
            return
        summary = result.get_summary()
        await update.message.reply_text(summary)
        if len(result.trades) > 0:
            recent_trades = result.trades[-5:]
            trades_text = "📋 **5 TRADES GẦN NHẤT:**\n"
            for i, trade in enumerate(recent_trades, 1):
                profit_emoji = "💚" if trade['pnl'] > 0 else "❤️"
                trades_text += f"{i}. {profit_emoji} {trade['side']} @{trade['entry_price']:.4f} → {trade['exit_price']:.4f}\n"
                trades_text += f"   P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%) - {trade['reason']}\n\n"
            if len(trades_text) < 4000:
                await update.message.reply_text(trades_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi khi chạy backtest: {str(e)}")
        log.exception(e)

async def cmd_backtest_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(STATE["symbols"]) > 5:
            await update.message.reply_text("⚠️ Quá nhiều symbols (>5). Dùng /backtest để test từng cái một.")
            return
        days = 30
        if context.args:
            try:
                days = int(context.args[0])
                days = max(1, min(days, 90))
            except ValueError:
                await update.message.reply_text("⚠️ Số ngày không hợp lệ. Sử dụng: /backtest_all [số_ngày]")
                return
        await update.message.reply_text(f"🔄 Đang chạy backtest cho {len(STATE['symbols'])} symbols - {days} ngày...")
        all_results = []
        for symbol in STATE["symbols"]:
            result, error = run_backtest(symbol, days=days, use_ha_in_ut=STATE["use_ha_in_ut"])
            if error or result is None:
                all_results.append(f"❌ {symbol}: {error}"); continue
            if result.total_trades == 0:
                all_results.append(f"⚠️ {symbol}: Không có trades"); continue
            win_rate = result.winning_trades / result.total_trades * 100
            total_return = (result.current_balance - result.initial_balance) / result.initial_balance * 100
            all_results.append(
                f"📊 **{symbol}**\n"
                f"   Trades: {result.total_trades} | Win: {win_rate:.1f}%\n"
                f"   Return: {total_return:+.2f}% | DD: {result.max_drawdown:.2f}%\n"
                f"   P&L: ${result.total_pnl:+,.2f}"
            )
        summary_text = f"📈 **BACKTEST TẤT CẢ SYMBOLS ({days} ngày)**\n\n" + "\n\n".join(all_results)
        if len(summary_text) > 4000:
            for result in all_results:
                await update.message.reply_text(result)
        else:
            await update.message.reply_text(summary_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi khi chạy backtest: {str(e)}")
        log.exception(e)

# ------- New command: RR/Winrate backtest with ATR trailing -------
async def cmd_backtest_rr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usage: /backtest_rr [symbol] [RR]
    sym = STATE["current_symbol"]
    rr = 2.0
    if context.args:
        if len(context.args) >= 1:
            sym = context.args[0].upper()
        if len(context.args) >= 2:
            try:
                rr = float(context.args[1])
            except ValueError:
                pass
    res = run_backtest_rr(
        symbol=sym,
        use_ha_in_ut=STATE["use_ha_in_ut"],
        rr=rr,
        ha_n=HA_CONSECUTIVE,
        atr_period=ATR_PERIOD,
        ut_mult=UT_MULT,
        trail_use_atr=TRAIL_USE_ATR,
        trail_mult=TRAIL_ATR_MULT,
        fee_bps=FEE_BPS,
        limit=BACKTEST_LIMIT
    )
    text = (
        f"📊 Backtest RR ({res['symbol']})\n"
        f"Trades: {res['trades']} | Wins: {res['wins']} | Losses: {res['losses']}\n"
        f"Winrate: {res['winrate_pct']}% | PF: {res['profit_factor']}\n"
        f"Total R: {res['total_R']} | Avg R: {res['avg_R']}\n"
        f"Params: {res['params']}"
    )
    await update.message.reply_text(text)

async def auto_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Check and push signals for all symbols periodically."""
    try:
        if STATE["auto"]:
            for sym in STATE["symbols"]:
                if sym not in AUTO_JOB_STATE:
                    AUTO_JOB_STATE[sym] = None
                df = build_signals(sym, use_ha_in_ut=STATE["use_ha_in_ut"])
                row = last_closed_row(df)
                ct = row["close_time"]
                if AUTO_JOB_STATE[sym] is None or ct != AUTO_JOB_STATE[sym]:
                    AUTO_JOB_STATE[sym] = ct
                    signal_text = build_signal_text(sym, row)
                    if len(STATE["symbols"]) > 1:
                        signal_text = f"🔄 Auto Monitor\n{signal_text}"
                    await send_msg(context.application, signal_text)
                    if len(STATE["symbols"]) > 1:
                        await asyncio.sleep(1)
        # TODO (optional): nếu ENABLE_TRADING và có STATE["pos"], cập nhật trailing stop SL ở đây
    except Exception as e:
        log.exception(e)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("symbol", cmd_symbol))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("use_ha", cmd_use_ha))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("signals_all", cmd_signals_all))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("backtest_all", cmd_backtest_all))
    app.add_handler(CommandHandler("backtest_rr", cmd_backtest_rr))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    # Schedule the auto check job to run every 15 seconds
    app.job_queue.run_repeating(auto_check_job, interval=15, first=10)

    log.info("Bot started.")
    app.run_polling()

if __name__ == "__main__":
    main()
