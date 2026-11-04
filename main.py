#!/usr/bin/env python3
"""
================================================================================
XAUUSD M15 Quantitative Signal Bot
================================================================================
Core Strategy Implementation:
1. 100 & 200 Simple Moving Average (SMA) Regime Filter & MA Stack/Pinch Detection
2. Wilder's RMA ATR(10) & Recursive Supertrend (10, 3.0) with explicit (+1/-1) polarity
3. London/NY Session Filter: Entries restricted strictly between 07:00 and 17:00 UTC
4. Zero-Lookahead Bias Protection: Evaluates latest closed M15 candle (index -2)
5. Continuous Live Polling Loop with real-time terminal telemetry and alerts
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==============================================================================
# Telegram Bot Configuration & Fallbacks
# ==============================================================================

DEFAULT_TELEGRAM_BOT_TOKEN = "9277271042:AAFAsehhFAFZuBdCe8WgSJlbxMILiR0_czY"
DEFAULT_TELEGRAM_CHAT_ID = "113941575"


def get_telegram_credentials() -> Tuple[str, str]:
    """Retrieves Telegram credentials with environment variable fallbacks."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", DEFAULT_TELEGRAM_BOT_TOKEN).strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", DEFAULT_TELEGRAM_CHAT_ID).strip()
    return token, chat_id


def send_telegram_message(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout_sec: int = 10,
) -> bool:
    """
    Sends a formatted message to a Telegram chat using Telegram Bot API.
    
    Args:
        text: The message content (formatted with Markdown).
        bot_token: Optional token override. Falls back to env / default.
        chat_id: Optional chat id override. Falls back to env / default.
        timeout_sec: HTTP request timeout in seconds.
        
    Returns:
        True if the message was accepted by Telegram (HTTP 200 and ok=True), False otherwise.
    """
    env_token, env_chat_id = get_telegram_credentials()
    active_token = bot_token or env_token
    active_chat_id = chat_id or env_chat_id

    if not active_token or not active_chat_id:
        logging.error("[TELEGRAM] Missing bot token or chat ID. Cannot dispatch alert.")
        return False

    url = f"https://api.telegram.org/bot{active_token}/sendMessage"
    payload = {
        "chat_id": active_chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout_sec)
        resp_data = response.json()
        if response.status_code == 200 and resp_data.get("ok"):
            logging.info("[TELEGRAM] Message dispatched successfully to chat %s.", active_chat_id)
            return True
        else:
            logging.error(
                "[TELEGRAM] API error %d: %s",
                response.status_code,
                resp_data.get("description", response.text),
            )
            return False
    except requests.exceptions.RequestException as e:
        logging.error("[TELEGRAM] Network/HTTP request failed: %s", e)
        return False


def format_telegram_signal(sig: StrategySignal) -> str:
    """
    Formats an outgoing trading signal for Telegram with Markdown bolding,
    entry price, Supertrend initial stop ($1R$), and confirmed candle timestamp.
    """
    is_buy = sig.signal_type in (SignalType.TRIGGER_A_BUY, SignalType.TRIGGER_B_BUY)
    action_str = "*BUY / LONG* 🟢" if is_buy else "*SELL / SHORT* 🔴"
    trigger_title = sig.signal_type.value.replace("_", " ")

    risk_pts = abs(sig.entry_price - sig.stop_loss)
    tp1_pts = abs(sig.take_profit_1 - sig.entry_price)
    tp2_pts = abs(sig.take_profit_2 - sig.entry_price)
    ts_str = sig.bar_timestamp_utc.strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"🚨 *XAUUSD M15 QUANTITATIVE SIGNAL* 🚨",
        "",
        f"• *Action:* {action_str}",
        f"• *Signal Trigger:* *{trigger_title}*",
        f"• *Asset / Symbol:* *{sig.symbol}* (Gold)",
        f"• *Candle Closed (UTC):* *{ts_str}*",
        "",
        f"📊 *Execution Levels:*",
        f"• *Entry Price:* *{sig.entry_price:.2f}*",
        f"• *Supertrend Initial Stop ($1R$):* *{sig.stop_loss:.2f}* ({risk_pts:.2f} pts)",
        f"• *Take Profit 1 ($1.5R$):* *{sig.take_profit_1:.2f}* (+{tp1_pts:.2f} pts)",
        f"• *Take Profit 2 ($2.5R$):* *{sig.take_profit_2:.2f}* (+{tp2_pts:.2f} pts)",
        f"• *Wilder ATR(10):* *{sig.atr:.2f}*",
        "",
        f"📝 *Regime & Filter Verification:*",
        f"{sig.notes}",
    ]
    return "\n".join(lines)


# ==============================================================================
# Configuration & Domain Models
# ==============================================================================

class Polarity(int, Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


class SignalType(str, Enum):
    TRIGGER_A_BUY = "TRIGGER_A_BUY"      # Supertrend Bullish Flip in Bull Regime
    TRIGGER_A_SELL = "TRIGGER_A_SELL"    # Supertrend Bearish Flip in Bear Regime
    TRIGGER_B_BUY = "TRIGGER_B_BUY"      # Pullback & Bounce off 100 SMA in Bull Trend
    TRIGGER_B_SELL = "TRIGGER_B_SELL"    # Pullback & Rejection off 100 SMA in Bear Trend
    NO_SIGNAL = "NO_SIGNAL"


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy configuration and hyperparameters."""
    symbol: str = "GC=F"                  # Gold futures (XAUUSD liquid proxy on Yahoo Finance)
    interval: str = "15m"                 # 15-Minute timeframe
    history_period: str = "60d"           # Sufficient lookback for 200 SMA & RMA warmup
    sma_fast_len: int = 100               # 100 SMA
    sma_slow_len: int = 200               # 200 SMA
    atr_len: int = 10                     # Wilder's RMA ATR period
    st_mult: float = 3.0                  # Supertrend multiplier
    pinch_atr_mult: float = 0.5           # Pinch threshold: MA Gap < 0.5 * ATR indicates squeeze/chop
    min_gap_pts: float = 0.5              # Minimum absolute dollar gap between 100 and 200 SMA
    pullback_tolerance_atr: float = 0.5   # Pullback zone tolerance in units of ATR
    session_start_utc: datetime.time = datetime.time(7, 0, 0)   # London Open (07:00 UTC)
    session_end_utc: datetime.time = datetime.time(17, 0, 0)    # NY Afternoon (17:00 UTC)
    poll_interval_sec: int = 60           # Polling loop cadence in seconds


@dataclass
class BarAnalysis:
    """State analysis of an individual market candle."""
    timestamp_utc: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_100: float
    sma_200: float
    sma_gap: float
    atr: float
    supertrend: float
    st_polarity: Polarity
    is_in_killzone: bool
    is_bull_regime: bool
    is_bear_regime: bool
    is_bull_stack: bool
    is_bear_stack: bool
    is_pinched: bool


@dataclass
class StrategySignal:
    """Actionable trade signal package."""
    signal_type: SignalType
    symbol: str
    bar_timestamp_utc: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    atr: float
    notes: str


# ==============================================================================
# Technical Indicator Core (Pure Vectorized / Recursive Quant Algorithms)
# ==============================================================================

def calculate_wilder_rma(series: np.ndarray, period: int) -> np.ndarray:
    """
    Computes Wilder's Running Moving Average (RMA) identical to TradingView / Pine Script `ta.rma`.
    Formula:
        RMA[0] = SMA(series[:period])
        RMA[t] = (RMA[t-1] * (period - 1) + series[t]) / period
    """
    n = len(series)
    rma = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return rma

    # Seed with initial arithmetic mean
    rma[period - 1] = np.mean(series[:period])
    alpha_inv = float(period)

    for i in range(period, n):
        rma[i] = (rma[i - 1] * (alpha_inv - 1.0) + series[i]) / alpha_inv

    return rma


def calculate_atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Computes Wilder's Average True Range ATR(10) with True Range calculation:
        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    """
    n = len(high)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    return calculate_wilder_rma(tr, period)


def calculate_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Recursive Supertrend (period, multiplier) with explicit +1 / -1 polarity.
    Matches standard Pine Script / TradingView definition:
        hl2 = (high + low) / 2
        basic_upper = hl2 + multiplier * ATR
        basic_lower = hl2 - multiplier * ATR
    
    Returns:
        (atr, supertrend_line, polarity)
        polarity is +1 for Bullish (price above Supertrend), -1 for Bearish (price below Supertrend)
    """
    n = len(high)
    atr = calculate_atr_wilder(high, low, close, period=period)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    supertrend = np.full(n, np.nan, dtype=np.float64)
    polarity = np.zeros(n, dtype=np.int32)

    start_idx = period - 1
    if start_idx >= n:
        return atr, supertrend, polarity

    # Initialize bands at start_idx
    final_upper[start_idx] = basic_upper[start_idx]
    final_lower[start_idx] = basic_lower[start_idx]
    
    if close[start_idx] >= basic_lower[start_idx]:
        polarity[start_idx] = 1
        supertrend[start_idx] = final_lower[start_idx]
    else:
        polarity[start_idx] = -1
        supertrend[start_idx] = final_upper[start_idx]

    # Recursive loop across time
    for i in range(start_idx + 1, n):
        # Final Lower Band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Final Upper Band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Polarity & Supertrend Trailing Stop Level
        if polarity[i - 1] == 1:
            if close[i] < final_lower[i]:
                polarity[i] = -1
                supertrend[i] = final_upper[i]
            else:
                polarity[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                polarity[i] = 1
                supertrend[i] = final_lower[i]
            else:
                polarity[i] = -1
                supertrend[i] = final_upper[i]

    return atr, supertrend, polarity


def compute_strategy_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Computes all moving averages, gap/pinch filters, ATR, and Supertrend on the dataset.
    Ensures zero lookahead by working strictly row-by-row on historical series.
    """
    df = df.copy()

    # Moving Averages
    df["SMA_100"] = df["Close"].rolling(window=config.sma_fast_len, min_periods=config.sma_fast_len).mean()
    df["SMA_200"] = df["Close"].rolling(window=config.sma_slow_len, min_periods=config.sma_slow_len).mean()

    # MA Gap & Stack
    df["SMA_Gap"] = df["SMA_100"] - df["SMA_200"]
    df["SMA_Gap_Pct"] = (df["SMA_Gap"] / df["SMA_200"]) * 100.0

    # ATR and Recursive Supertrend
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    atr, st_line, polarity = calculate_supertrend(
        high, low, close, period=config.atr_len, multiplier=config.st_mult
    )

    df["ATR"] = atr
    df["Supertrend"] = st_line
    df["ST_Polarity"] = polarity

    # Pinch filter: MAs converging within pinch threshold
    pinch_band = np.maximum(df["ATR"] * config.pinch_atr_mult, config.min_gap_pts)
    df["Is_Pinched"] = df["SMA_Gap"].abs() < pinch_band

    # MA Stack Definitions
    df["Bull_Stack"] = (df["Close"] > df["SMA_100"]) & (df["SMA_100"] > df["SMA_200"])
    df["Bear_Stack"] = (df["Close"] < df["SMA_100"]) & (df["SMA_100"] < df["SMA_200"])

    # Macro Regime (100 & 200 SMA filter)
    df["Bull_Regime"] = (df["Close"] > df["SMA_200"]) & (df["SMA_100"] > df["SMA_200"])
    df["Bear_Regime"] = (df["Close"] < df["SMA_200"]) & (df["SMA_100"] < df["SMA_200"])

    return df


# ==============================================================================
# Session & Killzone Filter (07:00 - 17:00 UTC)
# ==============================================================================

def is_in_session_killzone(
    timestamp_utc: pd.Timestamp,
    start_time: datetime.time = datetime.time(7, 0, 0),
    end_time: datetime.time = datetime.time(17, 0, 0),
) -> bool:
    """
    Checks if a candle's UTC timestamp falls strictly within the London/NY killzone (07:00 - 17:00 UTC).
    Weekend candles (Saturday/Sunday) are strictly excluded.
    """
    # Exclude weekends (5 = Saturday, 6 = Sunday)
    if timestamp_utc.weekday() >= 5:
        return False

    t = timestamp_utc.time()
    return start_time <= t <= end_time


# ==============================================================================
# Zero-Lookahead Signal Engine
# ==============================================================================

def evaluate_signals_zero_lookahead(
    df: pd.DataFrame, config: StrategyConfig
) -> Tuple[StrategySignal, BarAnalysis, BarAnalysis]:
    """
    Evaluates trading signals using STRICT Zero-Lookahead logic.
    - Index -1 is the live, in-progress candle (unconfirmed, ignored for triggers).
    - Index -2 is the latest fully closed and confirmed M15 candle.
    - Index -3 is the prior closed M15 candle (used to detect transitions/flips).
    
    Returns:
        (StrategySignal, current_closed_analysis, prior_closed_analysis)
    """
    if len(df) < max(config.sma_slow_len + 10, 205):
        raise ValueError(
            f"Insufficient historical bars ({len(df)}) for reliable 200 SMA and RMA warmup. Need >= 205."
        )

    # Index -2: Latest Closed Candle (Confirmed)
    curr = df.iloc[-2]
    curr_ts = df.index[-2]

    # Index -3: Previous Closed Candle (For transitions)
    prev = df.iloc[-3]
    prev_ts = df.index[-3]

    curr_polarity = Polarity(int(curr["ST_Polarity"]))
    prev_polarity = Polarity(int(prev["ST_Polarity"]))

    in_killzone = is_in_session_killzone(curr_ts, config.session_start_utc, config.session_end_utc)
    curr_pinched = bool(curr["Is_Pinched"])
    curr_bull_regime = bool(curr["Bull_Regime"])
    curr_bear_regime = bool(curr["Bear_Regime"])
    curr_bull_stack = bool(curr["Bull_Stack"])
    curr_bear_stack = bool(curr["Bear_Stack"])

    curr_analysis = BarAnalysis(
        timestamp_utc=curr_ts,
        open=float(curr["Open"]),
        high=float(curr["High"]),
        low=float(curr["Low"]),
        close=float(curr["Close"]),
        volume=float(curr["Volume"]),
        sma_100=float(curr["SMA_100"]),
        sma_200=float(curr["SMA_200"]),
        sma_gap=float(curr["SMA_Gap"]),
        atr=float(curr["ATR"]),
        supertrend=float(curr["Supertrend"]),
        st_polarity=curr_polarity,
        is_in_killzone=in_killzone,
        is_bull_regime=curr_bull_regime,
        is_bear_regime=curr_bear_regime,
        is_bull_stack=curr_bull_stack,
        is_bear_stack=curr_bear_stack,
        is_pinched=curr_pinched,
    )

    prev_analysis = BarAnalysis(
        timestamp_utc=prev_ts,
        open=float(prev["Open"]),
        high=float(prev["High"]),
        low=float(prev["Low"]),
        close=float(prev["Close"]),
        volume=float(prev["Volume"]),
        sma_100=float(prev["SMA_100"]),
        sma_200=float(prev["SMA_200"]),
        sma_gap=float(prev["SMA_Gap"]),
        atr=float(prev["ATR"]),
        supertrend=float(prev["Supertrend"]),
        st_polarity=prev_polarity,
        is_in_killzone=is_in_session_killzone(prev_ts, config.session_start_utc, config.session_end_utc),
        is_bull_regime=bool(prev["Bull_Regime"]),
        is_bear_regime=bool(prev["Bear_Regime"]),
        is_bull_stack=bool(prev["Bull_Stack"]),
        is_bear_stack=bool(prev["Bear_Stack"]),
        is_pinched=bool(prev["Is_Pinched"]),
    )

    entry_price = float(curr["Close"])
    atr_val = float(curr["ATR"])

    # Default: No Signal
    signal_type = SignalType.NO_SIGNAL
    sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    notes = "Market condition does not meet entry triggers."

    # --------------------------------------------------------------------------
    # Gate 1: Session Filter (07:00 - 17:00 UTC)
    # --------------------------------------------------------------------------
    if not in_killzone:
        notes = f"Filtered: Closed candle ({curr_ts.strftime('%H:%M')} UTC) is outside London/NY Session [07:00-17:00 UTC]."
        return (
            StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
            curr_analysis,
            prev_analysis,
        )

    # --------------------------------------------------------------------------
    # Gate 2: Pinch & MA Gap Filter (Squeeze Protection)
    # --------------------------------------------------------------------------
    if curr_pinched:
        notes = f"Filtered: MA Stack is Pinched (Gap: {curr_analysis.sma_gap:.2f} < threshold). Avoiding low-volatility chop."
        return (
            StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
            curr_analysis,
            prev_analysis,
        )

    # --------------------------------------------------------------------------
    # TRIGGER A: Supertrend Polarity Flip Breakout
    # Long: ST flips from -1 to +1 while in Bull Regime (Close > 200 SMA and 100 SMA > 200 SMA)
    # Short: ST flips from +1 to -1 while in Bear Regime (Close < 200 SMA and 100 SMA < 200 SMA)
    # --------------------------------------------------------------------------
    st_flipped_bullish = (prev_polarity == Polarity.BEARISH) and (curr_polarity == Polarity.BULLISH)
    st_flipped_bearish = (prev_polarity == Polarity.BULLISH) and (curr_polarity == Polarity.BEARISH)

    if st_flipped_bullish and curr_bull_regime:
        signal_type = SignalType.TRIGGER_A_BUY
        sl = float(curr["Supertrend"])
        risk = max(entry_price - sl, 0.5 * atr_val)
        tp1 = entry_price + 1.5 * risk
        tp2 = entry_price + 2.5 * risk
        notes = "Trigger A (Bullish Supertrend Flip in 100/200 SMA Bull Regime)."
        return (
            StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
            curr_analysis,
            prev_analysis,
        )

    if st_flipped_bearish and curr_bear_regime:
        signal_type = SignalType.TRIGGER_A_SELL
        sl = float(curr["Supertrend"])
        risk = max(sl - entry_price, 0.5 * atr_val)
        tp1 = entry_price - 1.5 * risk
        tp2 = entry_price - 2.5 * risk
        notes = "Trigger A (Bearish Supertrend Flip in 100/200 SMA Bear Regime)."
        return (
            StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
            curr_analysis,
            prev_analysis,
        )

    # --------------------------------------------------------------------------
    # TRIGGER B: Trend Pullback & Continuation / 100 SMA Bounce
    # Long: Supertrend already Bullish (+1), Price pulled back into 100 SMA zone and closed bullish
    # Short: Supertrend already Bearish (-1), Price pulled back into 100 SMA zone and closed bearish
    # --------------------------------------------------------------------------
    tol = config.pullback_tolerance_atr * atr_val

    # Bullish Pullback:
    # 1. Macro Regime is Bullish & Supertrend is confirmed Bullish
    # 2. Candle low touched near or below 100 SMA
    # 3. Candle closed back above 100 SMA with a bullish body (Close > Open)
    if curr_bull_regime and curr_polarity == Polarity.BULLISH and prev_polarity == Polarity.BULLISH:
        tested_100_sma = curr["Low"] <= (curr["SMA_100"] + tol)
        closed_above_100_sma = curr["Close"] > curr["SMA_100"]
        bullish_candle = curr["Close"] > curr["Open"]

        if tested_100_sma and closed_above_100_sma and bullish_candle:
            signal_type = SignalType.TRIGGER_B_BUY
            sl = min(float(curr["Low"]) - 0.5 * atr_val, float(curr["Supertrend"]))
            risk = max(entry_price - sl, 0.5 * atr_val)
            tp1 = entry_price + 1.5 * risk
            tp2 = entry_price + 2.5 * risk
            notes = "Trigger B (Bullish 100 SMA Pullback & Bounce with Supertrend Alignment)."
            return (
                StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
                curr_analysis,
                prev_analysis,
            )

    # Bearish Pullback:
    # 1. Macro Regime is Bearish & Supertrend is confirmed Bearish
    # 2. Candle high touched near or above 100 SMA
    # 3. Candle closed back below 100 SMA with a bearish body (Close < Open)
    if curr_bear_regime and curr_polarity == Polarity.BEARISH and prev_polarity == Polarity.BEARISH:
        tested_100_sma = curr["High"] >= (curr["SMA_100"] - tol)
        closed_below_100_sma = curr["Close"] < curr["SMA_100"]
        bearish_candle = curr["Close"] < curr["Open"]

        if tested_100_sma and closed_below_100_sma and bearish_candle:
            signal_type = SignalType.TRIGGER_B_SELL
            sl = max(float(curr["High"]) + 0.5 * atr_val, float(curr["Supertrend"]))
            risk = max(sl - entry_price, 0.5 * atr_val)
            tp1 = entry_price - 1.5 * risk
            tp2 = entry_price - 2.5 * risk
            notes = "Trigger B (Bearish 100 SMA Pullback & Rejection with Supertrend Alignment)."
            return (
                StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
                curr_analysis,
                prev_analysis,
            )

    # Fallback diagnostics
    if curr_bull_regime and curr_polarity == Polarity.BULLISH:
        notes = "Bullish trend active (ST +1, MA Stack Bull), waiting for Trigger A flip or Trigger B 100 SMA test."
    elif curr_bear_regime and curr_polarity == Polarity.BEARISH:
        notes = "Bearish trend active (ST -1, MA Stack Bear), waiting for Trigger A flip or Trigger B 100 SMA test."
    else:
        notes = f"Regime/Supertrend Divergence (ST Polarity: {curr_polarity.name}, 100/200 SMA: {'Bull' if curr_bull_regime else 'Bear' if curr_bear_regime else 'Neutral'})."

    return (
        StrategySignal(signal_type, config.symbol, curr_ts, entry_price, sl, tp1, tp2, atr_val, notes),
        curr_analysis,
        prev_analysis,
    )


# ==============================================================================
# Market Data Fetcher
# ==============================================================================

def fetch_m15_market_data(config: StrategyConfig) -> pd.DataFrame:
    """
    Downloads historical M15 data from Yahoo Finance and standardizes timezone to UTC.
    Uses GC=F (Gold Futures) as high-liquidity proxy for XAUUSD.
    """
    ticker = yf.Ticker(config.symbol)
    df = ticker.history(period=config.history_period, interval=config.interval)

    if df.empty:
        raise ValueError(f"No market data returned for symbol '{config.symbol}' on interval '{config.interval}'.")

    # Clean columns and index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize Datetime Index to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ==============================================================================
# Production Telemetry & Terminal UI Renderer
# ==============================================================================

class TerminalColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def render_terminal_dashboard(
    signal: StrategySignal,
    curr: BarAnalysis,
    forming_bar_ts: pd.Timestamp,
    config: StrategyConfig,
) -> None:
    """Renders a clean, institutional-grade status telemetry dashboard."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    c = TerminalColors

    session_str = (
        f"{c.GREEN}ACTIVE [07:00–17:00 UTC]{c.RESET}"
        if curr.is_in_killzone
        else f"{c.YELLOW}INACTIVE (Outside Killzone){c.RESET}"
    )

    regime_str = (
        f"{c.GREEN}BULLISH REGIME (Close > 200 SMA & 100 > 200){c.RESET}"
        if curr.is_bull_regime
        else f"{c.RED}BEARISH REGIME (Close < 200 SMA & 100 < 200){c.RESET}"
        if curr.is_bear_regime
        else f"{c.YELLOW}NEUTRAL / TRANSITIONAL REGIME{c.RESET}"
    )

    polarity_str = (
        f"{c.GREEN}+1 (BULLISH){c.RESET}"
        if curr.st_polarity == Polarity.BULLISH
        else f"{c.RED}-1 (BEARISH){c.RESET}"
        if curr.st_polarity == Polarity.BEARISH
        else "0 (NEUTRAL)"
    )

    pinch_str = (
        f"{c.RED}PINCHED (Gap: {curr.sma_gap:.2f} < Squeeze Band){c.RESET}"
        if curr.is_pinched
        else f"{c.GREEN}OPEN / EXPANDED (Gap: {curr.sma_gap:.2f} pts){c.RESET}"
    )

    print(f"\n{c.BOLD}{'=' * 80}{c.RESET}")
    print(f"{c.BOLD}{c.CYAN} XAUUSD (Gold) M15 QUANTITATIVE TRADING BOT | ZERO-LOOKAHEAD ENGINE{c.RESET}")
    print(f"{c.BOLD}{'=' * 80}{c.RESET}")
    print(f" System Clock (UTC)    : {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" London/NY Killzone    : {session_str}")
    print(f" Symbol & Timeframe    : {config.symbol} | {config.interval}")
    print(f" Forming Bar (Index -1): {forming_bar_ts.strftime('%Y-%m-%d %H:%M UTC')} [UNCONFIRMED - NOT EVALUATED]")
    print(f" Evaluated Bar (Idx -2): {c.BOLD}{curr.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}{c.RESET} [CONFIRMED CLOSED]")
    print(f"-" * 80)
    print(f" Bar OHLC              : Open: {curr.open:.2f} | High: {curr.high:.2f} | Low: {curr.low:.2f} | Close: {c.BOLD}{curr.close:.2f}{c.RESET}")
    print(f" 100 SMA / 200 SMA     : 100 SMA: {curr.sma_100:.2f} | 200 SMA: {curr.sma_200:.2f}")
    print(f" MA Regime             : {regime_str}")
    print(f" MA Gap / Pinch State  : {pinch_str}")
    print(f" Wilder RMA ATR(10)    : {curr.atr:.2f}")
    print(f" Supertrend (10, 3.0)  : Level: {curr.supertrend:.2f} | Polarity: {polarity_str}")
    print(f"-" * 80)

    # Signal Banner
    if signal.signal_type in (SignalType.TRIGGER_A_BUY, SignalType.TRIGGER_B_BUY):
        print(f"{c.BOLD}{c.GREEN} >>> [ALERT] {signal.signal_type.value} SIGNAL GENERATED <<<{c.RESET}")
        print(f"  Action      : {c.GREEN}{c.BOLD}BUY / LONG{c.RESET}")
        print(f"  Entry Price : {signal.entry_price:.2f}")
        print(f"  Stop Loss   : {signal.stop_loss:.2f} (-{signal.entry_price - signal.stop_loss:.2f} pts)")
        print(f"  TP1 (1.5 R) : {signal.take_profit_1:.2f} (+{signal.take_profit_1 - signal.entry_price:.2f} pts)")
        print(f"  TP2 (2.5 R) : {signal.take_profit_2:.2f} (+{signal.take_profit_2 - signal.entry_price:.2f} pts)")
        print(f"  Details     : {signal.notes}")
    elif signal.signal_type in (SignalType.TRIGGER_A_SELL, SignalType.TRIGGER_B_SELL):
        print(f"{c.BOLD}{c.RED} >>> [ALERT] {signal.signal_type.value} SIGNAL GENERATED <<<{c.RESET}")
        print(f"  Action      : {c.RED}{c.BOLD}SELL / SHORT{c.RESET}")
        print(f"  Entry Price : {signal.entry_price:.2f}")
        print(f"  Stop Loss   : {signal.stop_loss:.2f} (+{signal.stop_loss - signal.entry_price:.2f} pts)")
        print(f"  TP1 (1.5 R) : {signal.take_profit_1:.2f} (-{signal.entry_price - signal.take_profit_1:.2f} pts)")
        print(f"  TP2 (2.5 R) : {signal.take_profit_2:.2f} (-{signal.entry_price - signal.take_profit_2:.2f} pts)")
        print(f"  Details     : {signal.notes}")
    else:
        print(f"{c.YELLOW} [STATUS: NO ACTIVE SIGNAL]{c.RESET}")
        print(f" Reason / Context: {signal.notes}")

    print(f"{c.BOLD}{'=' * 80}{c.RESET}")


# ==============================================================================
# Continuous Polling Loop
# ==============================================================================

class TradingBot:
    """Production automated trading bot coordinator."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.is_running = True
        self.last_evaluated_candle_ts: Optional[pd.Timestamp] = None
        self._setup_signals()

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame) -> None:
        print("\n[INFO] Graceful shutdown requested. Stopping trading bot...")
        self.is_running = False

    def scan_once(self) -> Tuple[StrategySignal, BarAnalysis]:
        """Executes a single market scan cycle."""
        # 1. Download live market data
        df_raw = fetch_m15_market_data(self.config)

        # 2. Compute full indicator stack
        df_analyzed = compute_strategy_indicators(df_raw, self.config)

        # 3. Form unconfirmed bar reference (index -1)
        forming_bar_ts = df_analyzed.index[-1]

        # 4. Strict Zero-Lookahead Signal Evaluation on confirmed closed bar (index -2)
        sig, curr_analysis, _ = evaluate_signals_zero_lookahead(df_analyzed, self.config)

        # 5. Output telemetry
        render_terminal_dashboard(sig, curr_analysis, forming_bar_ts, self.config)

        # 6. Deduplicated alert dispatch
        if sig.signal_type != SignalType.NO_SIGNAL:
            if self.last_evaluated_candle_ts != sig.bar_timestamp_utc:
                self.dispatch_alert(sig)
                self.last_evaluated_candle_ts = sig.bar_timestamp_utc

        return sig, curr_analysis

    def dispatch_alert(self, sig: StrategySignal) -> None:
        """
        Dispatches signals to downstream consumers and broadcasts to Telegram.
        """
        logging.info(
            "*** NEW SIGNAL DISPATCHED: %s | %s @ %.2f | SL: %.2f | TP1: %.2f ***",
            sig.signal_type.value,
            sig.symbol,
            sig.entry_price,
            sig.stop_loss,
            sig.take_profit_1,
        )
        # Broadcast formatted alert to Telegram
        telegram_msg = format_telegram_signal(sig)
        send_telegram_message(telegram_msg)

    def send_startup_notification(self) -> bool:
        """Dispatches an initialization startup notification to Telegram."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        token, chat_id = get_telegram_credentials()
        # Obfuscate token for safe logging
        masked_token = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else "***"

        startup_msg = (
            f"🚀 *XAUUSD M15 BOT INITIALIZED & ONLINE* 🚀\n\n"
            f"• *Symbol:* `{self.config.symbol}` (Gold M15)\n"
            f"• *Regime Filter:* 100 & 200 SMA\n"
            f"• *Supertrend:* (10, 3.0) with Wilder RMA ATR\n"
            f"• *Session Killzone:* 07:00 – 17:00 UTC (London / NY)\n"
            f"• *Polling Cadence:* Every {self.config.poll_interval_sec}s\n"
            f"• *Zero-Lookahead:* Monitoring confirmed closed bars (Index -2)\n"
            f"• *Initialization Time:* `{now_utc}`\n"
            f"• *Engine Status:* 🟢 *ONLINE & ACTIVE*"
        )

        print(f"[INFO] Dispatching startup notification to Telegram chat {chat_id} (Token: {masked_token})...")
        ok = send_telegram_message(startup_msg)
        if ok:
            print(">>> ✅ Telegram startup notification successfully delivered! <<<\n")
        else:
            print(">>> ⚠️ Telegram startup notification delivery failed (see logs). <<<\n")
        return ok

    def run_loop(self) -> None:
        """Runs the continuous polling execution loop."""
        print(f"[INFO] Initializing continuous polling loop for {self.config.symbol} (interval: {self.config.poll_interval_sec}s)...")
        print("[INFO] Press Ctrl+C to terminate cleanly.\n")

        # Send initial startup notification to Telegram
        self.send_startup_notification()

        consecutive_errors = 0
        while self.is_running:
            try:
                self.scan_once()
                consecutive_errors = 0
            except KeyboardInterrupt:
                break
            except Exception as e:
                consecutive_errors += 1
                logging.error("Error during scan cycle: %s", e, exc_info=True)
                backoff = min(60, 5 * consecutive_errors)
                print(f"[WARN] Retrying after error in {backoff} seconds...")
                time.sleep(backoff)
                continue

            # Sleep in small increments for responsive SIGINT handling
            for _ in range(self.config.poll_interval_sec):
                if not self.is_running:
                    break
                time.sleep(1)

        print("[INFO] Bot execution loop exited cleanly.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XAUUSD M15 Automated Trading Signal Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="GC=F",
        help="Market symbol ticker on Yahoo Finance (GC=F is Gold Futures).",
    )
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=60,
        help="Polling loop cycle sleep interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan evaluation and exit immediately.",
    )
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Send a test trade alert to Telegram using the configured credentials and exit.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    if args.test_telegram:
        print("\n[INFO] Triggering Telegram test signal alert...")
        test_sig = StrategySignal(
            signal_type=SignalType.TRIGGER_A_BUY,
            symbol=args.symbol,
            bar_timestamp_utc=pd.Timestamp.now(tz="UTC"),
            entry_price=3945.50,
            stop_loss=3932.00,
            take_profit_1=3965.75,
            take_profit_2=3979.25,
            atr=6.20,
            notes="Test Signal: Supertrend (10, 3.0) Bullish Flip in 100/200 SMA Bull Regime (London/NY Session).",
        )
        msg = format_telegram_signal(test_sig)
        print("-" * 60)
        print("Message payload to send:")
        print(msg)
        print("-" * 60)
        ok = send_telegram_message(msg)
        if ok:
            print(">>> ✅ Telegram message successfully delivered to chat! <<<")
        else:
            print(">>> ❌ Telegram message delivery failed. See logs above. <<<")
        return

    config = StrategyConfig(
        symbol=args.symbol,
        poll_interval_sec=args.poll_sec,
    )

    bot = TradingBot(config)

    if args.once:
        bot.scan_once()
    else:
        bot.run_loop()


if __name__ == "__main__":
    main()
