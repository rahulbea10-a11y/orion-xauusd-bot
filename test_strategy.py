"""
Unit tests for XAUUSD M15 Core Trading Strategy.
Validates:
1. Wilder's RMA ATR(10) calculations
2. Recursive Supertrend (10, 3.0) +1 Bull / -1 Bear polarity
3. London/NY Session filter boundaries (07:00 - 17:00 UTC)
4. Rolling 100 & 200 SMA and Gap / Pinch detection
5. Zero-lookahead evaluation on closed bar index -2
6. Trigger A (Supertrend Flip) and Trigger B (Pullback Continuation) logic
"""

import datetime
import unittest
import numpy as np
import pandas as pd

from main import (
    DEFAULT_TELEGRAM_BOT_TOKEN,
    DEFAULT_TELEGRAM_CHAT_ID,
    Polarity,
    SignalType,
    StrategyConfig,
    StrategySignal,
    calculate_atr_wilder,
    calculate_supertrend,
    calculate_wilder_rma,
    compute_strategy_indicators,
    evaluate_signals_zero_lookahead,
    format_telegram_signal,
    get_telegram_credentials,
    is_in_session_killzone,
)


class TestIndicators(unittest.TestCase):

    def test_wilder_rma(self):
        values = np.array([10.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float64)
        rma = calculate_wilder_rma(values, period=3)
        self.assertTrue(np.isnan(rma[0]))
        self.assertTrue(np.isnan(rma[1]))
        # First valid is SMA of first 3 items: (10 + 12 + 14)/3 = 12.0
        self.assertAlmostEqual(rma[2], 12.0)
        # Next: (12 * 2 + 16) / 3 = 40 / 3 = 13.3333...
        self.assertAlmostEqual(rma[3], 40.0 / 3.0)
        # Next: (13.3333 * 2 + 18) / 3 = 44.6666 / 3 = 14.8888...
        self.assertAlmostEqual(rma[4], (40.0 / 3.0 * 2.0 + 18.0) / 3.0)

    def test_atr_wilder(self):
        high = np.array([10.0, 12.0, 11.0, 13.0, 15.0], dtype=np.float64)
        low = np.array([9.0, 10.0, 9.5, 11.0, 13.5], dtype=np.float64)
        close = np.array([9.5, 11.0, 10.0, 12.5, 14.0], dtype=np.float64)

        atr = calculate_atr_wilder(high, low, close, period=3)
        self.assertEqual(len(atr), 5)
        self.assertTrue(np.isnan(atr[0]))
        self.assertTrue(np.isnan(atr[1]))
        self.assertFalse(np.isnan(atr[2]))
        self.assertGreater(atr[2], 0.0)

    def test_supertrend_polarity_flip(self):
        # Construct synthetic series where price starts rising then collapses then rebounds
        n = 30
        high = np.zeros(n)
        low = np.zeros(n)
        close = np.zeros(n)

        # Bull phase
        for i in range(15):
            high[i] = 100.0 + i * 2.0 + 1.0
            low[i] = 100.0 + i * 2.0 - 1.0
            close[i] = 100.0 + i * 2.0

        # Bear dump
        for i in range(15, 30):
            high[i] = 130.0 - (i - 14) * 3.0 + 1.0
            low[i] = 130.0 - (i - 14) * 3.0 - 1.0
            close[i] = 130.0 - (i - 14) * 3.0

        atr, st, polarity = calculate_supertrend(high, low, close, period=5, multiplier=2.0)
        self.assertEqual(len(polarity), n)

        # In rising phase, polarity must be +1
        self.assertEqual(polarity[14], 1)
        # In dump phase, polarity must flip to -1
        self.assertEqual(polarity[-1], -1)


class TestSessionFilter(unittest.TestCase):

    def test_killzone_boundaries(self):
        # 07:00 UTC Wednesday -> True
        ts1 = pd.Timestamp("2026-11-04 07:00:00", tz="UTC")
        self.assertTrue(is_in_session_killzone(ts1))

        # 12:30 UTC Wednesday -> True
        ts2 = pd.Timestamp("2026-11-04 12:30:00", tz="UTC")
        self.assertTrue(is_in_session_killzone(ts2))

        # 17:00 UTC Wednesday -> True
        ts3 = pd.Timestamp("2026-11-04 17:00:00", tz="UTC")
        self.assertTrue(is_in_session_killzone(ts3))

        # 06:45 UTC Wednesday -> False
        ts4 = pd.Timestamp("2026-11-04 06:45:00", tz="UTC")
        self.assertFalse(is_in_session_killzone(ts4))

        # 17:15 UTC Wednesday -> False
        ts5 = pd.Timestamp("2026-11-04 17:15:00", tz="UTC")
        self.assertFalse(is_in_session_killzone(ts5))

        # Weekend: Saturday 12:00 UTC -> False
        ts_weekend = pd.Timestamp("2026-11-07 12:00:00", tz="UTC")
        self.assertFalse(is_in_session_killzone(ts_weekend))


class TestZeroLookaheadStrategy(unittest.TestCase):

    def _generate_synthetic_df(self, n_bars: int = 250) -> pd.DataFrame:
        # Base date on Wednesday 2026-11-04 00:00 UTC
        base = pd.Timestamp("2026-11-04 00:00:00", tz="UTC")
        times = [base + pd.Timedelta(minutes=15 * i) for i in range(n_bars)]

        # Gentle uptrend
        prices = np.linspace(2000.0, 2200.0, n_bars)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2.0,
                "Low": prices - 2.0,
                "Close": prices + 0.5,
                "Volume": np.full(n_bars, 1000.0),
            },
            index=pd.DatetimeIndex(times),
        )
        return df

    def test_indicator_computation(self):
        df = self._generate_synthetic_df(250)
        config = StrategyConfig()
        df_ind = compute_strategy_indicators(df, config)

        self.assertIn("SMA_100", df_ind.columns)
        self.assertIn("SMA_200", df_ind.columns)
        self.assertIn("ATR", df_ind.columns)
        self.assertIn("Supertrend", df_ind.columns)
        self.assertIn("ST_Polarity", df_ind.columns)
        self.assertIn("Is_Pinched", df_ind.columns)

        # Check that 200 SMA is valid on last 50 bars
        self.assertFalse(np.isnan(df_ind["SMA_200"].iloc[-1]))

    def test_zero_lookahead_evaluates_index_minus_two(self):
        df = self._generate_synthetic_df(250)
        config = StrategyConfig()
        df_ind = compute_strategy_indicators(df, config)

        sig, curr_analysis, prev_analysis = evaluate_signals_zero_lookahead(df_ind, config)

        # Evaluated bar must be index -2, NOT index -1
        expected_curr_ts = df_ind.index[-2]
        expected_prev_ts = df_ind.index[-3]

        self.assertEqual(curr_analysis.timestamp_utc, expected_curr_ts)
        self.assertEqual(prev_analysis.timestamp_utc, expected_prev_ts)
        self.assertEqual(sig.bar_timestamp_utc, expected_curr_ts)

    def test_trigger_a_bullish_flip(self):
        df = self._generate_synthetic_df(250)
        config = StrategyConfig()
        df_ind = compute_strategy_indicators(df, config)

        # Simulate closed bar index -2 falling inside session at 10:00 UTC
        eval_idx = -2
        prev_idx = -3

        # Force timestamps into killzone
        new_index = list(df_ind.index)
        new_index[eval_idx] = pd.Timestamp("2026-11-04 10:00:00", tz="UTC")
        new_index[prev_idx] = pd.Timestamp("2026-11-04 09:45:00", tz="UTC")
        df_ind.index = pd.DatetimeIndex(new_index)

        # Bull Regime: Close > 200 SMA and 100 SMA > 200 SMA
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("SMA_200")] = 2000.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("SMA_100")] = 2050.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Close")] = 2100.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Bull_Regime")] = True
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Is_Pinched")] = False

        # Supertrend flips from -1 on prev bar to +1 on curr bar
        df_ind.iloc[prev_idx, df_ind.columns.get_loc("ST_Polarity")] = -1
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("ST_Polarity")] = 1
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Supertrend")] = 2080.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("ATR")] = 5.0

        sig, curr_analysis, _ = evaluate_signals_zero_lookahead(df_ind, config)
        self.assertEqual(sig.signal_type, SignalType.TRIGGER_A_BUY)
        self.assertEqual(sig.entry_price, 2100.0)
        self.assertEqual(sig.stop_loss, 2080.0)
        self.assertGreater(sig.take_profit_1, sig.entry_price)
        self.assertGreater(sig.take_profit_2, sig.take_profit_1)

    def test_trigger_b_pullback_continuation(self):
        df = self._generate_synthetic_df(250)
        config = StrategyConfig()
        df_ind = compute_strategy_indicators(df, config)

        eval_idx = -2
        prev_idx = -3

        new_index = list(df_ind.index)
        new_index[eval_idx] = pd.Timestamp("2026-11-04 11:30:00", tz="UTC")
        new_index[prev_idx] = pd.Timestamp("2026-11-04 11:15:00", tz="UTC")
        df_ind.index = pd.DatetimeIndex(new_index)

        # Bull Regime: 100 > 200, Close > 200
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("SMA_200")] = 2000.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("SMA_100")] = 2050.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Open")] = 2049.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Low")] = 2048.0   # Dipped into 100 SMA
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Close")] = 2055.0  # Closed above 100 SMA bullishly
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("High")] = 2058.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Bull_Regime")] = True
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Is_Pinched")] = False

        # Supertrend was already Bullish (+1 on both prev and curr)
        df_ind.iloc[prev_idx, df_ind.columns.get_loc("ST_Polarity")] = 1
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("ST_Polarity")] = 1
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("Supertrend")] = 2030.0
        df_ind.iloc[eval_idx, df_ind.columns.get_loc("ATR")] = 6.0

        sig, curr_analysis, _ = evaluate_signals_zero_lookahead(df_ind, config)
        self.assertEqual(sig.signal_type, SignalType.TRIGGER_B_BUY)
        self.assertEqual(sig.entry_price, 2055.0)
        self.assertLess(sig.stop_loss, sig.entry_price)
        self.assertGreater(sig.take_profit_1, sig.entry_price)


class TestTelegramIntegration(unittest.TestCase):

    def test_default_credentials_and_fallback(self):
        token, chat_id = get_telegram_credentials()
        self.assertEqual(token, DEFAULT_TELEGRAM_BOT_TOKEN)
        self.assertEqual(chat_id, DEFAULT_TELEGRAM_CHAT_ID)

    def test_format_telegram_signal_markdown(self):
        ts = pd.Timestamp("2026-11-04 12:30:00", tz="UTC")
        sig = StrategySignal(
            signal_type=SignalType.TRIGGER_A_BUY,
            symbol="GC=F",
            bar_timestamp_utc=ts,
            entry_price=2750.50,
            stop_loss=2740.00,
            take_profit_1=2766.25,
            take_profit_2=2776.75,
            atr=5.50,
            notes="Supertrend Bullish Flip in 100/200 SMA Bull Regime",
        )

        msg = format_telegram_signal(sig)

        # Verify Markdown bolding and required fields
        self.assertIn("*BUY / LONG*", msg)
        self.assertIn("*Entry Price:* *2750.50*", msg)
        self.assertIn("*Supertrend Initial Stop ($1R$):* *2740.00*", msg)
        self.assertIn("2026-11-04 12:30 UTC", msg)
        self.assertIn("*Take Profit 1 ($1.5R$):* *2766.25*", msg)


if __name__ == "__main__":
    unittest.main()
