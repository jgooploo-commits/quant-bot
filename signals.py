"""
Quant S&P 500 Scanner — P1 Signal Engine
Multi-Timeframe Momentum + Relative Strength
แทนที่ compute_signals() เดิมใน backtest_engine.py
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from config import (
    EMA_SPAN, VOL_WINDOW,
    MOMENTUM_WINDOWS, MOMENTUM_WEIGHTS, SKIP_RECENT_DAYS,
    SECTOR_ETFS, RS_WEIGHT, MOMENTUM_WEIGHT,
    WARM_UP_DAYS,
)

log = logging.getLogger(__name__)


# ── 1. โหลด Sector ETF ─────────────────────────────────────────────────────────

def fetch_sector_etfs(start: str, end: str) -> pd.DataFrame:
    """
    ดึงราคา Sector ETF ทั้ง 11 ตัวจาก yfinance
    ใช้สำหรับคำนวณ Relative Strength ของหุ้นแต่ละตัวเทียบ Sector ตัวเอง
    """
    tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
    fetch_start = (
        pd.to_datetime(start) - pd.DateOffset(days=WARM_UP_DAYS)
    ).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=fetch_start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    log.info(f"fetch_sector_etfs: โหลด {prices.shape[1]} ETFs สำเร็จ")
    return prices


# ── 2. Map หุ้น → Sector ETF ───────────────────────────────────────────────────

def build_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    สร้าง mapping ระหว่าง Ticker → Sector ETF
    ดึงข้อมูล sector จาก yfinance info
    หุ้นที่ดึงไม่ได้จะ fallback ไปใช้ SPY แทน
    """
    sector_to_etf = {v: k for k, v in SECTOR_ETFS.items()}

    # Mapping คร่าวๆ ที่รู้แน่นอน (ป้องกัน API rate limit)
    known_map = {
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK",
        "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC",
        "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
        "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
        "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV",
        "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "MPC": "XLE",
        "CAT": "XLI", "GE": "XLI", "RTX": "XLI", "HON": "XLI",
        "LIN": "XLB", "APD": "XLB", "SHW": "XLB",
        "NEE": "XLU", "DUK": "XLU", "SO": "XLU",
        "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE",
        "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
        "COST": "XLP",
    }

    result = {}
    for ticker in tickers:
        result[ticker] = known_map.get(ticker, "SPY")   # fallback = SPY

    log.info(
        f"build_sector_map: map {len(result)} tickers | "
        f"fallback SPY = {sum(1 for v in result.values() if v == 'SPY')} ตัว"
    )
    return result


# ── 3. Multi-Timeframe Momentum Score ──────────────────────────────────────────

def compute_multitf_momentum(
    stock_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    คำนวณ Momentum หลายช่วงเวลาและรวมเป็นคะแนนเดียว

    สูตร:
    1. คำนวณ raw return แต่ละ Window (21, 63, 126, 252 วัน)
    2. Window ยาว (126, 252) ตัด SKIP_RECENT_DAYS ออก เพื่อกำจัด Short-term Reversal
       ret_126 = price[-SKIP_RECENT_DAYS] / price[-126] - 1
    3. Normalize แต่ละ Window เป็น Z-score ข้ามหุ้นทุกตัวในแถวเดียวกัน
       (เพื่อให้ scale เท่ากันก่อนรวม)
    4. Composite = ผลรวมถ่วงน้ำหนัก Z-scores ตาม MOMENTUM_WEIGHTS
    5. หาร Composite ด้วย Annualized Volatility → Risk-Adjusted Score

    คืนค่า: DataFrame (index=Date, columns=Tickers) เป็น Risk-Adjusted Score รายวัน
    """
    vol = stock_prices.pct_change().rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std() * np.sqrt(252)

    composite = pd.DataFrame(0.0, index=stock_prices.index, columns=stock_prices.columns)

    for window, weight in MOMENTUM_WEIGHTS.items():
        # Window ยาว: ตัดช่วงล่าสุดออก
        if window >= 126:
            ret = stock_prices.shift(SKIP_RECENT_DAYS).pct_change(window - SKIP_RECENT_DAYS)
        else:
            ret = stock_prices.pct_change(window)

        # Z-score ข้ามหุ้นทุกตัว (cross-sectional normalization)
        z = ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1).replace(0, np.nan), axis=0)
        composite += z * weight

    # Risk-Adjust: หาร composite ด้วย Volatility
    score = composite.div(vol.replace(0, np.nan))
    return score


# ── 4. Relative Strength Score ──────────────────────────────────────────────────

def compute_relative_strength(
    stock_prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
    sector_map: dict[str, str],
    window: int = 63,
) -> pd.DataFrame:
    """
    คำนวณ Relative Strength ของหุ้นแต่ละตัวเทียบ Sector ETF ตัวเอง

    สูตร:
    RS = (return ของหุ้น window วัน) - (return ของ Sector ETF window วัน)
    → หุ้นที่ขึ้นแรงกว่า Sector จะได้ RS สูง

    Z-score cross-sectional ก่อนส่งออก
    """
    stock_ret  = stock_prices.pct_change(window)
    sector_ret = sector_prices.pct_change(window)

    rs_df = pd.DataFrame(np.nan, index=stock_prices.index, columns=stock_prices.columns)

    for ticker in stock_prices.columns:
        etf = sector_map.get(ticker, "SPY")
        if etf not in sector_ret.columns:
            etf = "SPY"
        rs_df[ticker] = stock_ret[ticker] - sector_ret[etf]

    # Z-score cross-sectional
    rs_z = rs_df.sub(rs_df.mean(axis=1), axis=0).div(
        rs_df.std(axis=1).replace(0, np.nan), axis=0
    )
    return rs_z


# ── 5. Composite Score รวม P1 ───────────────────────────────────────────────────

def compute_p1_signals(
    prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    คำนวณ Signal ทั้งหมดของ P1:
    1. Trend Filter (EMA 200)
    2. Multi-Timeframe Momentum Score
    3. Relative Strength Score
    4. Composite Score = (MOMENTUM_WEIGHT × mom_score) + (RS_WEIGHT × rs_score)
       กรองเฉพาะหุ้นที่อยู่เหนือ EMA 200

    คืนค่า dict: ema200, momentum_score, rs_score, composite_score, trend_filter
    """
    stock_prices = prices.drop(columns=["SPY"], errors="ignore")
    stock_cols   = stock_prices.columns.tolist()

    # 1. Trend Filter
    ema200       = stock_prices.ewm(span=EMA_SPAN, adjust=False, min_periods=EMA_SPAN // 2).mean()
    trend_filter = stock_prices > ema200

    # 2. Multi-Timeframe Momentum
    log.info("compute_p1_signals: คำนวณ Multi-Timeframe Momentum...")
    mom_score = compute_multitf_momentum(stock_prices)

    # 3. Relative Strength
    log.info("compute_p1_signals: คำนวณ Relative Strength...")
    sector_map = build_sector_map(stock_cols)
    rs_score   = compute_relative_strength(stock_prices, sector_prices, sector_map)

    # 4. Composite: รวม 2 Signal ด้วยน้ำหนัก
    composite = (
        mom_score * MOMENTUM_WEIGHT
        + rs_score.reindex(columns=stock_cols) * RS_WEIGHT
    )

    # 5. กรองหุ้นที่หลุด EMA 200 ออก
    composite = composite.where(trend_filter)

    log.info("compute_p1_signals: คำนวณเสร็จสิ้น")
    return {
        "ema200":          ema200,
        "momentum_score":  mom_score,
        "rs_score":        rs_score.reindex(columns=stock_cols),
        "composite_score": composite,
        "trend_filter":    trend_filter,
    }


# ── ทดสอบเบื้องต้น ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from config import BACKTEST_START

    print("⏳ โหลดข้อมูลและคำนวณ P1 Signals...")
    prices, qa    = run_pipeline(BACKTEST_START, "2026-01-01")
    sector_prices = fetch_sector_etfs(BACKTEST_START, "2026-01-01")

    signals = compute_p1_signals(prices, sector_prices)
    score   = signals["composite_score"]

    # แสดงหุ้น Top 10 ล่าสุด
    latest     = score.iloc[-1].dropna().nlargest(10)
    mom_latest = signals["momentum_score"].iloc[-1]
    rs_latest  = signals["rs_score"].iloc[-1]

    print(f"\n📊 Top 10 หุ้น P1 Score (ล่าสุด {score.index[-1].date()}):")
    print(f"{'Rank':<5} {'Ticker':<8} {'Composite':>10} {'Momentum':>10} {'RS':>8}")
    print("-" * 45)
    for rank, (ticker, val) in enumerate(latest.items(), 1):
        m = mom_latest.get(ticker, np.nan)
        r = rs_latest.get(ticker, np.nan)
        print(f"{rank:<5} {ticker:<8} {val:>10.3f} {m:>10.3f} {r:>8.3f}")
