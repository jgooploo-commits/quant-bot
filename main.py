"""
Quant S&P 500 Scanner — Production Scanner (P1)
รันรายเดือนด้วย GitHub Actions ส่งรายงานเข้า Telegram
ใช้ P1 Signals: Multi-Timeframe Momentum + Relative Strength
พร้อม Sector Concentration Limit (Risk Engine)
"""

import os
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

from config import TOP_N, BACKTEST_START, SECTOR_ETFS
from data_pipeline import run_pipeline, get_sp500_tickers
from regime import detect_regime
from signals import compute_p1_signals, fetch_sector_etfs, build_sector_map

logging.basicConfig(
    filename="quant_pipeline.log", filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Risk Parameters ────────────────────────────────────────────────────────────
MAX_SECTOR_WEIGHT = 0.40   # Sector เดียวถือได้ไม่เกิน 40% ของพอร์ต
MAX_SINGLE_STOCK  = 0.20   # หุ้นตัวเดียวถือได้ไม่เกิน 20% ของพอร์ต


# ── 1. Sector Concentration Filter ────────────────────────────────────────────

def apply_sector_limit(
    candidates: list[str],
    sector_map: dict[str, str],
    top_n: int = TOP_N,
) -> list[str]:
    """
    คัดหุ้นออกถ้า Sector เดียวกันมีน้ำหนักเกิน MAX_SECTOR_WEIGHT
    ตัดหุ้นอันดับท้ายสุดของ Sector ที่แน่นเกินออกทีละตัว จนกว่าจะผ่าน

    ตัวอย่าง: ถ้า Top 10 มีหุ้น Energy 5 ตัว (50%) → ตัดจนเหลือ 4 ตัว (40%)
              แล้วเติมหุ้นอันดับถัดไปจาก Sector อื่นแทน
    """
    final = []
    sector_count: dict[str, int] = {}
    max_per_sector = max(1, int(top_n * MAX_SECTOR_WEIGHT))

    for ticker in candidates:
        sector = sector_map.get(ticker, "SPY")
        count  = sector_count.get(sector, 0)
        if count < max_per_sector:
            final.append(ticker)
            sector_count[sector] = count + 1
        if len(final) >= top_n:
            break

    return final


# ── 2. Telegram ───────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("send_telegram: ไม่พบ Secrets")
        print("❌ ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
        return False
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        if resp.json().get("ok"):
            log.info("send_telegram: สำเร็จ")
            return True
        log.error(f"send_telegram: API error — {resp.json()}")
        return False
    except requests.exceptions.Timeout:
        log.error("send_telegram: Timeout")
        return False
    except Exception as e:
        log.error(f"send_telegram: {e}")
        return False


# ── 3. สร้างข้อความ Telegram ─────────────────────────────────────────────────

def build_message(
    top_stocks: list[dict],
    regime: str,
    scale: float,
    spy_price: float,
    spy_ema: float,
    realized_vol: float,
) -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    regime_emoji = {"UPTREND": "🟢", "CAUTION": "🟡",
                    "DOWNTREND": "🔴", "CRISIS": "🚨"}.get(regime, "⚪")

    msg  = f"📊 *Quant S\\&P 500 Scanner \\(P1\\)*\n"
    msg += f"📅 `{today}`\n"
    msg += "━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*Regime:* {regime_emoji} `{regime}` → ลงทุน `{scale*100:.0f}%`\n"
    msg += f"*SPY:* `${spy_price:,.2f}` │ *EMA200:* `${spy_ema:,.2f}`\n"
    msg += f"*Vol:* `{realized_vol*100:.1f}%/ปี`\n"
    msg += "━━━━━━━━━━━━━━━━━━━\n"

    if scale == 0:
        msg += "\n⛔ *ระบบแนะนำ: ถือเงินสด 100%*\n"
        msg += f"_ตลาดอยู่ใน {regime} — รอสัญญาณกลับตัว_"
    elif not top_stocks:
        msg += "\n⚠️ ไม่มีหุ้นผ่านเกณฑ์สัปดาห์นี้"
    else:
        msg += f"\n🔥 *Top {len(top_stocks)} หุ้น \\(P1 Score\\)*\n\n"
        prev_sector = ""
        for i, s in enumerate(top_stocks, 1):
            sector_label = f"  `[{s['sector']}]`\n" if s['sector'] != prev_sector else ""
            prev_sector  = s['sector']
            msg += sector_label
            msg += f"*{i}\\. {s['ticker']}* — `${s['price']:,.2f}`\n"
            msg += f"   Score: `{s['score']:.2f}` │ Mom: `{s['mom_pct']:+.1f}%` │ RS: `{s['rs']:.2f}`\n"
        if scale < 1.0:
            msg += f"\n⚠️ _Regime CAUTION: ลงทุนแค่ {scale*100:.0f}%_"

    msg += "\n━━━━━━━━━━━━━━━━━━━"
    msg += "\n_Multi\\-TF Momentum \\+ Relative Strength vs Sector_"
    return msg


# ── 4. Main Scanner ───────────────────────────────────────────────────────────

def run_scanner():
    log.info("=" * 60)
    log.info(f"run_scanner: เริ่ม {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── โหลดข้อมูล ─────────────────────────────────────────────────────────────
    today_str = datetime.now().strftime("%Y-%m-%d")
    tickers   = get_sp500_tickers()
    prices, qa = run_pipeline(start=today_str, end=today_str, tickers=tickers)

    # โหลด Sector ETF
    sector_prices = fetch_sector_etfs(today_str, today_str)

    # ── ตรวจ Regime ────────────────────────────────────────────────────────────
    spy_prices   = prices["SPY"].dropna()
    regime_daily = detect_regime(spy_prices)
    latest       = regime_daily.iloc[-1]

    regime       = latest["regime"]
    scale        = float(latest["scale"])
    spy_price    = float(spy_prices.iloc[-1])
    spy_ema      = float(latest["spy_ema200"])
    realized_vol = float(latest["realized_vol"])

    log.info(f"run_scanner: Regime={regime}, Scale={scale:.0%}")

    # ── คำนวณ P1 Signal และคัดหุ้น ─────────────────────────────────────────────
    top_stocks: list[dict] = []

    if scale > 0:
        signals   = compute_p1_signals(prices, sector_prices)
        score_row = signals["composite_score"].iloc[-1]
        mom_row   = signals["momentum_score"].iloc[-1]
        rs_row    = signals["rs_score"].iloc[-1]

        # คัด Top 30 ก่อน แล้วค่อยกรอง Sector Limit
        stock_cols  = [c for c in prices.columns if c != "SPY"]
        sector_map  = build_sector_map(stock_cols)
        top30       = score_row.dropna().nlargest(30).index.tolist()
        top_limited = apply_sector_limit(top30, sector_map, top_n=TOP_N)

        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        for ticker in top_limited:
            try:
                etf    = sector_map.get(ticker, "SPY")
                sector = SECTOR_ETFS.get(etf, "Other")
                top_stocks.append({
                    "ticker":   ticker,
                    "price":    round(float(prices[ticker].iloc[-1]), 2),
                    "score":    round(float(score_row[ticker]), 3),
                    "mom_pct":  round(float(mom_row.get(ticker, 0)) * 100, 1),
                    "rs":       round(float(rs_row.get(ticker, 0)), 3),
                    "sector":   sector,
                })
            except Exception:
                continue

    # ── ส่ง Telegram ───────────────────────────────────────────────────────────
    msg     = build_message(top_stocks, regime, scale, spy_price, spy_ema, realized_vol)
    success = send_telegram(msg)

    status = "✅ สำเร็จ" if success else "❌ ล้มเหลว"
    print(f"{status} | Regime: {regime} | หุ้น: {len(top_stocks)} ตัว | Scale: {scale:.0%}")
    log.info(f"run_scanner: เสร็จสิ้น | {status}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_scanner()
