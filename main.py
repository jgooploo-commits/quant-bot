"""
Quant S&P 500 Scanner — Production Scanner (P1)
รันรายเดือนด้วย GitHub Actions ส่งรายงานเข้า Telegram
ใช้ P1 Signals: Multi-Timeframe Momentum + Relative Strength
พร้อม Sector Concentration Limit (Risk Engine)
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── บังคับ UTF-8 ทั้ง stdout และ stderr (แก้ปัญหา Windows CP874) ───────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import TOP_N, BACKTEST_START, SECTOR_ETFS, WARM_UP_DAYS
from data_pipeline import run_pipeline, get_sp500_tickers
from regime import detect_regime
from signals import compute_p1_signals, fetch_sector_etfs, build_sector_map

# ── Logging (UTF-8 สำหรับ Windows + Linux) ───────────────────────────────────
logging.basicConfig(
    filename="quant_pipeline.log", filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    encoding="utf-8",              # บังคับ UTF-8 ในไฟล์ log
)
log = logging.getLogger(__name__)

# Console handler แยก (ป้องกัน encode error บน Windows)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger().addHandler(_ch)

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Risk Parameters ────────────────────────────────────────────────────────────
MAX_SECTOR_WEIGHT = 0.40   # Sector เดียวถือได้ไม่เกิน 40% ของพอร์ต


# ── 1. Sector Concentration Filter ────────────────────────────────────────────

def apply_sector_limit(
    candidates: list,
    sector_map: dict,
    top_n: int = TOP_N,
) -> list:
    """
    จำกัด Sector Concentration: known sector (XLK/XLF/...) ≤ MAX_SECTOR_WEIGHT ของพอร์ต
    หุ้นที่ sector ไม่ทราบ (fallback="SPY") ไม่ถูก cap — เพราะเราไม่รู้ sector จริง

    ตัวอย่าง: TOP_N=10, MAX_SECTOR_WEIGHT=0.40 → known sector ≤ 4 ตัว
              หุ้น fallback SPY นับรวมแต่ไม่ถูก cap
    """
    known_etfs = set(SECTOR_ETFS.keys())   # XLK, XLV, XLF, ...
    final = []
    sector_count = {}
    max_per_sector = max(1, int(top_n * MAX_SECTOR_WEIGHT))

    for ticker in candidates:
        etf = sector_map.get(ticker, "SPY")

        if etf not in known_etfs:
            # fallback (ไม่รู้ sector จริง) → ผ่านเสมอ ไม่ถูก cap
            final.append(ticker)
        else:
            # known sector → ตรวจ cap
            count = sector_count.get(etf, 0)
            if count < max_per_sector:
                final.append(ticker)
                sector_count[etf] = count + 1

        if len(final) >= top_n:
            break

    return final


# ── 2. Telegram ───────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("send_telegram: ไม่พบ Secrets TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        print("[ERROR] ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",       # ใช้ HTML แทน Markdown (ง่ายกว่า)
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200 and resp.json().get("ok"):
            log.info("send_telegram: สำเร็จ")
            return True
        # ถ้า parse_mode error → ลองส่งแบบ plain text
        log.warning(f"send_telegram: API error {resp.status_code} — {resp.text[:200]}")
        payload2 = {"chat_id": TELEGRAM_CHAT_ID, "text": _strip_html(message)[:4000]}
        resp2 = requests.post(url, json=payload2, timeout=15)
        if resp2.status_code == 200 and resp2.json().get("ok"):
            log.info("send_telegram: ส่งแบบ plain text สำเร็จ")
            return True
        log.error(f"send_telegram: ล้มเหลวทั้งคู่ — {resp2.text[:200]}")
        return False
    except requests.exceptions.Timeout:
        log.error("send_telegram: Timeout")
        return False
    except Exception as e:
        log.error(f"send_telegram: {e}")
        return False


def _strip_html(text: str) -> str:
    """ลบ HTML tags ออกสำหรับ fallback plain text"""
    import re
    return re.sub(r"<[^>]+>", "", text)


# ── 3. สร้างข้อความ Telegram (HTML format) ────────────────────────────────────

def build_message(
    top_stocks: list,
    regime: str,
    scale: float,
    spy_price: float,
    spy_ema: float,
    realized_vol: float,
) -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    regime_emoji = {
        "UPTREND":   "🟢",
        "CAUTION":   "🟡",
        "DOWNTREND": "🔴",
        "CRISIS":    "🚨",
    }.get(regime, "⚪")

    lines = [
        "📊 <b>Quant S&amp;P 500 Scanner (P1)</b>",
        f"📅 {today}",
        "━━━━━━━━━━━━━━━━━━━",
        f"<b>Regime:</b> {regime_emoji} <code>{regime}</code> → ลงทุน <code>{scale*100:.0f}%</code>",
        f"<b>SPY:</b> <code>${spy_price:,.2f}</code> │ <b>EMA200:</b> <code>${spy_ema:,.2f}</code>",
        f"<b>Vol:</b> <code>{realized_vol*100:.1f}%/ปี</code>",
        "━━━━━━━━━━━━━━━━━━━",
    ]

    if scale == 0:
        lines += [
            "",
            "⛔ <b>ระบบแนะนำ: ถือเงินสด 100%</b>",
            f"<i>ตลาดอยู่ใน {regime} — รอสัญญาณกลับตัว</i>",
        ]
    elif not top_stocks:
        lines.append("\n⚠️ ไม่มีหุ้นผ่านเกณฑ์เดือนนี้")
    else:
        lines.append(f"\n🔥 <b>Top {len(top_stocks)} หุ้น (P1 Score)</b>\n")
        prev_sector = ""
        for i, s in enumerate(top_stocks, 1):
            if s["sector"] != prev_sector:
                lines.append(f"  <i>[{s['sector']}]</i>")
                prev_sector = s["sector"]
            lines.append(
                f"<b>{i}. {s['ticker']}</b> — <code>${s['price']:,.2f}</code>\n"
                f"   Score: <code>{s['score']:.2f}</code> │ "
                f"Mom: <code>{s['mom_pct']:+.1f}%</code> │ "
                f"RS: <code>{s['rs']:.2f}</code>"
            )
        if scale < 1.0:
            lines.append(f"\n⚠️ <i>Regime CAUTION: ลงทุนแค่ {scale*100:.0f}%</i>")

    lines += [
        "━━━━━━━━━━━━━━━━━━━",
        "<i>Multi-TF Momentum + Relative Strength vs Sector</i>",
    ]
    return "\n".join(lines)


# ── 4. Main Scanner ───────────────────────────────────────────────────────────

def run_scanner():
    log.info("=" * 60)
    log.info(f"run_scanner: เริ่ม {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── กำหนดช่วงเวลาดึงข้อมูล ──────────────────────────────────────────────────
    today     = datetime.now()
    end_str   = (today + timedelta(days=1)).strftime("%Y-%m-%d")   # yfinance exclusive end
    # ดึงย้อนหลัง 2 ปี (730 วัน) + warm-up เพื่อให้ EMA200 / Momentum มีข้อมูลเพียงพอ
    start_str = (today - timedelta(days=730 + WARM_UP_DAYS)).strftime("%Y-%m-%d")

    log.info(f"run_scanner: ดึงข้อมูล {start_str} → {end_str}")

    # ── โหลดข้อมูล ─────────────────────────────────────────────────────────────
    tickers = get_sp500_tickers()
    prices, qa = run_pipeline(start=start_str, end=end_str, tickers=tickers)

    if prices is None or prices.empty:
        log.error("run_scanner: prices ว่างเปล่า — หยุดทำงาน")
        sys.exit(1)

    log.info(f"run_scanner: โหลดสำเร็จ {prices.shape[1]} tickers x {prices.shape[0]} วัน")

    # โหลด Sector ETF (ดึงในช่วงเดียวกัน)
    sector_prices = fetch_sector_etfs(start=start_str, end=end_str)

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
    top_stocks = []

    if scale > 0:
        try:
            signals   = compute_p1_signals(prices, sector_prices)
            score_row = signals["composite_score"].iloc[-1]
            mom_row   = signals["momentum_score"].iloc[-1]
            rs_row    = signals["rs_score"].iloc[-1]

            # คัด Top 30 ก่อน แล้วค่อยกรอง Sector Limit
            stock_cols  = [c for c in prices.columns if c != "SPY"]
            sector_map  = build_sector_map(stock_cols)
            top30       = score_row.dropna().nlargest(30).index.tolist()
            top_limited = apply_sector_limit(top30, sector_map, top_n=TOP_N)

            for ticker in top_limited:
                try:
                    etf    = sector_map.get(ticker, "SPY")
                    # SECTOR_ETFS: {"XLK": "Technology", ...} → ดึง sector name
                    sector = SECTOR_ETFS.get(etf, "Other")
                    top_stocks.append({
                        "ticker":  ticker,
                        "price":   round(float(prices[ticker].iloc[-1]), 2),
                        "score":   round(float(score_row[ticker]), 3),
                        "mom_pct": round(float(mom_row.get(ticker, np.nan)) * 100, 1),
                        "rs":      round(float(rs_row.get(ticker, np.nan)), 3),
                        "sector":  sector,
                    })
                except Exception as e:
                    log.warning(f"run_scanner: skip {ticker} — {e}")
                    continue

        except Exception as e:
            log.error(f"run_scanner: compute_p1_signals ล้มเหลว — {e}")
            # ยังส่ง Telegram ได้ แต่ไม่มีรายชื่อหุ้น
            top_stocks = []

    # ── ส่ง Telegram ───────────────────────────────────────────────────────────
    msg     = build_message(top_stocks, regime, scale, spy_price, spy_ema, realized_vol)
    success = send_telegram(msg)

    status = "✅ สำเร็จ" if success else "❌ ล้มเหลว (Telegram)"
    print(f"\n{status}")
    print(f"Regime: {regime} | ลงทุน: {scale:.0%} | หุ้น: {len(top_stocks)} ตัว")

    log.info(f"run_scanner: เสร็จสิ้น | {status}")
    log.info("=" * 60)

    # Exit code 1 ถ้า Telegram ส่งไม่ได้ (ให้ GitHub Actions รู้ว่า fail)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    run_scanner()
