import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import io
from datetime import datetime

# ดึงรหัสความปลอดภัยจาก GitHub Secrets
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    tickers = tables[0]['Symbol'].tolist()
    return [t.replace('.', '-') for t in tickers]

def scan_market():
    tickers = get_sp500_tickers()
    all_tickers = tickers + ["SPY"]
    
    # โหลดข้อมูลย้อนหลัง 1 ปี (แบบซ่อนข้อความโหลด)
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)['Close']
    
    spy_prices = data['SPY'].dropna()
    stock_prices = data.drop(columns=['SPY']).dropna(how='all')
    
    # คำนวณภาพรวมตลาด (Market Filter)
    spy_current = spy_prices.iloc[-1]
    spy_ema200 = spy_prices.ewm(span=200, adjust=False).mean().iloc[-1]
    
    is_market_good = spy_current > spy_ema200
    
    if not is_market_good:
        return None, spy_current, spy_ema200 # ตลาดหมี
        
    # ถ้าตลาดดี ให้สแกนรายตัว
    ema200 = stock_prices.ewm(span=200, adjust=False).mean().iloc[-1]
    current_prices = stock_prices.iloc[-1]
    
    ret_3m = stock_prices.pct_change(63).iloc[-1]
    vol_3m = stock_prices.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
    
    # กรองหุ้น
    valid_mask = (current_prices > ema200) & (ret_3m > 0) & (vol_3m > 0)
    scores = ret_3m / vol_3m
    scores = scores[valid_mask]
    
    # คัด Top 10
    top10 = scores.nlargest(10)
    
    results = []
    for ticker in top10.index:
        results.append({
            "Ticker": ticker,
            "Price": round(current_prices[ticker], 2),
            "Score": round(scores[ticker], 2),
            "Ret_3M": round(ret_3m[ticker] * 100, 2)
        })
        
    return results, spy_current, spy_ema200

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

if __name__ == "__main__":
    top10_data, spy_price, spy_ema = scan_market()
    
    today = datetime.now().strftime("%d/%m/%Y")
    msg = f"📊 *Quant S&P 500 Scanner*\n📅 วันที่: {today}\n"
    msg += "━━━━━━━━━━━━━━━━\n"
    msg += f"📈 *SPY (ตลาด):* ${spy_price:.2f}\n"
    msg += f"📉 *EMA 200:* ${spy_ema:.2f}\n\n"
    
    if top10_data is None:
        msg += "🚨 *STATUS: MARKET DOWNTREND*\n"
        msg += "ตลาดหลุดเส้น EMA 200\nระบบแนะนำ: 💵 **ถือเงินสด 100%**"
    else:
        msg += "🟢 *STATUS: MARKET UPTREND*\n"
        msg += "ตลาดดี! *🔥 หุ้น Top 10 น่าลงทุน:*\n\n"
        for i, stock in enumerate(top10_data, 1):
            msg += f"*{i}. {stock['Ticker']}* (ราคา: ${stock['Price']})\n"
            msg += f"   └ Score: {stock['Score']} | กำไร 3 ด.: +{stock['Ret_3M']}%\n"
            
    send_telegram(msg)