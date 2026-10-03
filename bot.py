from pathlib import Path
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY", "")
CLIENT = os.getenv("ANGEL_CLIENT_CODE", "")
PIN = os.getenv("ANGEL_PIN", "")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

EXCHANGE = os.getenv("EXCHANGE", "NSE")
SYMBOL = os.getenv("SYMBOL", "NIFTY 50")
TOKEN = os.getenv("SYMBOL_TOKEN", "99926000")
INTERVAL = os.getenv("INTERVAL", "FIVE_MINUTE")
DAYS = int(os.getenv("LOOKBACK_DAYS", "5"))

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high-df.low,
        (df.high-pc).abs(),
        (df.low-pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def main():
    missing = [k for k,v in {
        "ANGEL_API_KEY":API_KEY, "ANGEL_CLIENT_CODE":CLIENT,
        "ANGEL_PIN":PIN, "ANGEL_TOTP_SECRET":TOTP_SECRET
    }.items() if not v]
    if missing:
        raise SystemExit("Missing in .env: " + ", ".join(missing))

    api = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    login = api.generateSession(CLIENT, PIN, totp)
    if not login.get("status"):
        raise SystemExit(f"Login failed: {login}")

    now = datetime.now()
    params = {
        "exchange": EXCHANGE,
        "symboltoken": TOKEN,
        "interval": INTERVAL,
        "fromdate": (now-timedelta(days=DAYS)).strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M")
    }
    result = api.getCandleData(params)
    if not result.get("status"):
        raise SystemExit(f"Candle request failed: {result}")

    df = pd.DataFrame(result["data"],
        columns=["timestamp","open","high","low","close","volume"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna().sort_values("timestamp")

    df["ema20"] = ema(df.close,20)
    df["ema50"] = ema(df.close,50)
    df["rsi14"] = rsi(df.close)
    df["macd"] = ema(df.close,12)-ema(df.close,26)
    df["macd_signal"] = ema(df.macd,9)
    df["atr14"] = atr(df)

    tp = (df.high+df.low+df.close)/3
    vol = df.volume.fillna(0)
    day = df.timestamp.dt.date
    df["vwap"] = (tp*vol).groupby(day).cumsum()/vol.groupby(day).cumsum().replace(0,np.nan)

    x=df.iloc[-1]
    score=0
    reasons=[]

    if x.close>x.ema20: score+=1; reasons.append("price above EMA20")
    else: score-=1; reasons.append("price below EMA20")
    if x.ema20>x.ema50: score+=1; reasons.append("EMA20 above EMA50")
    else: score-=1; reasons.append("EMA20 below EMA50")
    if x.macd>x.macd_signal: score+=1; reasons.append("MACD above signal")
    else: score-=1; reasons.append("MACD below signal")
    if x.close>x.vwap: score+=1; reasons.append("price above VWAP")
    else: score-=1; reasons.append("price below VWAP")
    if x.rsi14<=30: score+=1; reasons.append("RSI oversold zone")
    elif x.rsi14>=70: score-=1; reasons.append("RSI overbought zone")

    if score>=3: state="BULLISH TECHNICAL SETUP"
    elif score<=-3: state="BEARISH TECHNICAL SETUP"
    else: state="NEUTRAL / WAIT"

    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/latest_analysis.csv", index=False)

    print("\n=== ANGEL ONE CHART ANALYSIS ===")
    print("SAFE MODE: NO LIVE ORDERS")
    print(f"Instrument : {EXCHANGE}:{SYMBOL}")
    print(f"Interval   : {INTERVAL}")
    print(f"Last close : {x.close:.2f}")
    print(f"EMA20      : {x.ema20:.2f}")
    print(f"EMA50      : {x.ema50:.2f}")
    print(f"RSI14      : {x.rsi14:.2f}")
    print(f"MACD       : {x.macd:.4f}")
    print(f"MACD signal: {x.macd_signal:.4f}")
    print(f"ATR14      : {x.atr14:.2f}")
    print(f"VWAP       : {x.vwap:.2f}")
    print(f"\nTECHNICAL STATE: {state}")
    print(f"SCORE: {score:+d}/5")
    print("Reasons:")
    for r in reasons: print(" -",r)
    print("\nReport saved to reports/latest_analysis.csv")

if __name__=="__main__":
    main()
