from pathlib import Path
import csv
import json
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

# ============================================================
# ANGEL ONE F&O PAPER SCANNER
# PAPER ONLY - NO LIVE ORDERS
#
# Coverage:
#   - NSE F&O stocks: futures + options
#   - Bank Nifty: futures + options
#   - Sensex: futures + options
#
# Analysis:
#   - 15-minute trend
#   - 5-minute entry confirmation
#   - daily/latest news for shortlisted underlyings
#   - ATR-based entry / stop-loss / target
#   - paper-trade monitoring
#
# This file is intentionally separate from bot_paper_strategy.py.
# ============================================================

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY", "")
CLIENT = os.getenv("ANGEL_CLIENT_CODE", "")
PIN = os.getenv("ANGEL_PIN", "")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

PAPER_MODE = True

MASTER_URL = (
    "https://margincalculator.angelone.in/"
    "OpenAPI_File/files/OpenAPIScripMaster.json"
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

MASTER_FILE = REPORT_DIR / "angel_master.json"
CANDIDATES_FILE = REPORT_DIR / "fno_candidates.csv"
TRADES_FILE = REPORT_DIR / "fno_paper_trades.csv"
STATUS_FILE = REPORT_DIR / "fno_scanner_status.csv"

# Scanner timing
POLL_SECONDS = 300
MARKET_START = dtime(9, 15)
NEW_ENTRY_CUTOFF = dtime(15, 0)
MARKET_END = dtime(15, 40)

# Technical strategy
ATR_STOP_MULT = 1.5
RISK_REWARD = 2.0
MAX_OPEN_PAPER_TRADES = 8
MAX_NEW_TRADES_PER_DAY = 8

# To keep API traffic manageable, the bot first filters underlyings
# using 15m + 5m data, then checks options/futures only for shortlisted
# underlyings.
MAX_UNDERLYING_SHORTLIST = 25

# Option selection
OPTION_MIN_DELTA = 0.35
OPTION_MAX_DELTA = 0.65

# News is checked only for technical candidates, not every F&O contract.
NEWS_LOOKBACK_HOURS = 36

INDEX_UNDERLYINGS = {
    "BANKNIFTY": {"exchange": "NFO", "aliases": ["BANKNIFTY", "NIFTY BANK"]},
    "SENSEX": {"exchange": "BFO", "aliases": ["SENSEX"]},
}


def require_credentials():
    missing = [
        k for k, v in {
            "ANGEL_API_KEY": API_KEY,
            "ANGEL_CLIENT_CODE": CLIENT,
            "ANGEL_PIN": PIN,
            "ANGEL_TOTP_SECRET": TOTP_SECRET,
        }.items() if not v
    ]
    if missing:
        raise SystemExit("Missing in .env: " + ", ".join(missing))


def login():
    api = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    result = api.generateSession(CLIENT, PIN, totp)
    if not result.get("status"):
        raise RuntimeError(f"Angel One login failed: {result}")
    return api


def download_master():
    """Download the current daily Angel One instrument master."""
    request = urllib.request.Request(
        MASTER_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    MASTER_FILE.write_bytes(data)
    return json.loads(data.decode("utf-8"))


def load_master():
    """Use today's master when possible; otherwise refresh it."""
    try:
        if MASTER_FILE.exists():
            age = datetime.now() - datetime.fromtimestamp(
                MASTER_FILE.stat().st_mtime
            )
            if age < timedelta(hours=20):
                return json.loads(MASTER_FILE.read_text())
    except Exception:
        pass

    return download_master()


def clean_master(raw):
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append({
            "token": str(item.get("token", "")),
            "symbol": str(item.get("symbol", "")),
            "name": str(item.get("name", "")),
            "expiry": str(item.get("expiry", "")),
            "strike": str(item.get("strike", "")),
            "lotsize": str(item.get("lotsize", "")),
            "instrumenttype": str(item.get("instrumenttype", "")),
            "exch_seg": str(item.get("exch_seg", "")),
            "tick_size": str(item.get("tick_size", "")),
        })
    return pd.DataFrame(rows)


def expiry_date(value):
    if not value:
        return None
    for fmt in ("%d%b%Y", "%d%b%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.upper(), fmt).date()
        except ValueError:
            pass
    return None


def current_and_future_expiries(df):
    today = datetime.now().date()
    values = sorted({
        d for d in (expiry_date(x) for x in df["expiry"])
        if d and d >= today
    })
    return values


def get_fno_universe(master):
    """
    Build the actual tradable F&O universe from the daily master.

    NSE F&O:
      - OPTIDX / FUTIDX for index contracts
      - OPTSTK / FUTSTK for stock contracts

    BFO:
      - SENSEX and other BSE F&O contracts are discovered from the master.
    """
    df = clean_master(master)

    nfo = df[df["exch_seg"].str.upper() == "NFO"].copy()
    bfo = df[df["exch_seg"].str.upper() == "BFO"].copy()

    expiries = current_and_future_expiries(df)
    if not expiries:
        raise RuntimeError("No future expiries found in Angel One instrument master.")

    near_expiry = expiries[0]

    def expiry_matches(frame):
        return frame["expiry"].apply(expiry_date) == near_expiry

    nfo_near = nfo[expiry_matches(nfo)]
    bfo_near = bfo[expiry_matches(bfo)]

    stock_futures = nfo_near[
        nfo_near["instrumenttype"].isin(["FUTSTK"])
    ].copy()

    stock_options = nfo_near[
        nfo_near["instrumenttype"].isin(["OPTSTK"])
    ].copy()

    bank_futures = nfo_near[
        nfo_near["instrumenttype"].isin(["FUTIDX"])
        & nfo_near["name"].str.upper().isin(["BANKNIFTY", "NIFTY BANK"])
    ].copy()

    bank_options = nfo_near[
        nfo_near["instrumenttype"].isin(["OPTIDX"])
        & nfo_near["name"].str.upper().isin(["BANKNIFTY", "NIFTY BANK"])
    ].copy()

    sensex_futures = bfo_near[
        bfo_near["instrumenttype"].isin(["FUTIDX", "FUT"])
        & bfo_near["name"].str.upper().eq("SENSEX")
    ].copy()

    sensex_options = bfo_near[
        bfo_near["instrumenttype"].isin(["OPTIDX", "OPT"])
        & bfo_near["name"].str.upper().eq("SENSEX")
    ].copy()

    return {
        "stock_futures": stock_futures,
        "stock_options": stock_options,
        "bank_futures": bank_futures,
        "bank_options": bank_options,
        "sensex_futures": sensex_futures,
        "sensex_options": sensex_options,
        "nifty50": {
            "name": "NIFTY 50",
            "symbol": "NIFTY",
            "exchange": "NSE",
            "token": "99926000",
            "instrument_type": "NIFTY50",
        },
        "near_expiry": near_expiry,
    }


def candle_data(api, exchange, token, interval, days=5):
    now = datetime.now()
    params = {
        "exchange": exchange,
        "symboltoken": str(token),
        "interval": interval,
        "fromdate": (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    }

    result = api.getCandleData(params)
    if not result.get("status"):
        raise RuntimeError(f"Candle request failed: {result}")

    rows = result.get("data") or []
    if not rows:
        raise RuntimeError("No candle data returned.")

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    df["ema20"] = ema(df.close, 20)
    df["ema50"] = ema(df.close, 50)
    df["rsi14"] = rsi(df.close)
    df["macd"] = ema(df.close, 12) - ema(df.close, 26)
    df["macd_signal"] = ema(df.macd, 9)
    df["atr14"] = atr(df)

    typical = (df.high + df.low + df.close) / 3
    vol = df.volume.fillna(0)
    day = df.timestamp.dt.date
    cumulative_volume = vol.groupby(day).cumsum()
    df["vwap"] = (
        (typical * vol).groupby(day).cumsum()
        / cumulative_volume.replace(0, np.nan)
    )
    return df


def timeframe_signal(df):
    """Return BUY/SELL/WAIT plus a compact reason list."""
    x = df.iloc[-1]
    reasons = []

    long_checks = [
        x.close > x.ema20,
        x.ema20 > x.ema50,
        x.close > x.vwap,
        x.macd > x.macd_signal,
        50 <= x.rsi14 <= 68,
    ]
    short_checks = [
        x.close < x.ema20,
        x.ema20 < x.ema50,
        x.close < x.vwap,
        x.macd < x.macd_signal,
        32 <= x.rsi14 <= 50,
    ]

    if all(long_checks):
        signal = "BUY"
    elif all(short_checks):
        signal = "SELL"
    else:
        signal = "WAIT"

    if x.close > x.ema20:
        reasons.append("price>EMA20")
    else:
        reasons.append("price<EMA20")

    if x.ema20 > x.ema50:
        reasons.append("EMA20>EMA50")
    else:
        reasons.append("EMA20<EMA50")

    if x.macd > x.macd_signal:
        reasons.append("MACD bullish")
    else:
        reasons.append("MACD bearish")

    return {
        "signal": signal,
        "close": float(x.close),
        "rsi": float(x.rsi14),
        "atr": float(x.atr14),
        "vwap": float(x.vwap) if np.isfinite(x.vwap) else np.nan,
        "reasons": ", ".join(reasons),
    }


def underlying_signal(api, exchange, token):
    """
    15m = direction filter.
    5m  = entry confirmation.
    """
    df15 = add_indicators(candle_data(api, exchange, token, "FIFTEEN_MINUTE", 10))
    df5 = add_indicators(candle_data(api, exchange, token, "FIVE_MINUTE", 5))

    s15 = timeframe_signal(df15)
    s5 = timeframe_signal(df5)

    if s15["signal"] == "BUY" and s5["signal"] == "BUY":
        final = "BUY"
    elif s15["signal"] == "SELL" and s5["signal"] == "SELL":
        final = "SELL"
    else:
        final = "WAIT"

    return {
        "final": final,
        "15m": s15,
        "5m": s5,
    }


def parse_strike(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def option_type(symbol):
    s = str(symbol).upper()
    if s.endswith("CE"):
        return "CE"
    if s.endswith("PE"):
        return "PE"
    return ""


def get_ltp(api, exchange, row):
    result = api.ltpData(
        exchange,
        str(row["symbol"]),
        str(row["token"]),
    )
    if not result.get("status"):
        raise RuntimeError(f"LTP failed for {row['symbol']}: {result}")
    return float(result["data"]["ltp"])


def choose_option(master_options, direction, underlying_price):
    """
    Select a liquid-ish ATM/near-ATM contract from the current expiry.
    The final implementation can later add a volume/OI ranking once
    we validate the basic paper scanner.
    """
    if master_options.empty:
        return None

    options = master_options.copy()
    options["strike_num"] = options["strike"].map(parse_strike)
    options["otype"] = options["symbol"].map(option_type)

    desired = "CE" if direction == "BUY" else "PE"
    options = options[options["otype"] == desired].dropna(
        subset=["strike_num"]
    )
    if options.empty:
        return None

    options["distance"] = (options["strike_num"] - underlying_price).abs()
    options = options.sort_values("distance")

    # Keep the closest contract; later versions can use Greeks/OI/volume
    # to rank several candidates.
    return options.iloc[0]


def news_for_symbol(name):
    """
    Lightweight public-news check using Google News RSS.
    News is a filter/context signal, not a trade signal by itself.
    """
    query = urllib.parse.quote(f"{name} stock India")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            root = ET.fromstring(response.read())

        cutoff = datetime.now() - timedelta(hours=NEWS_LOOKBACK_HOURS)
        items = []

        for item in root.findall("./channel/item"):
            title = item.findtext("title") or ""
            pub = item.findtext("pubDate") or ""
            try:
                dt = datetime.strptime(
                    pub[:25], "%a, %d %b %Y %H:%M:%S"
                )
            except Exception:
                dt = datetime.now()

            if dt >= cutoff:
                items.append(title.strip())

        return items[:5]
    except Exception as exc:
        return [f"News check unavailable: {exc}"]


def make_trade(side, symbol, exchange, token, entry, atr_value, instrument_type):
    if side == "BUY":
        stop = entry - ATR_STOP_MULT * atr_value
        target = entry + RISK_REWARD * (entry - stop)
    else:
        stop = entry + ATR_STOP_MULT * atr_value
        target = entry - RISK_REWARD * (stop - entry)

    return {
        "trade_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "exchange": exchange,
        "token": token,
        "instrument_type": instrument_type,
        "side": side,
        "entry": round(entry, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "status": "OPEN_PAPER",
    }


def append_trade(trade):
    exists = TRADES_FILE.exists()
    with TRADES_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(trade)


def load_open_trades():
    if not TRADES_FILE.exists():
        return []

    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        return []

    return df[df["status"] == "OPEN_PAPER"].to_dict("records")


def monitor_open_trades(api):
    """Monitor paper positions using LTP. No live order is sent."""
    if not TRADES_FILE.exists():
        return

    all_df = pd.read_csv(TRADES_FILE)
    if all_df.empty or "status" not in all_df.columns:
        return

    open_mask = all_df["status"].eq("OPEN_PAPER")
    open_df = all_df[open_mask].copy()
    if open_df.empty:
        return

    for idx, trade in open_df.iterrows():
        try:
            ltp_result = api.ltpData(
                str(trade["exchange"]),
                str(trade["symbol"]),
                str(trade["token"]),
            )
            ltp = float(ltp_result["data"]["ltp"])

            side = str(trade["side"])
            stop = float(trade["stop_loss"])
            target = float(trade["target"])
            entry = float(trade["entry"])

            if side == "BUY":
                if ltp <= stop:
                    status = "STOP_HIT"
                elif ltp >= target:
                    status = "TARGET_HIT"
                else:
                    status = "OPEN_PAPER"
                pnl = ltp - entry
            else:
                if ltp >= stop:
                    status = "STOP_HIT"
                elif ltp <= target:
                    status = "TARGET_HIT"
                else:
                    status = "OPEN_PAPER"
                pnl = entry - ltp

            trade_id = trade["trade_id"]
            mask = all_df["trade_id"].astype(str).eq(str(trade_id))
            all_df.loc[mask, "last_ltp"] = round(ltp, 2)
            all_df.loc[mask, "pnl_points"] = round(pnl, 2)
            all_df.loc[mask, "status"] = status
            all_df.loc[mask, "last_checked"] = datetime.now().isoformat(
                timespec="seconds"
            )

            if status != "OPEN_PAPER":
                print(
                    f"📌 PAPER EXIT: {trade['symbol']} | "
                    f"{status} | P&L points: {round(pnl, 2)}"
                )

        except Exception as exc:
            mask = all_df["trade_id"].astype(str).eq(str(trade["trade_id"]))
            all_df.loc[mask, "monitor_error"] = str(exc)
            all_df.loc[mask, "last_checked"] = datetime.now().isoformat(
                timespec="seconds"
            )

    all_df.to_csv(TRADES_FILE, index=False)


def shortlist_underlyings(universe):
    """
    Return stock futures plus Bank Nifty/Sensex futures as underlying
    candidates. Options are selected only after the underlying qualifies.
    """
    result = []

    for _, row in universe["stock_futures"].iterrows():
        result.append({
            "name": row["name"],
            "symbol": row["symbol"],
            "exchange": row["exch_seg"].upper(),
            "token": row["token"],
            "instrument_type": "STOCK_FUTURE",
        })

    nifty = universe["nifty50"]
    result.append({
        "name": nifty["name"],
        "symbol": nifty["symbol"],
        "exchange": nifty["exchange"],
        "token": nifty["token"],
        "instrument_type": nifty["instrument_type"],
    })

    for label, frame, instrument_type in [
        ("BANKNIFTY", universe["bank_futures"], "BANKNIFTY_FUTURE"),
        ("SENSEX", universe["sensex_futures"], "SENSEX_FUTURE"),
    ]:
        if not frame.empty:
            row = frame.iloc[0]
            result.append({
                "name": label,
                "symbol": row["symbol"],
                "exchange": row["exch_seg"].upper(),
                "token": row["token"],
                "instrument_type": instrument_type,
            })

    return result


def scan_once(api):
    master = load_master()
    universe = get_fno_universe(master)

    print("\n============================================================")
    print("ANGEL ONE ALL-F&O PAPER SCANNER")
    print("PAPER MODE: NO LIVE ORDERS")
    print(f"Near expiry: {universe['near_expiry']}")
    print("Coverage: NIFTY 50 + NSE F&O stocks + BANKNIFTY + SENSEX")
    print("Timeframes: 15m trend + 5m entry")
    print("============================================================")

    candidates = []
    underlyings = shortlist_underlyings(universe)

    for i, u in enumerate(underlyings, start=1):
        try:
            print(
                f"[{i}/{len(underlyings)}] "
                f"{u['name']} {u['symbol']} -> checking 15m/5m"
            )

            sig = underlying_signal(api, u["exchange"], u["token"])

            if sig["final"] == "WAIT":
                continue

            # News is deliberately checked only after technical qualification.
            news = news_for_symbol(u["name"])
            news_text = " | ".join(news) if news else "No recent headlines found"

            print(
                f"  SIGNAL={sig['final']} | "
                f"15m={sig['15m']['signal']} | 5m={sig['5m']['signal']}"
            )

            # Underlying paper trade
            entry = sig["5m"]["close"]
            atr_value = sig["5m"]["atr"]
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            trade = make_trade(
                sig["final"],
                u["symbol"],
                u["exchange"],
                u["token"],
                entry,
                atr_value,
                u["instrument_type"],
            )
            trade["news"] = news_text
            candidates.append(trade)

            # Select an option for the same underlying when available.
            if u["instrument_type"] == "STOCK_FUTURE":
                opts = universe["stock_options"][
                    universe["stock_options"]["name"].str.upper()
                    == str(u["name"]).upper()
                ]
            elif u["instrument_type"] == "BANKNIFTY_FUTURE":
                opts = universe["bank_options"]
            else:
                opts = universe["sensex_options"]

            try:
                option = choose_option(opts, sig["final"], entry)
            except Exception:
                option = None

            if option is not None:
                try:
                    option_exchange = option["exch_seg"].upper()
                    option_df = add_indicators(
                        candle_data(
                            api,
                            option_exchange,
                            option["token"],
                            "FIVE_MINUTE",
                            5,
                        )
                    )
                    option_state = timeframe_signal(option_df)
                    option_ltp = float(option_state["close"])
                    option_atr = float(option_state["atr"])
                    if not np.isfinite(option_atr) or option_atr <= 0:
                        raise RuntimeError("Invalid option ATR")

                    option_trade = make_trade(
                        sig["final"],
                        option["symbol"],
                        option_exchange,
                        option["token"],
                        option_ltp,
                        option_atr,
                        "OPTION",
                    )
                    option_trade["underlying"] = u["name"]
                    option_trade["news"] = news_text
                    candidates.append(option_trade)
                except Exception as exc:
                    print(f"  Option quote skipped: {exc}")

            if len(candidates) >= MAX_UNDERLYING_SHORTLIST * 2:
                break

        except Exception as exc:
            print(f"  skipped: {exc}")

    if candidates:
        pd.DataFrame(candidates).to_csv(CANDIDATES_FILE, index=False)
        print("\nPAPER TRADE CANDIDATES")
        for c in candidates:
            print(
                f"{c['exchange']}:{c['symbol']} | {c['side']} | "
                f"Entry {c['entry']} | SL {c['stop_loss']} | "
                f"Target {c['target']}"
            )
    else:
        pd.DataFrame(
            columns=[
                "created", "symbol", "exchange", "token",
                "instrument_type", "side", "entry",
                "stop_loss", "target", "status"
            ]
        ).to_csv(CANDIDATES_FILE, index=False)
        print("\nNo qualifying paper trades right now.")

    return candidates


def main():
    require_credentials()

    print("Logging in to Angel One...")
    api = login()
    print("Angel One login successful.")
    print("PAPER MODE is ON. No live orders will be placed.")

    current_day = datetime.now().date()
    trades_today = 0

    while True:
        today = datetime.now().date()
        if today != current_day:
            current_day = today
            trades_today = 0

        if datetime.now().weekday() >= 5:
            print("Weekend — waiting for the next trading day...")
            time.sleep(300)
            continue

        try:
            now_t = datetime.now().time()

            # Always monitor existing paper trades during the F&O session.
            if MARKET_START <= now_t <= MARKET_END:
                monitor_open_trades(api)

            if (
                MARKET_START <= now_t <= NEW_ENTRY_CUTOFF
                and trades_today < MAX_NEW_TRADES_PER_DAY
            ):
                open_trades = load_open_trades()
                if len(open_trades) < MAX_OPEN_PAPER_TRADES:
                    new_candidates = scan_once(api)

                    # Paper-only: take at most one new candidate per scan
                    # cycle so the system does not fill all slots at once.
                    if new_candidates:
                        trade = new_candidates[0]
                        append_trade(trade)
                        trades_today += 1
                        print(
                            f"PAPER ENTRY: {trade['symbol']} | "
                            f"{trade['side']} | Entry={trade['entry']} | "
                            f"SL={trade['stop_loss']} | Target={trade['target']}"
                        )
                else:
                    print("Open paper-trade limit reached; monitoring only.")

        except Exception as exc:
            print(f"Scanner cycle error: {exc}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
