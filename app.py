import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Analysis App", layout="wide")

st.title("📈 Stock Analysis App")
st.write("Technical + Fundamentals + Financials + Entry Price Analysis")
st.warning("Educational analysis only. This is not guaranteed financial advice.")

# -----------------------------
# USER INPUT
# -----------------------------
stock = st.text_input(
    "Enter stock symbol",
    value="ADANIPOWER.NS",
    help="Examples: ADANIPOWER.NS, NTPC.NS, IREDA.NS, NHPC.NS"
).upper()

period = st.selectbox("Select period", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.selectbox("Select interval", ["1d", "1wk"], index=0)

run_analysis = st.button("Run Analysis")

# -----------------------------
# FUNCTIONS
# -----------------------------
def calculate_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_candle_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    open_price = float(last["Open"])
    close_price = float(last["Close"])
    high = float(last["High"])
    low = float(last["Low"])

    prev_open = float(prev["Open"])
    prev_close = float(prev["Close"])

    body = abs(close_price - open_price)
    upper_wick = high - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low

    if body == 0:
        body = 0.01

    if lower_wick > 2 * body and upper_wick < body:
        return "Hammer - Possible bullish reversal"
    elif upper_wick > 2 * body and lower_wick < body:
        return "Shooting Star - Possible bearish reversal"
    elif prev_close < prev_open and close_price > open_price and close_price > prev_open and open_price < prev_close:
        return "Bullish Engulfing - Strong bullish signal"
    elif prev_close > prev_open and close_price < open_price and open_price > prev_close and close_price < prev_open:
        return "Bearish Engulfing - Bearish signal"
    else:
        return "No strong candle pattern"


def format_large_number(num):
    if num is None:
        return "N/A"
    try:
        num = float(num)
        if abs(num) >= 1e12:
            return f"{num / 1e12:.2f} Trillion"
        elif abs(num) >= 1e9:
            return f"{num / 1e9:.2f} Billion"
        elif abs(num) >= 1e7:
            return f"{num / 1e7:.2f} Crore"
        else:
            return f"{num:.2f}"
    except Exception:
        return "N/A"


def format_percent(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


# -----------------------------
# MAIN APP
# -----------------------------
if run_analysis:
    try:
        ticker = yf.Ticker(stock)
        data = yf.download(stock, period=period, interval=interval, auto_adjust=False, progress=False)

        if data.empty:
            st.error("No data found. Please check the stock symbol.")
            st.stop()

        data.dropna(inplace=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Technical indicators
        data["SMA_20"] = data["Close"].rolling(20).mean()
        data["SMA_50"] = data["Close"].rolling(50).mean()
        data["SMA_200"] = data["Close"].rolling(200).mean()
        data["RSI"] = calculate_rsi(data)

        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

        recent_data = data.tail(60)

        support = float(recent_data["Low"].min())
        resistance = float(recent_data["High"].max())

        current_price = float(data["Close"].iloc[-1])
        sma_20 = float(data["SMA_20"].iloc[-1])
        sma_50 = float(data["SMA_50"].iloc[-1])
        sma_200 = float(data["SMA_200"].iloc[-1])
        rsi = float(data["RSI"].iloc[-1])
        macd = float(data["MACD"].iloc[-1])
        signal = float(data["Signal_Line"].iloc[-1])

        short_term_profit_percent = ((resistance - current_price) / current_price) * 100
        downside_risk_percent = ((current_price - support) / current_price) * 100

        # Entry price calculation
        pullback_entry_low = sma_20 * 0.98
        pullback_entry_high = sma_20 * 1.02

        safe_entry_low = sma_50 * 0.98
        safe_entry_high = sma_50 * 1.02

        breakout_entry = resistance * 1.01

        stop_loss_pullback = support * 0.97
        stop_loss_breakout = resistance * 0.97

        target_1 = resistance
        target_2 = resistance * 1.08

        candle_pattern = detect_candle_pattern(data)

        # Fundamentals
        try:
            info = ticker.info
        except Exception:
            info = {}

        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        pb_ratio = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        dividend_yield = info.get("dividendYield")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        # Financial performance
        latest_revenue = None
        latest_profit = None
        revenue_growth = None
        profit_growth = None

        try:
            financials = ticker.quarterly_financials
            revenue_row = financials.loc["Total Revenue"]
            profit_row = financials.loc["Net Income"]

            latest_revenue = float(revenue_row.iloc[0])
            previous_revenue = float(revenue_row.iloc[1])

            latest_profit = float(profit_row.iloc[0])
            previous_profit = float(profit_row.iloc[1])

            revenue_growth = ((latest_revenue - previous_revenue) / previous_revenue) * 100
            profit_growth = ((latest_profit - previous_profit) / previous_profit) * 100
        except Exception:
            pass

        # Scoring
        technical_score = 0
        fundamental_score = 0

        if current_price > sma_20:
            technical_score += 1
        if current_price > sma_50:
            technical_score += 1
        if current_price > sma_200:
            technical_score += 1
        if macd > signal:
            technical_score += 1
        if rsi < 70:
            technical_score += 1
        if "Bullish" in candle_pattern or "Hammer" in candle_pattern:
            technical_score += 1

        if roe is not None and roe > 0.15:
            fundamental_score += 2
        elif roe is not None and roe > 0.10:
            fundamental_score += 1

        if pe_ratio is not None and pe_ratio < 25:
            fundamental_score += 2
        elif pe_ratio is not None and pe_ratio < 40:
            fundamental_score += 1

        if pb_ratio is not None and pb_ratio < 3:
            fundamental_score += 2
        elif pb_ratio is not None and pb_ratio < 7:
            fundamental_score += 1

        if debt_to_equity is not None and debt_to_equity < 50:
            fundamental_score += 2
        elif debt_to_equity is not None and debt_to_equity < 100:
            fundamental_score += 1

        if profit_growth is not None and profit_growth > 20:
            fundamental_score += 2
        elif profit_growth is not None and profit_growth > 5:
            fundamental_score += 1

        total_score = technical_score + fundamental_score

        if total_score >= 11:
            final_view = "STRONG STOCK - Suitable for long-term holding, but buy only at good entry levels."
        elif total_score >= 8:
            final_view = "GOOD STOCK - Can be considered, but entry timing is important."
        elif total_score >= 5:
            final_view = "AVERAGE STOCK - Wait for better price or stronger confirmation."
        else:
            final_view = "WEAK / RISKY STOCK - Avoid fresh entry unless fundamentals improve."

        if rsi > 70:
            short_term_view = "Short-term: Overbought. Profit booking or pullback is possible."
        elif current_price > sma_20 > sma_50:
            short_term_view = "Short-term: Bullish momentum, but avoid chasing if price is far above 20 DMA."
        elif current_price < sma_50:
            short_term_view = "Short-term: Weak. Wait for recovery above 50 DMA."
        else:
            short_term_view = "Short-term: Neutral. Wait for clearer candle confirmation."

        if current_price > sma_200 and fundamental_score >= 6:
            long_term_view = "Long-term: Positive, if earnings continue growing."
        elif current_price < sma_200:
            long_term_view = "Long-term: Weak until price moves above 200 DMA."
        else:
            long_term_view = "Long-term: Neutral. Need stronger fundamentals or trend."

        if rsi > 70:
            entry_advice = "Do not enter now. Stock is overbought. Wait for pullback near 20 DMA or 50 DMA."
        elif current_price >= resistance * 0.98:
            entry_advice = f"Price is near resistance. Fresh entry is risky. Enter only above ₹{breakout_entry:.2f} after daily breakout confirmation."
        elif pullback_entry_low <= current_price <= pullback_entry_high:
            entry_advice = f"Possible aggressive entry near current level because price is near 20 DMA. Suggested Entry: ₹{pullback_entry_low:.2f} - ₹{pullback_entry_high:.2f}. Enter only if bullish candle appears."
        elif safe_entry_low <= current_price <= safe_entry_high:
            entry_advice = f"Good safer entry zone near 50 DMA. Suggested Entry: ₹{safe_entry_low:.2f} - ₹{safe_entry_high:.2f}."
        else:
            entry_advice = f"Best strategy: wait for pullback near ₹{safe_entry_low:.2f} - ₹{safe_entry_high:.2f}, or breakout above ₹{breakout_entry:.2f}."

        # Display report
        st.subheader(f"Report for {stock}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"₹{current_price:.2f}")
        c2.metric("20 DMA", f"₹{sma_20:.2f}")
        c3.metric("50 DMA", f"₹{sma_50:.2f}")
        c4.metric("200 DMA", f"₹{sma_200:.2f}")

        st.subheader("Final View")
        st.info(short_term_view)
        st.info(long_term_view)
        st.success(final_view)
        st.warning(entry_advice)

        st.subheader("Entry Price Plan")
        entry_table = pd.DataFrame({
            "Item": [
                "Aggressive Entry Zone near 20 DMA",
                "Safer Entry Zone near 50 DMA",
                "Breakout Entry",
                "Stop Loss for Pullback Entry",
                "Stop Loss for Breakout Entry",
                "Target 1",
                "Target 2"
            ],
            "Value": [
                f"₹{pullback_entry_low:.2f} - ₹{pullback_entry_high:.2f}",
                f"₹{safe_entry_low:.2f} - ₹{safe_entry_high:.2f}",
                f"Above ₹{breakout_entry:.2f}",
                f"₹{stop_loss_pullback:.2f}",
                f"₹{stop_loss_breakout:.2f}",
                f"₹{target_1:.2f}",
                f"₹{target_2:.2f}"
            ]
        })
        st.table(entry_table)

        st.subheader("Technical Analysis")
        technical_table = pd.DataFrame({
            "Indicator": ["Support", "Resistance", "RSI", "MACD", "Signal Line", "Candle Pattern"],
            "Value": [
                f"₹{support:.2f}",
                f"₹{resistance:.2f}",
                f"{rsi:.2f}",
                f"{macd:.2f}",
                f"{signal:.2f}",
                candle_pattern
            ]
        })
        st.table(technical_table)

        st.subheader("Fundamental Analysis")
        fundamental_table = pd.DataFrame({
            "Metric": [
                "Sector", "Industry", "Market Cap", "P/E Ratio", "Forward P/E",
                "P/B Ratio", "ROE", "Debt to Equity", "EPS", "Book Value", "Dividend Yield"
            ],
            "Value": [
                sector, industry, format_large_number(market_cap), pe_ratio, forward_pe,
                pb_ratio, format_percent(roe), debt_to_equity, eps, book_value, format_percent(dividend_yield)
            ]
        })
        st.table(fundamental_table)

        st.subheader("Financial Performance")
        if latest_revenue is not None:
            financial_table = pd.DataFrame({
                "Metric": [
                    "Latest Quarterly Revenue",
                    "Latest Quarterly Profit",
                    "Quarterly Revenue Growth",
                    "Quarterly Profit Growth"
                ],
                "Value": [
                    format_large_number(latest_revenue),
                    format_large_number(latest_profit),
                    f"{revenue_growth:.2f}%",
                    f"{profit_growth:.2f}%"
                ]
            })
            st.table(financial_table)
        else:
            st.write("Financial performance data not available from yfinance.")

        st.subheader("Expected Short-Term Profit / Risk")
        c5, c6, c7 = st.columns(3)
        c5.metric("Upside till resistance", f"{short_term_profit_percent:.2f}%")
        c6.metric("Downside till support", f"{downside_risk_percent:.2f}%")
        c7.metric("Total Score", f"{total_score}/16")

        # Candlestick chart
        st.subheader("Candlestick Chart with DMA and Entry Zones")

        apds = [
            mpf.make_addplot(data["SMA_20"], color="blue"),
            mpf.make_addplot(data["SMA_50"], color="orange"),
            mpf.make_addplot(data["SMA_200"], color="red"),
        ]

        horizontal_lines = dict(
            hlines=[support, resistance, pullback_entry_low, pullback_entry_high, safe_entry_low, safe_entry_high, breakout_entry],
            colors=["green", "red", "blue", "blue", "orange", "orange", "purple"],
            linestyle="--",
            linewidths=1
        )

        fig, axlist = mpf.plot(
            data,
            type="candle",
            style="yahoo",
            volume=True,
            addplot=apds,
            hlines=horizontal_lines,
            title=f"{stock} Technical Chart with Entry Zones",
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8),
            returnfig=True
        )
        st.pyplot(fig)

        # RSI Chart
        st.subheader("RSI Chart")
        fig_rsi, ax_rsi = plt.subplots(figsize=(14, 4))
        ax_rsi.plot(data.index, data["RSI"], label="RSI")
        ax_rsi.axhline(70, linestyle="--", label="Overbought")
        ax_rsi.axhline(30, linestyle="--", label="Oversold")
        ax_rsi.set_title(f"{stock} RSI")
        ax_rsi.set_ylabel("RSI")
        ax_rsi.legend()
        st.pyplot(fig_rsi)

        # MACD Chart
        st.subheader("MACD Chart")
        fig_macd, ax_macd = plt.subplots(figsize=(14, 4))
        ax_macd.plot(data.index, data["MACD"], label="MACD")
        ax_macd.plot(data.index, data["Signal_Line"], label="Signal Line")
        ax_macd.set_title(f"{stock} MACD")
        ax_macd.legend()
        st.pyplot(fig_macd)

    except Exception as e:
        st.error(f"Error: {e}")