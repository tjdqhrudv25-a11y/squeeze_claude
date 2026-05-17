import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
 
st.set_page_config(page_title="Squeeze Momentum 스캐너", layout="wide")
 
st.title("⚡ Squeeze Momentum + EMA200 스캐너")
st.markdown("""
**포착 조건:**
- ✅ 종가 > EMA 200 (장기 상승 추세)
- ✅ Squeeze **해제** 상태 (핑크색 점): 볼린저밴드가 켈트너채널 밖으로 확장
- ✅ 모멘텀 값 > 0 이고 전날보다 증가 (밝은 초록 막대)
""")
 
# ──────────────────────────────────────────
# 티커 로드
# ──────────────────────────────────────────
@st.cache_data
def get_tickers():
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r") as f:
            return [line.strip().upper() for line in f.readlines() if line.strip()]
    return []
 
# ──────────────────────────────────────────
# 지표 계산 함수
# ──────────────────────────────────────────
 
def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()
 
def calc_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()
 
def calc_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr
 
def calc_linreg(series: pd.Series, length: int) -> pd.Series:
    """Rolling linear regression — returns the last fitted value at each bar."""
    result = series.copy() * np.nan
    arr = series.values
    for i in range(length - 1, len(arr)):
        y = arr[i - length + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(length)
        # least-squares fit
        xm = x.mean()
        ym = y.mean()
        slope = np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)
        intercept = ym - slope * xm
        result.iloc[i] = slope * (length - 1) + intercept
    return result
 
def calc_squeeze_momentum(df: pd.DataFrame,
                          bb_length: int = 20,
                          bb_mult: float = 2.0,
                          kc_length: int = 20,
                          kc_mult: float = 1.5,
                          use_true_range: bool = True):
    """
    LazyBear Squeeze Momentum — Pine Script 로직을 Python으로 재현.
 
    Returns
    -------
    val      : 모멘텀 히스토그램 값
    sqz_on   : 스퀴즈 ON  (검정 점)
    sqz_off  : 스퀴즈 OFF (회색/핑크 점)  ← 우리가 원하는 상태
    no_sqz   : 스퀴즈 없음 (파란 점)
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
 
    # ── 볼린저밴드 ──
    bb_basis = calc_sma(close, bb_length)
    bb_dev   = close.rolling(bb_length).std(ddof=0) * kc_mult   # LazyBear 원본: BB dev에 KC MultFactor(1.5) 사용
    upper_bb = bb_basis + bb_dev
    lower_bb = bb_basis - bb_dev
 
    # ── 켈트너채널 ──
    kc_ma  = calc_sma(close, kc_length)
    rng    = calc_true_range(df) if use_true_range else (high - low)
    rng_ma = calc_sma(rng, kc_length)
    upper_kc = kc_ma + rng_ma * kc_mult
    lower_kc = kc_ma - rng_ma * kc_mult
 
    # ── 스퀴즈 상태 ──
    sqz_on  = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz  = (~sqz_on) & (~sqz_off)
 
    # ── 모멘텀 값 (linreg) ──
    highest_high = high.rolling(kc_length).max()
    lowest_low   = low.rolling(kc_length).min()
    mid_hl       = (highest_high + lowest_low) / 2
    mid_sma      = calc_sma(close, kc_length)
    delta        = close - (mid_hl + mid_sma) / 2
 
    val = calc_linreg(delta, kc_length)
 
    return val, sqz_on, sqz_off, no_sqz
 
 
# ──────────────────────────────────────────
# 종목 스캔 함수
# ──────────────────────────────────────────
def scan_stock(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 220:
            return None
 
        # Multi-index 해제
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
 
        for col in ['High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['High', 'Low', 'Close'], inplace=True)
 
        if len(df) < 220:
            return None
 
        # ── EMA 200 ──
        df['EMA200'] = calc_ema(df['Close'], 200)
 
        # ── Squeeze Momentum ──
        val, sqz_on, sqz_off, no_sqz = calc_squeeze_momentum(df)
        df['val']     = val
        df['sqz_off'] = sqz_off
 
        # 마지막 두 행
        last = df.iloc[-1]
        prev = df.iloc[-2]
 
        # ── 조건 판단 ──
        # 1) 종가 > EMA200 (장기 상승 추세)
        cond_ema = float(last['Close']) > float(last['EMA200'])
 
        # 2) 스퀴즈 해제 (sqz_off = True → 핑크색 점)
        cond_sqz_off = bool(last['sqz_off'])
 
        # 3) 모멘텀 > 0 이고 전날보다 증가 (밝은 초록 막대)
        cond_momentum = (
            float(last['val']) > 0 and
            float(last['val']) > float(prev['val'])
        )
 
        if cond_ema and cond_sqz_off and cond_momentum:
            ema200_val = float(last['EMA200'])
            close_val  = float(last['Close'])
            return {
                "티커":       ticker,
                "현재가":     round(close_val, 2),
                "EMA200":    round(ema200_val, 2),
                "EMA 이격(%)": round((close_val - ema200_val) / ema200_val * 100, 2),
                "모멘텀 값":  round(float(last['val']), 4),
                "모멘텀 변화": round(float(last['val']) - float(prev['val']), 4),
                "신호":       "⚡ Squeeze 해제 + 상승 모멘텀",
            }
 
    except Exception:
        return None
 
    return None
 
 
# ──────────────────────────────────────────
# UI
# ──────────────────────────────────────────
tickers = get_tickers()
 
if not tickers:
    st.error("❌ tickers.txt 파일이 없거나 비어 있습니다. 같은 폴더에 tickers.txt를 넣어주세요.")
    st.stop()
 
st.info(f"📋 총 **{len(tickers)}**개 종목을 스캔합니다.")
 
# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 파라미터 설정")
    bb_length = st.slider("BB Length",      10, 50, 20)
    bb_mult   = st.slider("BB MultFactor",  1.0, 3.0, 2.0, 0.1)
    kc_length = st.slider("KC Length",      10, 50, 20)
    kc_mult   = st.slider("KC MultFactor",  1.0, 3.0, 1.5, 0.1)
    max_workers = st.slider("병렬 스캔 수", 2, 20, 10)
 
if st.button("⚡ Squeeze Momentum 스캔 시작", type="primary"):
    results = []
    progress = st.progress(0)
    status   = st.empty()
    total    = len(tickers)
 
    def scan_wrapper(ticker):
        return scan_stock(ticker)
 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, res in enumerate(executor.map(scan_wrapper, tickers)):
            if res:
                results.append(res)
            progress.progress((i + 1) / total)
            status.text(f"스캔 중... {i+1}/{total}  |  포착: {len(results)}개")
 
    progress.empty()
    status.empty()
 
    if results:
        st.success(f"✅ 총 **{len(results)}**개 종목 포착!")
        result_df = pd.DataFrame(results).sort_values("모멘텀 값", ascending=False)
 
        # 컬럼 색상 강조
        def highlight_momentum(val):
            return "color: #00c853; font-weight: bold"
 
        st.dataframe(
            result_df.style.map(highlight_momentum, subset=["모멘텀 값", "모멘텀 변화"]),
            use_container_width=True,
            height=600,
        )
 
        # CSV 다운로드
        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv,
            file_name="squeeze_momentum_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("⚠️ 현재 조건을 충족하는 종목이 없습니다. 파라미터를 조정해보세요.")
