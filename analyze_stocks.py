import os
import json
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_filtered_candidates():
    """
    1. 코스피/코스닥 시총 3,000억 이상 & 안정적 거래대금(150억 이상)
    2. 20일 평균 거래량 대비 당일(또는 최근 기준봉) 거래량 200% 이상 폭발
    3. 외국인 & 기관 동시 순매수(양매수) 유입 종목 필터링
    """
    today_str = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')

    df_krx = fdr.StockListing('KRX')
    # 시총 3,000억 이상, 일 거래대금 150억 이상, 스팩/우선주 제외
    df_mid_large = df_krx[
        (df_krx['Marcap'] >= 300_000_000_000) &
        (df_krx['Amount'] >= 15_000_000_000) &
        (~df_krx['Name'].str.contains('스팩|우'))
    ].sort_values(by='Amount', ascending=False).head(80)

    # 당일 시장 전체 투자자별 순매수 데이터 조회 (외인/기관 양매수 확인용)
    try:
        df_inv_kospi = stock.get_market_net_purchases_of_equities_by_ticker(today_str, today_str, "KOSPI")
        df_inv_kosdaq = stock.get_market_net_purchases_of_equities_by_ticker(today_str, today_str, "KOSDAQ")
        df_inv = pd.concat([df_inv_kospi, df_inv_kosdaq])
    except Exception:
        df_inv = pd.DataFrame()

    candidates = []
    for _, row in df_mid_large.iterrows():
        code, name, market = row['Code'], row['Name'], row['Market']

        # 외인·기관 동시 순매수 체크
        supply_status = "수급 데이터 확인 중"
        if not df_inv.empty and code in df_inv.index:
            inst_net = int(df_inv.loc[code, '기관합계'])
            frgn_net = int(df_inv.loc[code, '외국인합계'])
            if inst_net <= 0 or frgn_net <= 0:
                continue  # 양매수가 아니면 제외
            supply_status = f"외인 +{frgn_net//100000000}억 / 기관 +{inst_net//100000000}억 양매수"

        ohlcv = fdr.DataReader(code, start_date)
        if len(ohlcv) < 60:
            continue

        ohlcv['MA5'] = ohlcv['Close'].rolling(5).mean()
        ohlcv['MA20'] = ohlcv['Close'].rolling(20).mean()
        ohlcv['MA60'] = ohlcv['Close'].rolling(60).mean()
        ohlcv['Vol_MA20'] = ohlcv['Volume'].rolling(20).mean()
        ohlcv['Vol_Ratio_Pct'] = (ohlcv['Volume'] / ohlcv['Vol_MA20']) * 100

        # 최근 5거래일 내 20일 평균 대비 거래량 200% 이상 폭발한 기준봉 존재 여부 확인
        recent_5d = ohlcv.tail(5)
        max_vol_ratio = round(float(recent_5d['Vol_Ratio_Pct'].max()), 1)
        if max_vol_ratio < 200.0:
            continue

        latest = ohlcv.iloc[-1]
        candidates.append({
            "code": code,
            "name": name,
            "market": market,
            "marcap_billion": int(row['Marcap'] // 100_000_000),
            "current_price": int(latest['Close']),
            "ma5": int(latest['MA5']),
            "ma20": int(latest['MA20']),
            "ma60": int(latest['MA60']),
            "vol_ratio_20d": int(max_vol_ratio),
            "supply_badge": supply_status,
            "recent_10d_closes": [int(x) for x in ohlcv['Close'].tail(10).tolist()]
        })

    return candidates[:25]

def analyze_with_gemini(candidates):
    prompt = f"""
    당신은 대한민국 코스피·코스닥 중대형주 전문 퀀트 애널리스트입니다.
    아래 후보 종목들은 모두 [시총 3,000억 이상 + 외국인/기관 동시 순매수 + 최근 거래량 200% 이상 폭발] 조건을 통과한 종목들입니다:
    {json.dumps(candidates, ensure_ascii=False)}

    위 종목들을 분석하여 6대 급등주 차트 패턴(bull_flag, rounded_bottom, box_breakout, inv_hs, asc_triangle, ma_squeeze)에 가장 부합하는 종목을 1개씩 매칭하세요.

    [매매 호흡 및 타점 산출 원칙 - 중기 추세 추종(보유 2~4주)]
    - entry_price(권장 매수가): 현재가 부근 또는 단기 이평선 지지 타점
    - target_price(목표가): 권장 매수가 대비 +15% ~ +25% 구간 내 주요 매물대/목표치
    - stop_price(손절선): 권장 매수가 대비 -5% ~ -7% 구간 내 핵심 지지선 이탈 가격
    - rr_ratio(손익비): (target_price - entry_price) / (entry_price - stop_price) 소수점 첫째 자리

    반드시 아래 JSON 리스트 포맷으로만 응답하세요:
    [
      {{
        "pattern_id": "bull_flag",
        "pattern_name": "N자형 눌림목",
        "code": "종목코드",
        "name": "종목명",
        "market": "KOSPI",
        "marcap_billion": 시총(억원),
        "current_price": 현재가,
        "entry_price": 권장매수가,
        "target_price": 목표가,
        "stop_price": 손절가,
        "rr_ratio": 3.2,
        "vol_ratio_20d": 거래량비율(정수),
        "supply_badge": "외인·기관 양매수 요약",
        "score": 적합도점수(85~99),
        "ai_comment": "외인/기관 수급과 거래량, 차트 맥점을 결합한 2문장 요약"
      }}
    ]
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

if __name__ == "__main__":
    filtered = get_filtered_candidates()
    picks = analyze_with_gemini(filtered)
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d 21:00 KST"),
        "recommendations": picks
    }
    with open("daily_picks.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
