import os
import sys
import json
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ 오류: GEMINI_API_KEY가 설정되지 않았습니다. GitHub Secrets를 확인하세요.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

def get_latest_trading_day():
    """오늘부터 역산하여 수급 데이터가 존재하는 가장 최근 영업일(YYYYMMDD)을 반환"""
    for i in range(7):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        try:
            df = stock.get_market_net_purchases_of_equities_by_ticker(target_date, target_date, "KOSPI")
            if not df.empty and df['기관합계'].abs().sum() > 0:
                print(f"✅ 기준 영업일 확인: {target_date}")
                return target_date
        except Exception:
            continue
    return datetime.now().strftime('%Y%m%d')

def get_filtered_candidates():
    target_date = get_latest_trading_day()
    start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')

    print("📊 KRX 전종목 리스트 수집 중...")
    df_krx = fdr.StockListing('KRX')
    
    marcap_col = 'Marcap' if 'Marcap' in df_krx.columns else 'MarketCap'
    df_mid_large = df_krx[
        (df_krx[marcap_col] >= 300_000_000_000) &
        (~df_krx['Name'].str.contains('스팩|우'))
    ].copy()

    if 'Amount' in df_mid_large.columns and df_mid_large['Amount'].sum() > 0:
        df_mid_large = df_mid_large[df_mid_large['Amount'] >= 10_000_000_000]
        df_mid_large = df_mid_large.sort_values(by='Amount', ascending=False).head(70)
    else:
        df_mid_large = df_mid_large.sort_values(by=marcap_col, ascending=False).head(70)

    try:
        df_inv_kospi = stock.get_market_net_purchases_of_equities_by_ticker(target_date, target_date, "KOSPI")
        df_inv_kosdaq = stock.get_market_net_purchases_of_equities_by_ticker(target_date, target_date, "KOSDAQ")
        df_inv = pd.concat([df_inv_kospi, df_inv_kosdaq])
    except Exception as e:
        print(f"⚠️ 수급 조회 우회 처리: {e}")
        df_inv = pd.DataFrame()

    candidates = []
    strict_candidates = []

    for _, row in df_mid_large.iterrows():
        code, name = str(row['Code']), str(row['Name'])
        market = str(row.get('Market', 'KOSPI'))
        marcap_b = int(row[marcap_col] // 100_000_000)

        try:
            ohlcv = fdr.DataReader(code, start_date)
            if len(ohlcv) < 60:
                continue

            ohlcv['MA5'] = ohlcv['Close'].rolling(5).mean()
            ohlcv['MA20'] = ohlcv['Close'].rolling(20).mean()
            ohlcv['MA60'] = ohlcv['Close'].rolling(60).mean()
            ohlcv['Vol_MA20'] = ohlcv['Volume'].rolling(20).mean()
            ohlcv['Vol_Ratio_Pct'] = (ohlcv['Volume'] / ohlcv['Vol_MA20']) * 100

            recent_5d = ohlcv.tail(5)
            max_vol_ratio = round(float(recent_5d['Vol_Ratio_Pct'].max()), 1)
            latest = ohlcv.iloc[-1]

            is_dual_buy = False
            supply_status = f"거래량 {int(max_vol_ratio)}% 급증"
            if not df_inv.empty and code in df_inv.index:
                inst_net = int(df_inv.loc[code, '기관합계'])
                frgn_net = int(df_inv.loc[code, '외국인합계'])
                if inst_net > 0 and frgn_net > 0:
                    is_dual_buy = True
                    supply_status = f"외인 +{frgn_net//100000000}억 · 기관 +{inst_net//100000000}억 양매수"

            item = {
                "code": code,
                "name": name,
                "market": market,
                "marcap_billion": marcap_b,
                "current_price": int(latest['Close']),
                "ma5": int(latest['MA5']),
                "ma20": int(latest['MA20']),
                "ma60": int(latest['MA60']),
                "vol_ratio_20d": int(max_vol_ratio),
                "supply_badge": supply_status,
                "recent_10d_closes": [int(x) for x in ohlcv['Close'].tail(10).tolist()]
            }

            if is_dual_buy and max_vol_ratio >= 200.0:
                strict_candidates.append(item)
            elif is_dual_buy or max_vol_ratio >= 150.0:
                candidates.append(item)

        except Exception:
            continue

    final_list = (strict_candidates + candidates)[:25]
    print(f"✅ 최종 분석 대상 후보군 {len(final_list)}개 추출 완료")
    return final_list

def analyze_with_gemini(candidates):
    prompt = f"""
    당신은 대한민국 코스피·코스닥 중대형주 전문 퀀트 애널리스트입니다.
    아래 후보 종목들을 분석하여 6대 급등주 차트 패턴(bull_flag, rounded_bottom, box_breakout, inv_hs, asc_triangle, ma_squeeze)에 가장 부합하는 대표 종목을 1개씩(총 6개) 선정하세요.
    후보 데이터:
    {json.dumps(candidates, ensure_ascii=False)}

    [중기 추세 추종(보유 2~4주) 타점 산출 원칙]
    - entry_price(권장 매수가): 현재가 부근 또는 단기 이평선 지지 가격(정수)
    - target_price(목표가): 권장 매수가 대비 +15% ~ +25% 구간(정수)
    - stop_price(손절선): 권장 매수가 대비 -5% ~ -7% 구간(정수)
    - rr_ratio(손익비): 소수점 첫째 자리 숫자 (예: 3.2)

    반드시 아래 JSON 배열 스키마로만 응답하세요:
    [
      {{
        "pattern_id": "bull_flag",
        "pattern_name": "N자형 눌림목",
        "code": "종목코드",
        "name": "종목명",
        "market": "KOSPI",
        "marcap_billion": 10000,
        "current_price": 100000,
        "entry_price": 99000,
        "target_price": 120000,
        "stop_price": 93000,
        "rr_ratio": 3.5,
        "vol_ratio_20d": 230,
        "supply_badge": "외인·기관 양매수 요약",
        "score": 92,
        "ai_comment": "수급과 거래량, 차트 맥점을 결합한 2문장 이내 핵심 코멘트"
      }}
    ]
    """
    # 최신 모델 gemini-3.0-flash를 1순위로 호출
    for model_name in ["gemini-3.0-flash", "gemini-3.0-pro", "gemini-2.5-pro"]:
        try:
            print(f"🤖 {model_name} 모델로 분석 요청 중...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(raw_text)
        except Exception as e:
            print(f"⚠️ {model_name} 호출 실패, 다음 모델 시도: {e}")
            continue
    raise RuntimeError("Gemini 모델 호출에 모두 실패했습니다.")

if __name__ == "__main__":
    filtered = get_filtered_candidates()
    picks = analyze_with_gemini(filtered)
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M KST"),
        "recommendations": picks
    }
    with open("daily_picks.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("🎉 daily_picks.json 생성이 완료되었습니다!")
