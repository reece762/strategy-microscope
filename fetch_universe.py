#!/usr/bin/env python3
"""擴大股票池，用來公平檢驗 Rayner 布林通道的「多股票版」。

為什麼需要
----------
用 40 支股票跑他原話的規則，28 年只得到 145 筆交易，平均曝險 1.7%。
這不是策略的問題，是我的股票池太小：他明說自己在 Russell 10000（數千支）
裡篩選，而我只有 40 支。股票池越小，能填滿 5 個部位的機會越少。
若不擴大就下結論，會把「我的資料不足」誤判成「他的策略無效」。
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data_yahoo')
os.makedirs(CACHE, exist_ok=True)

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')

TICKERS = """
ACN ADP AEP ALL AMGN APD AXP BBY BDX BEN BIIB BK BMY BSX C CAH CB CCL
CHTR CI CL CLX CMCSA CME CMG COP CSX CVS D DAL DE DHR DOW DUK EBAY ECL
ED EIX EL EMR EOG EQIX EQR ES ETN EW EXC EXPD FANG FAST FCX FDX FE FFIV
FIS FISV FITB FMC FOXA FRT GILD GIS GLW GPC GPN GPS GRMN GWW HAL HAS
HBAN HCA HIG HII HON HPQ HRL HSY HUM ICE IFF ILMN IP IPG IR ITW IVZ
JBHT JCI JKHY JNPR K KEY KHC KLAC KMB KMI KMX KR LEG LEN LH LKQ LLY
LMT LNC LUV LW LYB M MA MAR MAS MCHP MCK MCO MDLZ MDT MET MGM MKC
MKTX MLM MMM MNST MO MOS MRO MS MSCI MSI MTB MTD NEE NEM NKE NOC NOV
NRG NUE NWL O OKE OMC ORLY OXY PAYX PBCT PCAR PKG PLD PM PNC PPG PPL
PRU PSA PVH PWR RCL REG REGN RF RHI RJF RMD ROK ROL ROP ROST RSG SBAC
SCHW SEE SHW SLB SLG SNA SNPS SO SPG SPGI SRE STT STX STZ SWK SWKS SYF
SYK SYY TAP TDG TEL TFX TGT TJX TMO TMUS TPR TRIP TROW TRV TSN TTWO
TXT UNH UNM UNP UPS URI USB V VFC VLO VMC VNO VRSK VRSN VRTX VTR WAT
WBA WDC WEC WELL WFC WHR WM WMB WU XEL XRAY XRX XYL YUM ZBH ZION ZTS
ABBV ABMD ALGN ALK AME ANSS ARE AVB AVY BLL BRO BWA CHD CINF CMS CNP
COF COO CPB CRM(dup) CSGP CTAS CTLT CTXS DISCA DFS DISH DRE DRI DTU
ETFC EVRG EXR FDS FLIR FLS FRT(dup) GCI GEVO GILD(dup) HBI HES HPE HST
IEX INFO J JUUL KSU LB LEG(dup) LNT LUMN MAA MKC(dup) MOL NDSN NLOK
NVR ODFL ORCL(dup) PAYC PEAK PGR PH PKI PNR POOL PTC RJF(dup) ROL(dup)
SAI SBNY SGEN SIVB Sjm SNA(dup) SRE(dup) STE STI STOR SYK(dup) TECH
TIF TREE TROW(dup) TSS UDR UHS UNVR VNO(dup) VST WAB WAT(dup) WHR(dup)
WLTW WY WRB WYNN XEC XLNX ZBRA ZG
""".split()


def fetch_one(sym, rng='30y'):
    path = os.path.join(CACHE, f'{sym}_30y.csv')
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return 'cached'
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
           f'?range={rng}&interval=1d')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    import pandas as pd
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.load(r)
            res = d['chart']['result'][0]
            q = res['indicators']['quote'][0]
            df = pd.DataFrame({
                'open': q['open'], 'high': q['high'], 'low': q['low'],
                'close': q['close'], 'volume': q.get('volume'),
            }, index=pd.to_datetime(res['timestamp'], unit='s', utc=True)
               .tz_convert('America/New_York').normalize().tz_localize(None))
            df = df.dropna(subset=['close']).sort_index()
            df = df[~df.index.duplicated(keep='last')]
            if len(df) < 500:
                return f'too-short({len(df)})'
            df.to_csv(path)
            return f'ok({len(df)})'
        except Exception as e:
            if attempt == 2:
                return type(e).__name__
            time.sleep(8 * (attempt + 1))
    return 'fail'


def main():
    seen = set()
    clean = []
    for t in TICKERS:
        t = t.upper()
        if '(dup)' in t.lower() or not t.isalpha():
            continue
        if t in seen:
            continue
        seen.add(t)
        clean.append(t)
    print(f'目標 {len(clean)} 支', flush=True)
    ok = cached = fail = 0
    for i, s in enumerate(clean, 1):
        r = fetch_one(s)
        if r == 'cached':
            cached += 1
        elif r.startswith('ok'):
            ok += 1
            print(f'  [{i}/{len(clean)}] {s} {r}', flush=True)
            time.sleep(1.4)
        else:
            fail += 1
            print(f'  [{i}/{len(clean)}] {s} → {r}', flush=True)
            time.sleep(2)
    print(f'\n完成：新增 {ok}、已有 {cached}、失敗 {fail}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
