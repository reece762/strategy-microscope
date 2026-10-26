#!/usr/bin/env python3
"""OpenD 連線與資料品質探測（唯讀）。

目的：在把 327 支股票池的抓取工作交給 OpenD 之前，先確認三件事
  1. 登入狀態（listening ≠ logged in；未登入時呼叫會 20 秒逾時）
  2. 能不能拿到真實報價（不是空值或 0）
  3. 歷史日 K 的欄位與覆蓋率（額外欄位以 0.0 表示缺值，不是 NaN）
"""
import sys

from futu import RET_OK, OpenQuoteContext

HOST, PORT = '127.0.0.1', 11111


def main():
    c = OpenQuoteContext(host=HOST, port=PORT)
    try:
        r, d = c.get_global_state()
        if r == RET_OK and isinstance(d, dict):
            print('登入狀態 :', d.get('qot_logined'))
            print('程式狀態 :', d.get('program_status_type'), d.get('program_status_desc'))
            print('市場狀態 :', {k: v for k, v in d.items()
                             if 'market' in k.lower() and 'state' in k.lower()})
        else:
            print('get_global_state 失敗 :', d)
            return 1

        # --- 報價 ---
        codes = ['US.AAPL', 'US.MSFT', 'US.NVDA', 'US.BRK.B']
        r, s = c.get_market_snapshot(codes)
        if r == RET_OK:
            print(f'\n快照 {len(s)} 筆')
            cols = [x for x in ('code', 'last_price', 'prev_close_price',
                                'total_market_val', 'update_time') if x in s.columns]
            print(s[cols].to_string(index=False))
        else:
            print('快照失敗 :', s)

        # --- 歷史日 K ---
        print('\n--- 歷史日 K（AAPL，前復權）---')
        r, data, page = c.request_history_kline(
            'US.AAPL', start='2025-01-01', end='2025-12-31', autype='qfq')
        if r == RET_OK:
            print(f'  筆數 {len(data)}  page_req_key={page}')
            print('  欄位:', list(data.columns))
            if len(data):
                print(data.head(3).to_string(index=False))
                print('  ...')
                print(data.tail(2).to_string(index=False))
                # 覆蓋率：非零比例
                print('  非零覆蓋率:')
                for col in data.columns:
                    try:
                        nz = (data[col].fillna(0) != 0).mean()
                        print(f'    {col:<20} {nz*100:5.1f}%')
                    except Exception:
                        pass
        else:
            print('  失敗 :', data)

        # --- 額度 ---
        r, q = c.get_history_kl_quota(get_detail=False)
        print('\n歷史 K 額度 :', q if r == RET_OK else f'查詢失敗 {q}')

        # --- 長時間區間測試（25 年）---
        print('\n--- 25 年區間（確認可回溯到多早）---')
        r, data, page = c.request_history_kline(
            'US.AAPL', start='2000-01-01', end='2025-10-24',
            autype='qfq', max_count=1000)
        if r == RET_OK:
            print(f'  本次回傳 {len(data)} 筆，還有下一頁: {page is not None}')
            if len(data):
                print(f'  區間 {data["time_key"].iloc[0]} ~ {data["time_key"].iloc[-1]}')
        else:
            print('  失敗 :', data)
        return 0
    finally:
        c.close()


if __name__ == '__main__':
    sys.exit(main())
