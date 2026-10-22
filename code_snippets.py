"""Pull the exact source of the strategy functions that produced the results.

Rendered into the detail pages behind a 顯示代碼 toggle. Using inspect.getsource
means the panel can never drift from the code that actually ran — if a parameter
changes in run.py/btc.py/extra.py, the page changes with it.

Each entry gets a short header naming the file, the function and the cost /
look-ahead convention, because a bare function body invites the reader to assume
defaults that were overridden at the call site.
"""
import inspect

import extra
import run
import btc

# card id -> (module, function name, note)
MAP = {
    'wd-us':      (extra, 'weekday_only', '只做週一：整個回測期間只有週一持倉，其餘平倉。'),
    'skip-nas':   (run, 'skip_thursday', '跳過週四，其餘每天持有。美股成本每趟 1 bp。'),
    'skip-etf':   (run, 'skip_thursday', '同上，改用在 SPY／QQQ 日線上。'),
    'regime':     (run, 'regime_strategy', '均線×星期：用前一日收盤與 MA20 判斷多空，再依星期決定持倉。'),
    'btc-skip':   (btc, 'skip_day', '跳過週四，其餘每天持有。加密成本每趟 5 bp。'),
    'btc-regime': (btc, 'regime', '均線×星期，日曆為 7 天（含週末），加密成本每趟 5 bp。'),
    'calendar':   (extra, 'avoid_windows', '每年 12/10 起連續 5 個交易日平倉，其餘時間持有。'),
}

PRELUDE = '''# 執行環境：Python 3 + pandas / numpy
# 資料：日線收盤價，欄位 close
# 重點：第 i 根的持倉只用第 i-1 根收盤前的資訊決定，無前視（no look-ahead）
# 成本：美股每趟 1 bp（萬分之一），加密每趟 5 bp
'''


def snippet(card_id):
    """Return (title, code) for a card, or None when the strategy has no code."""
    item = MAP.get(card_id)
    if not item:
        return None
    mod, fn, note = item
    src = inspect.getsource(getattr(mod, fn))
    title = f'{mod.__name__}.py · {fn}()\n{note}'
    return title, PRELUDE + '\n' + src.rstrip() + '\n'


if __name__ == '__main__':
    for cid in MAP:
        got = snippet(cid)
        title, code = got
        print('=' * 70)
        print(cid, '|', title.splitlines()[0])
        print(code[:300])
        print(f'... ({len(code)} chars)')
