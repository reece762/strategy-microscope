#!/usr/bin/env python3
"""Assemble aligned daily series for every card.

Each card compares ONE strategy against benchmarks appropriate to its market:
  * US cards      -> 納斯達克綜合指數, SPY, 道瓊斯
  * crypto cards  -> BTC, ETH

Alignment rule: the strategy's own trading calendar is the master. Benchmarks are
reindexed onto it (forward-filled across holidays that differ between markets,
e.g. crypto trades on US holidays). All series are rebased to 1.0 at the first
date where every benchmark is available, so the chart compares like with like.

Dates are stored as days since 1970-01-01 to keep the payload small.
"""
import glob, io, json, os, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
EPOCH = pd.Timestamp('1970-01-01').normalize()


def fred(sid):
    df = pd.read_csv(os.path.join(DATA, f'{sid}.csv'))
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().set_index('date').sort_index()['close']


def binance(sym):
    """Daily close from the monthly klines archived at data.binance.vision."""
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, f'{sym}-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    # Binance switched to microsecond open_time partway through the archive, so
    # the unit has to be detected per row: 16-digit values are microseconds.
    ts = ts.where(ts < 1e14, ts / 1000.0)
    d['date'] = pd.to_datetime(ts, unit='ms')
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    return (d.dropna(subset=['date', 'close'])
             .drop_duplicates('date').set_index('date').sort_index()['close'])


def to_days(idx):
    return [(pd.Timestamp(x).normalize() - EPOCH).days for x in idx]


US_BENCH = {
    'nasdaq': dict(label='納斯達克', color='#58a6ff'),
    'spy': dict(label='SPY', color='#3fb950'),
    'djia': dict(label='道瓊斯', color='#d29922'),
}
CRYPTO_BENCH = {
    'btc': dict(label='BTC', color='#f7931a'),
    'eth': dict(label='ETH', color='#8c8cff'),
}


def build(name, strat, bench, label, ppy, meta):
    """Align benchmarks onto the strategy calendar and rebase to 1.0."""
    frame = pd.DataFrame({label: strat}).join(bench, how='left')
    frame = frame.ffill().dropna()   # drop leading rows where a benchmark is missing
    if len(frame) < 60:
        raise SystemExit(f'{name}: only {len(frame)} aligned rows')
    frame = frame / frame.iloc[0]
    return dict(
        id=name, label=label, ppy=ppy,
        d=to_days(frame.index),
        series=[dict(id='strat', label=label, color='#f85149', ppy=ppy,
                     v=[round(float(x), 4) for x in frame[label]])]
        + [dict(id=k, label=v['label'], color=v['color'], ppy=ppy,
                v=[round(float(x), 4) for x in frame[v['label']]])
           for k, v in meta.items()],
        start=str(frame.index[0].date()), end=str(frame.index[-1].date()),
    )


if __name__ == '__main__':
    res = json.load(open(os.path.join(OUT, 'results.json')))
    extra = json.load(open(os.path.join(OUT, 'extra.json')))
    cur = res['curves']

    nas = fred('NASDAQCOM')
    djia = fred('DJIA')
    spy = pd.Series(cur['SPY（2018 起）']['bh'],
                    index=pd.to_datetime(cur['SPY（2018 起）']['idx']))
    us_bench = pd.DataFrame({'納斯達克': nas, 'SPY': spy, '道瓊斯': djia}).sort_index()

    btc = binance('BTCUSDT')
    eth = binance('ETHUSDT')
    crypto_bench = pd.DataFrame({'BTC': btc, 'ETH': eth}).sort_index()
    print(f'BTC {btc.index[0].date()} → {btc.index[-1].date()}  ({len(btc)} 日)')
    print(f'ETH {eth.index[0].date()} → {eth.index[-1].date()}  ({len(eth)} 日)')

    def s_of(key, src):
        return pd.Series(src[key], index=pd.to_datetime(src['idx']))

    # cid -> (series, label, ppy, benchmark set)
    specs = {
        'wd-us':      (s_of('v', extra['wdonly_all']['mon']), '只做週一', 252, 'us'),
        'skip-nas':   (s_of('skip', cur['納斯達克綜合指數（1971 起）']), '跳過週四（納斯達克）', 252, 'us'),
        'skip-etf':   (s_of('skip', cur['SPY（2018 起）']), '跳過週四（SPY）', 252, 'us'),
        'regime':     (s_of('regime', cur['QQQ（2018 起）']), '均線×星期（QQQ）', 252, 'us'),
        'btc-skip':   (s_of('skip', cur['BTC（2017 起）']), '跳過週四（BTC）', 365, 'crypto'),
        'btc-regime': (s_of('regime', cur['BTC（2017 起）']), '均線×星期（BTC）', 365, 'crypto'),
        'calendar':   (s_of('v', extra['avoid_dec']['eq']), '避開 12/10 後 5 日', 252, 'us'),
    }

    sets = {'us': (us_bench, US_BENCH), 'crypto': (crypto_bench, CRYPTO_BENCH)}
    cards = {}
    for cid, (s, label, ppy, kind) in specs.items():
        bench, meta = sets[kind]
        cards[cid] = build(cid, s, bench, label, ppy, meta)
        c = cards[cid]
        tail = '  '.join(f"{x['label']} {x['v'][-1]:5.2f}x" for x in c['series'][1:])
        print(f'  {cid:12s} [{kind:6s}] {c["start"]} → {c["end"]}  {len(c["d"]):5d} 日  '
              f'策略 {c["series"][0]["v"][-1]:7.2f}x   {tail}')

    json.dump(dict(cards=cards), open(os.path.join(OUT, 'series.json'), 'w'),
              separators=(',', ':'))
    print(f'\nwritten out/series.json  '
          f'{os.path.getsize(os.path.join(OUT, "series.json"))/1024:.0f} KB')
