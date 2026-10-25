"""Detail-page template and the interactive date-range chart block.

Kept out of build_page.py because the embedded JavaScript contains braces and
quotes that fight with Python string interpolation in a single file. Templates
are plain strings filled by .replace(), so `{` and `}` need no escaping.
"""

DETAIL_TEMPLATE = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 詳細資料</title>
<style>__CSS__</style></head>
<body><div class="wrap"><div class="prose">
<a class="back" href="index.html">← 回到策略列表</a>
<div class="tags"><span class="tag">__TOPIC__</span>
<span class="tag vid">__VIDEO__</span>
<span class="tag author">__AUTHOR__</span>
<span class="grade" style="--gc:__COLOR__">__LETTER__</span></div>
<h1>__TITLE__</h1>
<p class="lead">__LEAD__</p>
__VLINK__
__CHART__
<div class="metrics">__METRICS__</div>
<div class="body">__BODY__</div>
__IMGS__
<div class="verdict"><b>判定</b>：__VERDICT__</div>
<div class="evid">依據：__EVID__</div>
__CODE__
<div class="build">版本 __VERSION__ · 建置於 __BUILD_AT__</div>
<a class="back bottom" href="index.html">← 回到策略列表</a>
</div></div></body></html>'''


CHART_BLOCK = r'''
<div class="chartcard">
  <div class="chartctl">
    <div class="daterange">
      <label>開始<input type="date" id="d0" value="__START__" min="__START__" max="__END__"></label>
      <span class="dash">—</span>
      <label>結束<input type="date" id="d1" value="__END__" min="__START__" max="__END__"></label>
    </div>
    <div class="quick">
      <button data-span="1">1 年</button>
      <button data-span="3">3 年</button>
      <button data-span="5">5 年</button>
      <button data-span="0" class="on">全部</button>
    </div>
  </div>
  <div class="legend" id="legend"></div>
  <div class="canvaswrap">
    <canvas id="chart"></canvas>
    <div id="tip" class="tip"></div>
  </div>
  <div class="tabwrap">
    <table id="stats">
      <thead><tr>
        <th>曲線</th><th>期間報酬</th><th>年化報酬</th>
        <th>最大回撤</th><th>年化波動</th><th>Sharpe</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="chartnote" id="note"></div>
</div>
<script>
var DATA = __DATA__;
var DAY = 86400000;
var iso = function (n) { return new Date(n * DAY).toISOString().slice(0, 10); };
var parseDay = function (s) {
  var p = s.split('-');
  return Math.floor(Date.UTC(+p[0], +p[1] - 1, +p[2]) / DAY);
};

var d0el = document.getElementById('d0'), d1el = document.getElementById('d1');
var cv = document.getElementById('chart'), ctx = cv.getContext('2d');
var tip = document.getElementById('tip'), legend = document.getElementById('legend');
var hoverIdx = -1, curIdx = null, curXs = null;

DATA.series.forEach(function (sr) {
  var el = document.createElement('span');
  el.className = 'lg';
  el.innerHTML = '<i style="background:' + sr.color + '"></i>' + sr.label;
  legend.appendChild(el);
});

function windowIdx() {
  var a = parseDay(d0el.value), b = parseDay(d1el.value), out = [];
  for (var i = 0; i < DATA.d.length; i++) if (DATA.d[i] >= a && DATA.d[i] <= b) out.push(i);
  return out;
}

function calcStats(v, ppy) {
  var r = [], i;
  for (i = 1; i < v.length; i++) r.push(v[i] / v[i - 1] - 1);
  var n = r.length;
  if (!n) return null;
  var mean = 0;
  for (i = 0; i < n; i++) mean += r[i];
  mean /= n;
  var acc = 0;
  for (i = 0; i < n; i++) acc += Math.pow(r[i] - mean, 2);
  var sd = Math.sqrt(acc / Math.max(n - 1, 1));
  var peak = v[0], mdd = 0;
  for (i = 0; i < v.length; i++) {
    if (v[i] > peak) peak = v[i];
    if (v[i] / peak - 1 < mdd) mdd = v[i] / peak - 1;
  }
  var total = v[n] / v[0] - 1;
  var years = n / ppy;
  return {
    total: total,
    cagr: Math.pow(1 + total, 1 / years) - 1,
    mdd: mdd,
    vol: sd * Math.sqrt(ppy),
    sharpe: sd ? (mean / sd) * Math.sqrt(ppy) : 0
  };
}

function draw() {
  var idx = windowIdx();
  var wrap = cv.parentElement;
  var w = wrap.clientWidth;
  var h = Math.max(300, Math.min(540, w * 0.46));
  var dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  if (idx.length < 2) {
    ctx.fillStyle = '#6e7681'; ctx.font = '13px sans-serif';
    ctx.fillText('此區間資料不足，請放寬日期範圍', 16, 40);
    return;
  }

  var padL = 64, padR = 16, padT = 16, padB = 32;
  var first = idx[0], last = idx[idx.length - 1];
  var xs = function (i) { return padL + (i - first) / (last - first) * (w - padL - padR); };

  var lo = Infinity, hi = -Infinity, k, j, v;
  var norm = [];
  for (k = 0; k < DATA.series.length; k++) {
    var base = DATA.series[k].v[first];
    var arr = [];
    for (j = 0; j < idx.length; j++) {
      v = DATA.series[k].v[idx[j]] / base;
      arr.push(v);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    norm.push(arr);
  }
  if (!isFinite(lo) || lo <= 0) lo = 0.5;
  var llo = Math.log(lo) / Math.LN10 - 0.05;
  var lhi = Math.log(hi) / Math.LN10 + 0.05;
  var ys = function (val) {
    return padT + (lhi - Math.log(val) / Math.LN10) / (lhi - llo) * (h - padT - padB);
  };

  ctx.strokeStyle = '#1e242d'; ctx.lineWidth = 1;
  ctx.font = '11px -apple-system,sans-serif';
  var ticks = [], e, m;
  for (e = Math.floor(llo); e <= Math.ceil(lhi); e++) {
    var cands = [1, 2, 3, 5];
    for (m = 0; m < 4; m++) {
      var val = cands[m] * Math.pow(10, e);
      if (val >= Math.pow(10, llo) && val <= Math.pow(10, lhi)) ticks.push(val);
    }
  }
  if (ticks.length > 9) ticks = ticks.filter(function (x, i) { return i % 2 === 0; });
  ticks.forEach(function (val) {
    var y = ys(val);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillStyle = '#6e7681';
    ctx.textAlign = 'right';
    ctx.fillText(val >= 1 ? (val >= 10 ? val.toFixed(0) : val.toFixed(1)) + 'x'
                          : val.toFixed(2) + 'x', padL - 10, y + 3.5);
  });
  ctx.textAlign = 'left';

  var y0 = DATA.d[first] / 365.25, y1 = DATA.d[last] / 365.25;
  var span = y1 - y0;
  var step = span > 25 ? 10 : span > 12 ? 5 : span > 5 ? 2 : 1;
  for (j = 0; j < idx.length; j++) {
    var dt = new Date(DATA.d[idx[j]] * DAY);
    if (dt.getUTCMonth() === 0 && dt.getUTCDate() <= 2 &&
        dt.getUTCFullYear() % step === 0) {
      var x = xs(idx[j]);
      ctx.strokeStyle = '#1a2028';
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke();
      ctx.fillStyle = '#6e7681';
      ctx.fillText(String(dt.getUTCFullYear()), x - 13, h - padB + 16);
    }
  }

  for (k = 0; k < DATA.series.length; k++) {
    ctx.beginPath();
    for (j = 0; j < idx.length; j++) {
      var px = xs(idx[j]), py = ys(norm[k][j]);
      if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = DATA.series[k].color;
    ctx.lineWidth = k === 0 ? 2.2 : 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  if (hoverIdx >= 0 && hoverIdx < idx.length) {
    var hx = xs(idx[hoverIdx]);
    ctx.strokeStyle = '#4a5568'; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, h - padB); ctx.stroke();
    ctx.setLineDash([]);
    for (k = 0; k < DATA.series.length; k++) {
      ctx.beginPath();
      ctx.arc(hx, ys(norm[k][hoverIdx]), 3.4, 0, 7);
      ctx.fillStyle = DATA.series[k].color;
      ctx.fill();
    }
  }

  var tb = document.querySelector('#stats tbody');
  tb.innerHTML = '';
  var pct = function (x) { return (x * 100).toFixed(1) + '%'; };
  for (k = 0; k < DATA.series.length; k++) {
    var sr = DATA.series[k];
    var sb = sr.v[first], sv = [];
    for (j = 0; j < idx.length; j++) sv.push(sr.v[idx[j]] / sb);
    var st = calcStats(sv, sr.ppy);
    if (!st) continue;
    var tr = document.createElement('tr');
    if (k === 0) tr.className = 'hi';
    tr.innerHTML = '<td><i style="background:' + sr.color + '"></i>' + sr.label + '</td>' +
      '<td>' + pct(st.total) + '</td><td>' + pct(st.cagr) + '</td>' +
      '<td class="neg">' + pct(st.mdd) + '</td><td>' + pct(st.vol) + '</td>' +
      '<td>' + st.sharpe.toFixed(2) + '</td>';
    tb.appendChild(tr);
  }

  var s0 = DATA.series[0];
  var a0 = s0.v[last] / s0.v[first];
  var parts = [];
  for (k = 1; k < DATA.series.length; k++) {
    var bk = DATA.series[k];
    var ab = bk.v[last] / bk.v[first];
    parts.push(bk.label + ' ' + ((a0 / ab - 1) * 100).toFixed(1) + '%');
  }
  document.getElementById('note').textContent =
    '區間 ' + iso(DATA.d[first]) + ' 至 ' + iso(DATA.d[last]) +
    '（' + idx.length + ' 個交易日）。策略相對各基準的累積超額報酬：' + parts.join('、') + '。';

  curIdx = idx; curXs = xs;
}

cv.addEventListener('mousemove', function (e) {
  if (!curIdx) return;
  var r = cv.getBoundingClientRect();
  var x = e.clientX - r.left;
  var best = -1, bd = 1e9, j, d;
  for (j = 0; j < curIdx.length; j++) {
    d = Math.abs(curXs(curIdx[j]) - x);
    if (d < bd) { bd = d; best = j; }
  }
  hoverIdx = best;
  if (best >= 0) {
    var i = curIdx[best];
    var html = '<b>' + iso(DATA.d[i]) + '</b>';
    for (var k = 0; k < DATA.series.length; k++) {
      var sr = DATA.series[k];
      var p = (sr.v[i] / sr.v[curIdx[0]] - 1) * 100;
      html += '<div><i style="background:' + sr.color + '"></i>' + sr.label +
              '<span>' + p.toFixed(1) + '%</span></div>';
    }
    tip.innerHTML = html;
    tip.style.display = 'block';
    var tw = tip.offsetWidth;
    var left = x + 14;
    if (left + tw > cv.clientWidth) left = x - tw - 14;
    tip.style.left = Math.max(4, left) + 'px';
    tip.style.top = '12px';
  }
  draw();
});

cv.addEventListener('mouseleave', function () {
  hoverIdx = -1;
  tip.style.display = 'none';
  draw();
});

function onDateChange(el) {
  if (parseDay(d0el.value) > parseDay(d1el.value)) {
    if (el === d0el) d1el.value = d0el.value; else d0el.value = d1el.value;
  }
  var qs = document.querySelectorAll('.quick button');
  for (var i = 0; i < qs.length; i++) qs[i].classList.remove('on');
  draw();
}
d0el.addEventListener('change', function () { onDateChange(d0el); });
d1el.addEventListener('change', function () { onDateChange(d1el); });

var qbtns = document.querySelectorAll('.quick button');
for (var qi = 0; qi < qbtns.length; qi++) {
  qbtns[qi].onclick = (function (btn) {
    return function () {
      var all = document.querySelectorAll('.quick button');
      for (var i = 0; i < all.length; i++) all[i].classList.remove('on');
      btn.classList.add('on');
      var yrs = Number(btn.getAttribute('data-span'));
      if (!yrs) {
        d0el.value = d0el.min;
        d1el.value = d1el.max;
      } else {
        var endDay = parseDay(d1el.max);
        var startDay = Math.max(parseDay(d0el.min), endDay - Math.round(yrs * 365.25));
        d0el.value = iso(startDay);
        d1el.value = d1el.max;
      }
      draw();
    };
  })(qbtns[qi]);
}

window.addEventListener('resize', function () { draw(); });
draw();
</script>
'''


CHART_CSS = '''
.chartcard{background:#0d1219;border:1px solid var(--line);border-radius:11px;
padding:16px 16px 14px;margin:0 0 24px}
.chartctl{display:flex;justify-content:space-between;align-items:flex-end;
gap:14px;flex-wrap:wrap;margin-bottom:12px}
.daterange{display:flex;align-items:flex-end;gap:9px}
.daterange label{display:flex;flex-direction:column;gap:4px;font-size:11.5px;color:var(--tx3)}
.daterange input[type=date]{background:#161b22;border:1px solid var(--line);
color:var(--tx);border-radius:6px;padding:6px 9px;font-size:13px;
font-family:ui-monospace,monospace;color-scheme:dark}
.daterange input[type=date]:focus{outline:none;border-color:var(--acc)}
.dash{color:var(--tx3);padding-bottom:8px;font-size:12px}
.quick{display:flex;gap:5px}
.quick button{background:transparent;border:1px solid var(--line);color:var(--tx3);
border-radius:6px;padding:5px 11px;font-size:12px;cursor:pointer;transition:.15s}
.quick button:hover{color:var(--tx);border-color:#4a5568}
.quick button.on{background:var(--acc);border-color:var(--acc);color:#0d1117;font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;font-size:12px;color:#c9d1d9}
.legend .lg{display:flex;align-items:center;gap:6px}
.legend i,.tabwrap i{display:inline-block;width:10px;height:3px;border-radius:2px}
.canvaswrap{position:relative}
#chart{display:block;width:100%}
.tip{position:absolute;display:none;background:#161b22;border:1px solid #30363d;
border-radius:7px;padding:8px 11px;font-size:12px;color:#c9d1d9;pointer-events:none;
z-index:5;box-shadow:0 6px 20px rgba(0,0,0,.5);line-height:1.65;white-space:nowrap}
.tip b{color:#fff;font-size:11.5px;font-family:ui-monospace,monospace}
.tip div{display:flex;align-items:center;gap:6px;margin-top:3px}
.tip div i{width:9px;height:3px;border-radius:2px}
.tip div span{margin-left:auto;padding-left:12px;font-family:ui-monospace,monospace;color:#fff}
.tabwrap{margin-top:16px;overflow-x:auto}
#stats{width:100%;border-collapse:collapse;font-size:12.5px}
#stats th{text-align:right;color:var(--tx3);font-weight:500;font-size:11px;
padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
#stats th:first-child{text-align:left}
#stats td{text-align:right;padding:9px;border-bottom:1px solid #1c232c;
font-family:ui-monospace,monospace;color:#c9d1d9;white-space:nowrap}
#stats td:first-child{text-align:left;font-family:inherit}
#stats td:first-child i{margin-right:7px;vertical-align:middle}
#stats tr.hi td{background:#151d29;color:#fff;font-weight:600}
#stats td.neg{color:#f0883e}
.chartnote{margin-top:10px;font-size:11.5px;color:var(--tx3);line-height:1.7}
.nochart{background:#0d1219;border:1px dashed var(--line);border-radius:10px;
padding:26px;text-align:center;color:var(--tx3);font-size:13px;margin:0 0 24px}
@media(max-width:760px){
  .chartctl{flex-direction:column;align-items:stretch}
  .daterange{flex-wrap:wrap}
  .daterange input[type=date]{flex:1;min-width:0}
  .quick{justify-content:flex-start}
  #stats{font-size:11.5px}
  #stats th,#stats td{padding:7px 5px}
}
'''


CODE_BLOCK = r'''
<div class="codecard">
  <button class="codetoggle" id="codetoggle" aria-expanded="false" aria-controls="codewrap">
    <span class="caret">▾</span> 顯示代碼
  </button>
  <div class="codewrap" id="codewrap" hidden>
    <div class="codeinfo">
      <span class="codeinfo-txt">__CODE_TITLE__</span>
      <button class="copybtn" id="copybtn" type="button" title="複製代碼" aria-label="複製代碼">
        <svg class="ico-copy" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
          <path fill="currentColor" d="M4 1.5A1.5 1.5 0 0 1 5.5 0h5A1.5 1.5 0 0 1 12 1.5v1h.5A1.5 1.5 0 0 1 14 4v10.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 14.5V4a1.5 1.5 0 0 1 1.5-1.5H4v-1Zm1.5.5v1h5v-1h-5ZM3.5 4a.5.5 0 0 0-.5.5v10a.5.5 0 0 0 .5.5h9a.5.5 0 0 0 .5-.5V4.5a.5.5 0 0 0-.5-.5h-9Z"/>
        </svg>
        <svg class="ico-ok" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
          <path fill="currentColor" d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.5 6.5a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 1 1 1.06-1.06l2.47 2.47 5.97-5.97a.75.75 0 0 1 1.06 0Z"/>
        </svg>
        <span class="copytext">複製</span>
      </button>
    </div>
    <pre><code>__CODE__</code></pre>
    <div class="codefoot">以上是跑出本頁數字所用的實際函式原始碼，由建置腳本自動擷取，不是另行重寫的版本。</div>
  </div>
</div>
<script>
(function () {
  var btn = document.getElementById('codetoggle');
  var box = document.getElementById('codewrap');
  if (!btn || !box) return;
  btn.addEventListener('click', function () {
    var open = box.hidden;
    box.hidden = !open;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.classList.toggle('open', open);
  });

  var cbtn = document.getElementById('copybtn');
  var code = document.querySelector('#codewrap code');
  if (!cbtn || !code) return;
  var timer = null;
  cbtn.addEventListener('click', function () {
    var text = code.textContent;
    var done = function () {
      cbtn.classList.add('ok');
      cbtn.querySelector('.copytext').textContent = '已複製';
      clearTimeout(timer);
      timer = setTimeout(function () {
        cbtn.classList.remove('ok');
        cbtn.querySelector('.copytext').textContent = '複製';
      }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
    } else {
      fallback(text, done);
    }
  });

  /* Older mobile browsers and non-secure origins reject the async clipboard
     API, so fall back to a hidden textarea + execCommand. */
  function fallback(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, text.length);
    try { document.execCommand('copy'); done(); } catch (e) {}
    document.body.removeChild(ta);
  }
})();
</script>
'''


CODE_CSS = '''
.codecard{margin-top:20px;border-top:1px solid var(--line);padding-top:18px}
.codetoggle{display:inline-flex;align-items:center;gap:7px;background:#161b22;
border:1px solid var(--line);color:var(--tx2);border-radius:8px;padding:9px 16px;
font-size:13px;cursor:pointer;transition:.15s;font-family:inherit}
.codetoggle:hover{color:#fff;border-color:var(--acc)}
.codetoggle.open{color:var(--acc);border-color:var(--acc)}
.codetoggle .caret{font-size:10px;display:inline-block;transition:transform .18s}
.codetoggle.open .caret{transform:rotate(180deg)}
.codewrap{margin-top:14px;border:1px solid var(--line);border-radius:9px;
background:#0a0e14;overflow:hidden}
.codeinfo{display:flex;align-items:flex-start;gap:12px;font-size:11.5px;color:#d29922;
background:#141a22;border-bottom:1px solid var(--line);padding:9px 14px;
font-family:ui-monospace,monospace;line-height:1.65}
.codeinfo-txt{flex:1 1 auto;white-space:pre-wrap;min-width:0}
.copybtn{flex:0 0 auto;display:inline-flex;align-items:center;gap:6px;
background:#1b232d;border:1px solid var(--line);color:var(--tx2);border-radius:6px;
padding:5px 10px;font-size:11px;cursor:pointer;transition:.15s;font-family:inherit;
margin:-2px -2px -2px 0}
.copybtn:hover{color:#fff;border-color:var(--acc);background:#233040}
.copybtn:active{transform:scale(.96)}
.copybtn .ico-ok,.copybtn.ok .ico-copy{display:none}
.copybtn.ok .ico-ok{display:inline-block}
.copybtn.ok{color:#3fb950;border-color:#2d5c3a;background:#16261c}
.copybtn .copytext{white-space:nowrap}
.codewrap pre{margin:0;padding:14px 16px;overflow-x:auto;background:#0a0e14}
.codewrap code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:12.2px;line-height:1.75;color:#c9d1d9;white-space:pre}
.codefoot{font-size:11px;color:var(--tx3);padding:9px 14px;border-top:1px solid var(--line);
background:#0d1219;line-height:1.7}
@media(max-width:760px){
  .codewrap code{font-size:11px}
  .codetoggle{width:100%;justify-content:center}
}
'''
