"""
Ostap: This is an LLM-generated wrapper and report provider for the purpose of just running experiments using already
existing code and features to do so.

report_generator.py  —  Self-contained HTML report from experiment results.

Index page (report.html):
  - Sortable comparison table with ALL metrics
  - Per-metric overlay plots (all experiments on one chart) for ALL 8 metrics
  - Gaussian count + step timing overlay
  - Inline filmstrip thumbnails per experiment (pred evolution over time)
  - Links to per-experiment detail pages

Per-experiment pages (<name>/report.html):
  - All 8 metric plots (3D + 2D)
  - Full checkpoint metrics table
  - Multi-camera filmstrip with ref/pred side by side
  - Image history: pred-only strip per camera showing training progress
  - Health events, step timing, full config
"""

from __future__ import annotations

import base64
import json
import math
from datetime import timedelta
from pathlib import Path

_PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------
_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #1e1e2e; color: #cdd6f4;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px; padding: 24px; max-width: 1600px; margin: 0 auto;
}
h1 { color: #89b4fa; font-size: 1.6rem; margin-bottom: 6px; }
h2 { color: #cba6f7; font-size: 1.1rem; margin: 32px 0 10px;
     border-bottom: 1px solid #313244; padding-bottom: 6px; }
h3 { color: #89dceb; font-size: 0.95rem; margin: 16px 0 6px; }
.sub  { color: #6c7086; font-size: 13px; margin-bottom: 18px; }
.badge { display:inline-block; padding:2px 10px; border-radius:4px;
  font-size:12px; font-weight:600; }
.badge-ok   { background:#a6e3a122; color:#a6e3a1; border:1px solid #a6e3a1; }
.badge-warn { background:#f9e2af22; color:#f9e2af; border:1px solid #f9e2af; }
.badge-err  { background:#f38ba822; color:#f38ba8; border:1px solid #f38ba8; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
.grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.card { background:#181825; border:1px solid #313244; border-radius:8px; padding:16px; }
.kv-val { font-size:1.5rem; font-weight:700; color:#89b4fa; }
.kv-lbl { font-size:11px; color:#6c7086; margin-top:2px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#181825; color:#6c7086; padding:7px 10px;
     text-align:left; position:sticky; top:0; white-space:nowrap; }
td { padding:6px 10px; border-bottom:1px solid #1e1e2e; white-space:nowrap; }
tr:hover td { background:#1e1e3a; }
a { color:#89b4fa; text-decoration:none; }
a:hover { text-decoration:underline; }
.back { margin-bottom:18px; font-size:13px; }
.scroll-x { overflow-x:auto; }
/* Filmstrip */
.fs-wrap { overflow-x:auto; padding-bottom:6px; }
.fs-table { border-collapse:collapse; }
.fs-label { writing-mode:vertical-rl; color:#89b4fa; font-size:11px;
            padding:4px 2px; white-space:nowrap; vertical-align:middle;
            background:#181825; border-right:1px solid #313244; }
.fs-cell { padding:3px 5px; border-right:1px solid #181825;
           text-align:center; vertical-align:top; }
.fs-step { color:#6c7086; font-size:10px; margin-bottom:2px; }
.fs-img  { display:block; border-radius:3px; cursor:pointer;
           transition:transform .15s; }
.fs-img:hover { transform:scale(1.05); }
.fs-tag  { font-size:9px; color:#45475a; margin-top:1px; }
/* Lightbox */
#lb { display:none; position:fixed; inset:0; background:#000c;
      z-index:9999; align-items:center; justify-content:center; }
#lb.open { display:flex; }
#lb img  { max-width:90vw; max-height:90vh; border-radius:6px; }
#lb-close { position:fixed; top:16px; right:24px; font-size:2rem;
             color:#fff; cursor:pointer; line-height:1; }
/* Metrics table highlight */
.best { color:#a6e3a1; font-weight:700; }
.worst { color:#f38ba8; }
/* Exp section on index */
.exp-block { background:#181825; border:1px solid #313244;
             border-radius:8px; padding:16px; margin-bottom:20px; }
.exp-header { display:flex; align-items:baseline; gap:12px; margin-bottom:10px; }
.exp-name   { color:#89b4fa; font-size:1.05rem; font-weight:600; }
.exp-desc   { color:#6c7086; font-size:12px; }
"""

_LIGHTBOX = """
<div id="lb"><span id="lb-close" onclick="closeLB()">✕</span>
<img id="lb-img" src=""></div>
<script>
function openLB(src){document.getElementById('lb-img').src=src;
  document.getElementById('lb').classList.add('open');}
function closeLB(){document.getElementById('lb').classList.remove('open');}
document.getElementById('lb').addEventListener('click',function(e){
  if(e.target===this)closeLB();});
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLB();});
</script>
"""

# ---------------------------------------------------------------------------
# Palette + Plotly layout
# ---------------------------------------------------------------------------
_PAL = ["#89b4fa","#cba6f7","#a6e3a1","#fab387","#f9e2af",
        "#94e2d5","#f38ba8","#eba0ac","#b4befe","#74c7ec"]

_LAY = {
    "paper_bgcolor":"#1e1e2e","plot_bgcolor":"#181825",
    "font":{"color":"#cdd6f4","size":12},
    "xaxis":{"gridcolor":"#313244","title":"Step"},
    "yaxis":{"gridcolor":"#313244"},
    "margin":{"t":36,"b":36,"l":52,"r":10},
    "legend":{"bgcolor":"#18182500","font":{"size":11}},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{d}f}"

def _psnr_color(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "#6c7086"
    if v >= 30: return "#a6e3a1"
    if v >= 22: return "#f9e2af"
    return "#f38ba8"

def _badge(s):
    cls = "badge-ok" if s=="completed" else "badge-warn" if "interrupt" in s else "badge-err"
    return f'<span class="badge {cls}">{s}</span>'

def _b64(path):
    if not path or not Path(path).exists(): return None
    try:
        with open(path,"rb") as f: data=f.read()
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {"png":"image/png","jpg":"image/jpeg"}.get(ext,"image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except: return None

def _img(path, w=None, cls="fs-img", onclick=True):
    b = _b64(path)
    if not b: return '<span style="color:#45475a;font-size:10px">n/a</span>'
    style = f'width:{w}px;' if w else ''
    oc = f' onclick="openLB(this.src)"' if onclick else ''
    return f'<img src="{b}" class="{cls}" style="{style}"{oc}>'

def _load(exp_dir):
    p = Path(exp_dir)/"results.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except: return None

def _trace(x, y, name, color, dash="solid", width=2):
    cx = [xi for xi,yi in zip(x,y) if yi is not None]
    cy = [yi for yi in y if yi is not None]
    if not cx: return None
    return (f'{{"x":{json.dumps(cx)},"y":{json.dumps(cy)},'
            f'"mode":"lines+markers","name":{json.dumps(name)},'
            f'"line":{{"color":{json.dumps(color)},"dash":"{dash}","width":{width}}},'
            f'"marker":{{"size":4}}}}')

def _chart(uid, traces, title, h=240, ytitle=""):
    if not traces:
        return f'<p style="color:#45475a;padding:8px">No data — {title}</p>'
    lay = {**_LAY,"title":{"text":title,"font":{"size":13}},"height":h}
    if ytitle: lay["yaxis"] = {**lay.get("yaxis",{}),"title":ytitle}
    traces_js = "["+",".join(t for t in traces if t)+"]"
    return (f'<div id="{uid}"></div>'
            f'<script>Plotly.newPlot("{uid}",{traces_js},'
            f'{json.dumps(lay)},{{responsive:true,displayModeBar:false}});</script>')

# ---------------------------------------------------------------------------
# Filmstrip builder  (used on both index and exp pages)
# ---------------------------------------------------------------------------

def _filmstrip(checkpoints, exp_dir, thumb_w=120, show_ref=True):
    """
    Returns HTML for a horizontal filmstrip.
    Rows = cameras, columns = checkpoints.
    Each cell shows pred (and optionally ref) thumbnail.
    """
    cam_names, seen = [], set()
    for cp in checkpoints:
        for r in cp.get("renders",[]):
            c = r.get("camera","cam")
            if c not in seen: cam_names.append(c); seen.add(c)
    if not cam_names:
        return '<p style="color:#45475a">No renders saved.</p>'

    rows_html = ""
    for cam in cam_names:
        cells = f'<td class="fs-label">{cam}</td>'
        for cp in checkpoints:
            s = cp.get("step",0)
            rmap = {r["camera"]:r for r in cp.get("renders",[])}
            if cam not in rmap:
                cells += f'<td class="fs-cell"><div class="fs-step">step {s}</div>—</td>'
                continue
            r = rmap[cam]
            pred_p = Path(exp_dir)/r.get("pred","")
            ref_p  = Path(exp_dir)/r.get("ref","")
            cell = f'<div class="fs-step">step {s}</div>'
            if show_ref:
                cell += _img(ref_p,  thumb_w) + '<div class="fs-tag">ref</div>'
            cell += _img(pred_p, thumb_w) + '<div class="fs-tag">pred</div>'
            cells += f'<td class="fs-cell">{cell}</td>'
        rows_html += f"<tr>{cells}</tr>"

    return (f'<div class="fs-wrap"><table class="fs-table">'
            f'<tbody>{rows_html}</tbody></table></div>')


def _pred_history_strip(checkpoints, exp_dir, cam_name, thumb_w=100):
    """Pred-only strip for one camera — compact training progress view."""
    cells = ""
    for cp in checkpoints:
        s = cp.get("step",0)
        rmap = {r["camera"]:r for r in cp.get("renders",[])}
        if cam_name not in rmap: continue
        pred_p = Path(exp_dir)/rmap[cam_name].get("pred","")
        cells += (f'<td class="fs-cell">'
                  f'<div class="fs-step">{s}</div>'
                  f'{_img(pred_p, thumb_w)}</td>')
    if not cells: return ""
    return (f'<div class="fs-wrap"><table class="fs-table"><tbody>'
            f'<tr><td class="fs-label">{cam_name}</td>{cells}</tr>'
            f'</tbody></table></div>')

# ---------------------------------------------------------------------------
# Full metrics table
# ---------------------------------------------------------------------------

def _metrics_table(checkpoints):
    hdr = ("<tr><th>Step</th><th>Gaussians</th>"
           "<th>PSNR 3D</th><th>L1 3D</th><th>IoU 3D</th><th>SSIM 3D</th>"
           "<th>PSNR 2D</th><th>L1 2D</th><th>IoU 2D</th><th>SSIM 2D</th></tr>")
    rows = ""
    for cp in checkpoints:
        m3 = cp.get("metrics_3d") or {}
        m2 = cp.get("metrics_2d") or {}
        p3c = _psnr_color(m3.get("psnr"))
        p2c = _psnr_color(m2.get("psnr"))
        rows += (f"<tr>"
                 f"<td>{cp.get('step',0)}</td>"
                 f"<td>{cp.get('gaussian_count',0):,}</td>"
                 f"<td style='color:{p3c}'>{_fmt(m3.get('psnr'),2)}</td>"
                 f"<td>{_fmt(m3.get('l1'),5)}</td>"
                 f"<td>{_fmt(m3.get('iou'),4)}</td>"
                 f"<td>{_fmt(m3.get('ssim_mean'),4)}</td>"
                 f"<td style='color:{p2c}'>{_fmt(m2.get('psnr'),2)}</td>"
                 f"<td>{_fmt(m2.get('l1'),5)}</td>"
                 f"<td>{_fmt(m2.get('iou'),4)}</td>"
                 f"<td>{_fmt(m2.get('ssim'),4)}</td>"
                 f"</tr>")
    return (f'<div class="scroll-x"><table>'
            f'<thead>{hdr}</thead><tbody>{rows}</tbody></table></div>')

# ---------------------------------------------------------------------------
# Per-experiment page
# ---------------------------------------------------------------------------

def _exp_page(results, exp_dir):
    name     = results.get("name","unnamed")
    desc     = results.get("description","")
    status   = results.get("status","?")
    wall     = results.get("wall_time_seconds",0)
    final_g  = results.get("final_gaussian_count",0)
    cps      = results.get("checkpoints",[])
    health   = results.get("health_events",[])
    st_sum   = results.get("step_time_summary",{})
    st_samp  = results.get("step_time_samples",[])
    exp_dir  = Path(exp_dir)

    # --- Build metric series ---
    def _series(cps, tier, key):
        xs, ys = [], []
        for cp in cps:
            m = (cp.get(f"metrics_{tier}") or {})
            v = m.get(key)
            if v is not None: xs.append(cp["step"]); ys.append(v)
        return xs, ys

    uid = name.replace(" ","_")

    def _dual(uid, title, key3, key2, h=220):
        x3,y3 = _series(cps,"3d",key3); x2,y2 = _series(cps,"2d",key2)
        t1 = _trace(x3,y3,"3D Vol",_PAL[0]); t2 = _trace(x2,y2,"2D Screen",_PAL[1])
        return _chart(uid,[t for t in [t1,t2] if t],title,h)

    charts_r1 = (f'<div class="grid-2">'
                 f'{_dual(f"{uid}_psnr","PSNR (dB)","psnr","psnr")}'
                 f'{_dual(f"{uid}_l1","L1 / MAE","l1","l1")}'
                 f'</div>')
    charts_r2 = (f'<div class="grid-2" style="margin-top:12px">'
                 f'{_dual(f"{uid}_iou","IoU","iou","iou")}'
                 f'{_dual(f"{uid}_ssim","SSIM","ssim_mean","ssim")}'
                 f'</div>')

    # Gaussian count
    gx = [cp["step"] for cp in cps]; gy = [cp.get("gaussian_count") for cp in cps]
    g_chart = _chart(f"{uid}_g",[_trace(gx,gy,"Gaussians",_PAL[2])],
                     "Gaussian Count",200)

    # Step time
    if st_samp:
        sx=[s for s,_ in st_samp]; sy=[ms for _,ms in st_samp]
        st_chart = _chart(f"{uid}_st",[_trace(sx,sy,"ms/step",_PAL[3])],
                          "Step Time (ms)",200)
    else:
        st_chart = '<p style="color:#45475a">No step-time data.</p>'

    # KV cards
    def _last(tier, key):
        for cp in reversed(cps):
            v = (cp.get(f"metrics_{tier}") or {}).get(key)
            if v is not None: return v
        return None

    def _card(val, lbl):
        return (f'<div class="card"><div class="kv-val">{val}</div>'
                f'<div class="kv-lbl">{lbl}</div></div>')

    cards = "".join([
        _card(_fmt(_last("3d","psnr"),2),  "PSNR 3D (dB)"),
        _card(_fmt(_last("2d","psnr"),2),  "PSNR 2D (dB)"),
        _card(_fmt(_last("3d","ssim_mean")),"SSIM 3D"),
        _card(_fmt(_last("3d","iou")),      "IoU 3D"),
        _card(_fmt(_last("2d","ssim")),     "SSIM 2D"),
        _card(_fmt(_last("2d","iou")),      "IoU 2D"),
        _card(f"{final_g:,}",              "Final Gaussians"),
        _card(f"{st_sum.get('mean_ms','?')} ms","Avg Step"),
    ])

    # Filmstrip (ref + pred)
    fs = _filmstrip(cps, exp_dir, thumb_w=140, show_ref=True)

    # Pred-only history per camera
    cam_names = list({r["camera"] for cp in cps for r in cp.get("renders",[])})
    hist_strips = "".join(_pred_history_strip(cps, exp_dir, c, 110) for c in cam_names)

    # Health
    h_rows = "".join(
        f'<tr><td>{e.get("step","?")}</td><td style="color:#f38ba8">{e.get("type","?")}</td>'
        f'<td>{_fmt(e.get("grad_max"),4)}</td><td>{_fmt(e.get("grad_mean"),6)}</td></tr>'
        for e in health
    ) or '<tr><td colspan="4" style="color:#a6e3a1">No events</td></tr>'
    health_tbl = (f'<table><thead><tr><th>Step</th><th>Type</th>'
                  f'<th>Grad Max</th><th>Grad Mean</th></tr></thead>'
                  f'<tbody>{h_rows}</tbody></table>')

    # Step time summary table
    st_rows = "".join(
        f"<tr><td>{k}</td><td>{v} ms</td></tr>"
        for k,v in st_sum.items() if v is not None
    )

    # Config
    cfg = results.get("config", {})
    t = cfg.get("training",{}); sg = cfg.get("sgld",{}); ad = cfg.get("adc",{})
    cfg_rows = ""
    for sec, d in [("training",t),("sgld",sg),("adc",ad)]:
        for k,v in d.items():
            if k == "lr_schedule": continue
            cfg_rows += f"<tr><td style='color:#6c7086'>{sec}</td><td>{k}</td><td>{v}</td></tr>"
    for ev in t.get("lr_schedule",[]):
        cfg_rows += (f"<tr><td style='color:#6c7086'>lr_schedule</td>"
                     f"<td>@ step {ev.get('at_step','?')}</td>"
                     f"<td>{json.dumps({k:v for k,v in ev.items() if k!='at_step'})}</td></tr>")

    wall_s = str(timedelta(seconds=int(wall)))

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8">
<title>{name} — Experiment</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head><body>
{_LIGHTBOX}
<div class="back"><a href="../report.html">← Back to comparison</a></div>
<h1>{name}</h1>
<p class="sub">{desc}</p>
<p class="sub">
  {_badge(status)} &nbsp;|&nbsp; Wall: {wall_s} &nbsp;|&nbsp;
  Gaussians: {final_g:,} &nbsp;|&nbsp;
  Avg step: {st_sum.get('mean_ms','?')} ms &nbsp;|&nbsp;
  p95: {st_sum.get('p95_ms','?')} ms
</p>

<div class="grid-4" style="margin-bottom:20px">{cards}</div>

<h2>Metrics Over Training</h2>
{charts_r1}{charts_r2}

<h2>Gaussian Count &amp; Step Timing</h2>
<div class="grid-2">{g_chart}{st_chart}</div>

<h2>All Checkpoint Metrics</h2>
{_metrics_table(cps)}

<h2>Training Progress — Pred Evolution</h2>
<p class="sub" style="margin-bottom:8px">Prediction only across all checkpoints — click any image to enlarge</p>
{hist_strips}

<h2>Render Snapshots — Ref vs Pred</h2>
<p class="sub" style="margin-bottom:8px">Top row: reference · Bottom row: prediction — click to enlarge</p>
{fs}

<h2>Health Events</h2>
{health_tbl}

<h2>Step Timing Summary</h2>
<table style="width:auto"><thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{st_rows or "<tr><td colspan=2>—</td></tr>"}</tbody></table>

<h2>Configuration</h2>
<div class="scroll-x"><table>
<thead><tr><th>Section</th><th>Key</th><th>Value</th></tr></thead>
<tbody>{cfg_rows or "<tr><td colspan=3>Config not saved in results.json</td></tr>"}</tbody>
</table></div>

</body></html>"""

# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def _index_page(summaries, all_results, base_dir):
    base_dir = Path(base_dir)

    # ── Comparison table ──────────────────────────────────────────────────
    sorted_s = sorted(summaries,
                      key=lambda x: x.get("final_psnr_3d") or 0, reverse=True)

    # Determine best/worst per metric for highlighting
    def _best_worst(key):
        vals = [s.get(key) for s in sorted_s if s.get(key) is not None]
        if not vals: return None, None
        higher_better = "psnr" in key or "ssim" in key or "iou" in key
        return (max(vals), min(vals)) if higher_better else (min(vals), max(vals))

    metrics_keys = ["final_psnr_3d","final_l1_3d","final_iou_3d","final_ssim_3d",
                    "final_psnr_2d","final_l1_2d","final_iou_2d","final_ssim_2d"]
    bests  = {k: _best_worst(k)[0] for k in metrics_keys}
    worsts = {k: _best_worst(k)[1] for k in metrics_keys}

    def _cell(s, key, d=4):
        v = s.get(key)
        if v is None: return "<td>—</td>"
        cls = "best" if v == bests.get(key) else "worst" if v == worsts.get(key) else ""
        color = f"color:{_psnr_color(v)};" if "psnr" in key else ""
        return f'<td style="text-align:right;{color}" class="{cls}">{_fmt(v,d)}</td>'

    rows = ""
    for s in sorted_s:
        name   = s["name"]
        status = s.get("status","?")
        wall   = str(timedelta(seconds=int(s.get("wall_time_seconds",0))))
        ms     = s.get("step_time_summary",{}).get("mean_ms","?")
        gc     = s.get("final_gaussian_count")

        # Thumbnail — last checkpoint, first camera pred
        thumb = ""
        r = all_results.get(name,{})
        cps = r.get("checkpoints",[])
        if cps:
            for rend in cps[-1].get("renders",[]):
                b = _b64(base_dir/name/rend.get("pred",""))
                if b:
                    thumb = f'<img src="{b}" style="height:50px;border-radius:3px;vertical-align:middle;cursor:pointer" onclick="openLB(this.src)">'
                    break

        rows += (f'<tr>'
                 f'<td><a href="{name}/report.html">{name}</a></td>'
                 f'<td style="text-align:center">{thumb}</td>'
                 f'<td>{_badge(status)}</td>'
                 f'<td style="text-align:right">{wall}</td>'
                 f'<td style="text-align:right">{ms} ms</td>'
                 f'<td style="text-align:right">{f"{gc:,}" if isinstance(gc,int) else "—"}</td>'
                 + _cell(s,"final_psnr_3d",2) + _cell(s,"final_l1_3d",5)
                 + _cell(s,"final_iou_3d",4) + _cell(s,"final_ssim_3d",4)
                 + _cell(s,"final_psnr_2d",2) + _cell(s,"final_l1_2d",5)
                 + _cell(s,"final_iou_2d",4) + _cell(s,"final_ssim_2d",4)
                 + f'</tr>')

    table = (f'<div class="scroll-x"><table>'
             f'<thead><tr><th>Name</th><th>Preview</th><th>Status</th>'
             f'<th>Wall</th><th>Step</th><th>Gaussians</th>'
             f'<th>PSNR3D↑</th><th>L1-3D↓</th><th>IoU3D↑</th><th>SSIM3D↑</th>'
             f'<th>PSNR2D↑</th><th>L1-2D↓</th><th>IoU2D↑</th><th>SSIM2D↑</th>'
             f'</tr></thead><tbody>{rows}</tbody></table></div>')

    # ── Overlay charts — all 8 metrics ────────────────────────────────────
    def _overlay(uid, title, get_y, tier, h=280):
        traces = []
        for i,(n,r) in enumerate(all_results.items()):
            xs,ys = [],[]
            for cp in r.get("checkpoints",[]):
                v = get_y(cp.get(f"metrics_{tier}") or {})
                if v is not None: xs.append(cp["step"]); ys.append(v)
            t = _trace(xs,ys,n,_PAL[i%len(_PAL)])
            if t: traces.append(t)
        return _chart(uid, traces, title, h)

    ov_psnr3  = _overlay("ov_psnr3","PSNR 3D — all experiments",  lambda m:m.get("psnr"),      "3d")
    ov_l1_3   = _overlay("ov_l1_3", "L1 3D — all experiments",    lambda m:m.get("l1"),        "3d")
    ov_iou3   = _overlay("ov_iou3", "IoU 3D — all experiments",   lambda m:m.get("iou"),       "3d")
    ov_ssim3  = _overlay("ov_ssim3","SSIM 3D — all experiments",  lambda m:m.get("ssim_mean"), "3d")
    ov_psnr2  = _overlay("ov_psnr2","PSNR 2D — all experiments",  lambda m:m.get("psnr"),      "2d")
    ov_l1_2   = _overlay("ov_l1_2", "L1 2D — all experiments",    lambda m:m.get("l1"),        "2d")
    ov_iou2   = _overlay("ov_iou2", "IoU 2D — all experiments",   lambda m:m.get("iou"),       "2d")
    ov_ssim2  = _overlay("ov_ssim2","SSIM 2D — all experiments",  lambda m:m.get("ssim"),      "2d")

    # Gaussian count + step time overlays
    gauss_traces, st_traces = [], []
    for i,(n,r) in enumerate(all_results.items()):
        cps = r.get("checkpoints",[])
        gx=[cp["step"] for cp in cps]; gy=[cp.get("gaussian_count") for cp in cps]
        t = _trace(gx,gy,n,_PAL[i%len(_PAL)])
        if t: gauss_traces.append(t)
        samp = r.get("step_time_samples",[])
        if samp:
            t2 = _trace([s for s,_ in samp],[ms for _,ms in samp],n,_PAL[i%len(_PAL)])
            if t2: st_traces.append(t2)

    ov_gauss = _chart("ov_gauss", gauss_traces, "Gaussian Count — all experiments", 260)
    ov_st    = _chart("ov_st",    st_traces,    "Step Time ms — all experiments",   220)

    # ── Per-experiment inline blocks ───────────────────────────────────────
    exp_blocks = ""
    for s in sorted_s:
        name = s["name"]
        r    = all_results.get(name,{})
        cps  = r.get("checkpoints",[])
        desc = r.get("description","") or s.get("description","")
        st   = s.get("status","?")
        wall = str(timedelta(seconds=int(s.get("wall_time_seconds",0))))
        ms   = s.get("step_time_summary",{}).get("mean_ms","?")
        gc   = s.get("final_gaussian_count")
        p3   = s.get("final_psnr_3d")
        p2   = s.get("final_psnr_2d")

        # Mini metric plots (PSNR 3D + 2D only, compact)
        uid = name.replace(" ","_").replace("-","_")
        mini_traces = []
        for cp in cps:
            pass  # built below
        x3,y3,x2,y2 = [],[],[],[]
        for cp in cps:
            m3 = cp.get("metrics_3d") or {}
            m2 = cp.get("metrics_2d") or {}
            if m3.get("psnr") is not None: x3.append(cp["step"]); y3.append(m3["psnr"])
            if m2.get("psnr") is not None: x2.append(cp["step"]); y2.append(m2["psnr"])
        t1 = _trace(x3,y3,"PSNR 3D",_PAL[0]); t2 = _trace(x2,y2,"PSNR 2D",_PAL[1])
        mini_psnr = _chart(f"mini_{uid}_psnr",[t for t in [t1,t2] if t],
                           "PSNR (dB)",180)

        xl,yl3,yl2 = [],[],[]
        for cp in cps:
            m3 = cp.get("metrics_3d") or {}; m2 = cp.get("metrics_2d") or {}
            if m3.get("l1") is not None: xl.append(cp["step"]); yl3.append(m3["l1"])
        t3 = _trace(xl,yl3,"L1 3D",_PAL[2])
        mini_l1 = _chart(f"mini_{uid}_l1",[t for t in [t3] if t],"L1",180)

        # Pred history strip (first camera only, compact)
        cam_names = list(dict.fromkeys(
            r["camera"] for cp in cps for r in cp.get("renders",[])
        ))
        strips = "".join(_pred_history_strip(cps, base_dir/name, c, 90)
                         for c in cam_names[:3])  # max 3 cams on index

        exp_blocks += f"""
<div class="exp-block">
  <div class="exp-header">
    <span class="exp-name"><a href="{name}/report.html">{name}</a></span>
    {_badge(st)}
    <span class="exp-desc">{desc}</span>
    <span style="margin-left:auto;color:#6c7086;font-size:12px">
      wall {wall} &nbsp;·&nbsp; {ms} ms/step &nbsp;·&nbsp;
      {f"{gc:,}" if isinstance(gc,int) else "—"} gaussians &nbsp;·&nbsp;
      PSNR3D <span style="color:{_psnr_color(p3)};font-weight:600">{_fmt(p3,2)}</span> dB &nbsp;·&nbsp;
      PSNR2D <span style="color:{_psnr_color(p2)};font-weight:600">{_fmt(p2,2)}</span> dB
    </span>
  </div>
  <div class="grid-2" style="margin-bottom:10px">
    {mini_psnr}{mini_l1}
  </div>
  {strips}
</div>"""

    n_ok  = sum(1 for s in summaries if s.get("status")=="completed")
    n_exp = len(summaries)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8">
<title>Gaussian Splat Experiments</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head><body>
{_LIGHTBOX}
<h1>Gaussian Splat Experiments</h1>
<p class="sub">{n_exp} experiments &nbsp;·&nbsp; {n_ok} completed &nbsp;·&nbsp; {n_exp-n_ok} failed/aborted</p>

<h2>Results Summary</h2>
<p class="sub">Green = best in column · Red = worst · Click name for full experiment page</p>
{table}

<h2>3D Volumetric Metrics — All Experiments</h2>
<div class="grid-2">{ov_psnr3}{ov_l1_3}</div>
<div class="grid-2" style="margin-top:12px">{ov_iou3}{ov_ssim3}</div>

<h2>2D Screen-Space Metrics — All Experiments</h2>
<div class="grid-2">{ov_psnr2}{ov_l1_2}</div>
<div class="grid-2" style="margin-top:12px">{ov_iou2}{ov_ssim2}</div>

<h2>Population &amp; Performance</h2>
<div class="grid-2">{ov_gauss}{ov_st}</div>

<h2>Per-Experiment Overview</h2>
<p class="sub">Each block shows PSNR curves + prediction history · Click experiment name for full detail page</p>
{exp_blocks}

</body></html>"""

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(base_dir, summaries):
    base_dir = Path(base_dir)

    all_results = {}
    for s in summaries:
        name = s["name"]
        r = _load(base_dir/name)
        if r:
            # Attach config from config_used.yaml if available
            cfg_path = base_dir/name/"config_used.yaml"
            if cfg_path.exists():
                try:
                    import yaml
                    with open(cfg_path) as f:
                        r["config"] = yaml.safe_load(f)
                except: pass
            all_results[name] = r

    # Per-experiment pages
    for name, r in all_results.items():
        html = _exp_page(r, base_dir/name)
        (base_dir/name/"report.html").write_text(html, encoding="utf-8")

    # Index
    index_html = _index_page(summaries, all_results, base_dir)
    idx = base_dir/"report.html"
    idx.write_text(index_html, encoding="utf-8")
    return idx


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json as _json
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    args = p.parse_args()
    base = Path(args.results_dir)

    manifest = base/"batch_summary.json"
    if not manifest.exists():
        print(f"No batch_summary.json in {base}"); raise SystemExit(1)

    with open(manifest) as f:
        data = _json.load(f)

    sums = data.get("experiments",[])
    if not sums:
        print("No experiments found"); raise SystemExit(1)

    idx = generate(base, sums)
    print(f"Report → {idx}")
    print(f"Open  : file://{idx.resolve()}")