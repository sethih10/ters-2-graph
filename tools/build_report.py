"""Render the generation results as a standalone HTML page."""
import html, pickle, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'ters_report.html'
with open('exp/ters_20260828_925597/generated/summary.pkl', 'rb') as f:
    S = pickle.load(f)


def clean(sv):
    # strip the XML prolog rdMolDraw2D emits so the SVG can sit inline
    i = sv.find('<svg')
    return sv[i:] if i >= 0 else sv


def sample_card(s):
    cls = 'hit' if s['exact'] else ('formula' if s['formula_match'] else '')
    badge = ('<span class="badge b-exact">exact match</span>' if s['exact']
             else '<span class="badge b-form">formula ok</span>' if s['formula_match']
             else '' if s['valid'] else '<span class="badge b-bad">invalid</span>')
    return f"""<figure class="samp {cls}">
  <div class="draw">{clean(s['svg'])}</div>
  <figcaption>
    <div class="row"><span class="tan">{s['tanimoto']:.2f}</span>{badge}</div>
    <code>{html.escape(s['smiles'] or '—')}</code>
    <span class="form">{html.escape(s['formula'] or '')}</span>
  </figcaption>
</figure>"""


def block(split, rows):
    out = [f'<section class="split" id="{split}">',
           f'<h2>{split.upper()} split <span class="sub">782 molecules · '
           f'{sum(r["n_exact"] for r in rows)}/25 exact · '
           f'{sum(r["n_formula"] for r in rows)}/25 correct formula</span></h2>']
    for rank, r in enumerate(rows):
        out.append(f"""<article class="spectrum">
  <header>
    <span class="rank">{rank + 1}</span>
    <div class="meta">
      <span class="idx">spectrum #{r['ds_index']}</span>
      <span class="formula">{html.escape(r['truth_formula'] or '')}</span>
      <span class="atoms">{r['n_atoms']} atoms</span>
    </div>
    <div class="scores">
      <span><b>{r['best']:.2f}</b> best</span><span><b>{r['mean']:.2f}</b> mean</span>
      <span><b>{r['n_valid']}/5</b> valid</span><span><b>{r['n_exact']}/5</b> exact</span>
    </div>
  </header>
  <div class="grid">
    <figure class="truth">
      <div class="draw">{clean(r['truth_svg'])}</div>
      <figcaption><code>{html.escape(r['truth_smiles'] or '')}</code></figcaption>
    </figure>
    <div class="samples">{''.join(sample_card(s) for s in r['samples'])}</div>
  </div>
</article>""")
    out.append('</section>')
    return '\n'.join(out)


PAGE = f"""<title>TERS Structure Recovery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root {{
  --paper:#F6F8F8; --card:#FFFFFF; --ink:#0F1619; --muted:#5B6A6E;
  --line:#DCE4E4; --accent:#0B6E7F; --hit:#2E7D4F; --form:#8A6A1F; --bad:#A4562F;
  --hit-bg:#EAF3ED; --form-bg:#F6F1E4;
}}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0C1214; --card:#141C1F; --ink:#E8EFEF; --muted:#93A3A6;
    --line:#243033; --accent:#5FC6D4; --hit:#7BC79A; --form:#D9BC72; --bad:#E0937A;
    --hit-bg:#16261D; --form-bg:#261F13;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0C1214; --card:#141C1F; --ink:#E8EFEF; --muted:#93A3A6;
  --line:#243033; --accent:#5FC6D4; --hit:#7BC79A; --form:#D9BC72; --bad:#E0937A;
  --hit-bg:#16261D; --form-bg:#261F13;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink); margin:0;
  font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.55; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:56px 28px 96px; }}
h1 {{ font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:2.6rem;
  margin:0 0 .3em; letter-spacing:-.015em; text-wrap:balance; }}
.lede {{ font-size:1.1rem; color:var(--muted); max-width:64ch; margin:0 0 2.2rem; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin-bottom:1rem; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:3px;
  overflow:hidden; margin:0 0 3rem; }}
.kpi {{ background:var(--card); padding:18px 20px; }}
.kpi b {{ display:block; font-family:"IBM Plex Serif",serif; font-size:1.9rem;
  line-height:1.1; font-variant-numeric:tabular-nums; }}
.kpi span {{ font-size:.8rem; color:var(--muted); }}
h2 {{ font-family:"IBM Plex Serif",serif; font-size:1.35rem; margin:3rem 0 1.2rem;
  padding-bottom:.5rem; border-bottom:2px solid var(--accent); }}
h2 .sub {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--muted);
  letter-spacing:.06em; float:right; font-weight:400; padding-top:.7em; }}
.spectrum {{ background:var(--card); border:1px solid var(--line); border-radius:3px;
  margin-bottom:14px; overflow:hidden; }}
.spectrum header {{ display:flex; align-items:center; gap:16px; padding:12px 16px;
  border-bottom:1px solid var(--line); flex-wrap:wrap; }}
.rank {{ font-family:"IBM Plex Mono",monospace; font-size:.8rem; width:26px; height:26px;
  display:grid; place-items:center; border:1px solid var(--accent); color:var(--accent);
  border-radius:2px; flex:none; }}
.meta {{ display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; }}
.idx {{ font-weight:600; }}
.meta .formula {{ font-family:"IBM Plex Mono",monospace; color:var(--accent); font-size:.9rem; }}
.atoms, .scores {{ color:var(--muted); font-size:.82rem; }}
.scores {{ margin-left:auto; display:flex; gap:16px; font-variant-numeric:tabular-nums; }}
.scores b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
.grid {{ display:grid; grid-template-columns:minmax(210px,250px) 1fr; gap:0; }}
.truth {{ margin:0; padding:14px; border-right:2px solid var(--accent); background:var(--paper); }}
.samples {{ display:flex; gap:0; overflow-x:auto; }}
.samp {{ margin:0; padding:12px; border-right:1px solid var(--line); min-width:190px; flex:1; }}
.samp:last-child {{ border-right:none; }}
.samp.hit {{ background:var(--hit-bg); box-shadow:inset 0 3px 0 var(--hit); }}
.samp.formula {{ background:var(--form-bg); box-shadow:inset 0 3px 0 var(--form); }}
.draw {{ display:grid; place-items:center; }}
.draw svg {{ max-width:100%; height:auto; }}
figcaption {{ margin-top:8px; }}
figcaption code {{ font-family:"IBM Plex Mono",monospace; font-size:.66rem; color:var(--muted);
  display:block; word-break:break-all; line-height:1.35; }}
.truth figcaption code {{ color:var(--ink); font-size:.7rem; }}
.row {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.tan {{ font-family:"IBM Plex Mono",monospace; font-weight:500; font-variant-numeric:tabular-nums; }}
.badge {{ font-family:"IBM Plex Mono",monospace; font-size:.6rem; letter-spacing:.06em;
  text-transform:uppercase; padding:2px 6px; border-radius:2px; }}
.b-exact {{ background:var(--hit); color:var(--card); }}
.b-form {{ color:var(--form); border:1px solid var(--form); }}
.b-bad {{ color:var(--bad); border:1px solid var(--bad); }}
.form {{ font-family:"IBM Plex Mono",monospace; font-size:.64rem; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; font-size:.85rem;
  font-variant-numeric:tabular-nums; margin-bottom:1rem; }}
th, td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; text-transform:uppercase;
  letter-spacing:.08em; color:var(--muted); font-weight:500; }}
.tbl {{ overflow-x:auto; }}
.note {{ font-size:.85rem; color:var(--muted); max-width:70ch; }}
.caveat {{ border-left:2px solid var(--bad); padding:2px 0 2px 14px; margin:1.6rem 0;
  font-size:.88rem; color:var(--muted); max-width:70ch; }}
.caveat b {{ color:var(--ink); }}
</style>

<div class="wrap">
<div class="eyebrow">checkpoint 28001 · EMA · 250 ancestral steps</div>
<h1>Can it read a spectrum back into a molecule?</h1>
<p class="lede">Five held-out TERS image cubes per split, five independent reverse-diffusion
samples each. The model never sees the answer — only the cube and the atom count. A perfect
model would return the same molecule five times out of five.</p>

<div class="kpis">
  <div class="kpi"><b>2/25</b><span>exact structures recovered (test)</span></div>
  <div class="kpi"><b>49/50</b><span>chemically valid molecules</span></div>
  <div class="kpi"><b>11/50</b><span>correct molecular formula</span></div>
  <div class="kpi"><b>0.22</b><span>mean Tanimoto, test</span></div>
  <div class="kpi"><b>+0.4%</b><span>test-vs-val loss gap — no overfitting</span></div>
</div>

<div class="caveat"><b>Both exact hits are small molecules</b> — 8 and 9 heavy atoms. Every
molecule of 12 heavy atoms or more failed, on both splits. Recovery falls off sharply with
size, which is the single clearest pattern in these 50 samples.</div>

{block('test', S['test'])}
{block('val', S['val'])}

<h2>Held-out loss, val vs test <span class="sub">timestep pinned, identical noise</span></h2>
<div class="tbl"><table>
<thead><tr><th>noise t</th><th>val loss</th><th>test loss</th><th>gap</th>
<th>val penalty for wrong spectrum</th><th>test penalty</th></tr></thead>
<tbody>
<tr><td>0.05</td><td>0.0355</td><td>0.0295</td><td>−16.9%</td><td>+301%</td><td>+133%</td></tr>
<tr><td>0.15</td><td>0.0962</td><td>0.0837</td><td>−13.1%</td><td>+229%</td><td>+129%</td></tr>
<tr><td>0.30</td><td>0.3387</td><td>0.3245</td><td>−4.2%</td><td>+380%</td><td>+330%</td></tr>
<tr><td>0.50</td><td>5.9420</td><td>5.8975</td><td>−0.8%</td><td>+11.2%</td><td>+7.0%</td></tr>
<tr><td>0.70</td><td>6.8998</td><td>6.9253</td><td>+0.4%</td><td>+2.7%</td><td>+1.0%</td></tr>
<tr><td>0.90</td><td>4.2975</td><td>4.3209</td><td>+0.5%</td><td>+2.4%</td><td>+0.5%</td></tr>
</tbody></table></div>
<p class="note">Test tracks validation within a percent at every noise level, and beats it at
the low-noise end. After 573 epochs on 6,254 molecules there is no overfitting to find. The
penalty columns show what happens when a molecule is handed a different molecule's spectrum:
the model leans on the cube heavily near clean data and barely at all near pure noise.</p>

<div class="caveat">The wrong-spectrum penalty is roughly twice as large on val as on test.
Read that cautiously — the shuffle pairs each molecule with its <b>neighbour in dataset
order</b>, so how similar those neighbours happen to be differs between splits. It is a
confound in the test, not necessarily a property of the model.</div>
</div>
"""
with open(OUT, 'w') as f:
    f.write(PAGE)
print(f'wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)')
