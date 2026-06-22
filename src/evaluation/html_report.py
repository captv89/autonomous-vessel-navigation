"""
Static HTML leaderboard generated from benchmark results.json.

Dependency-free output (single self-contained page, inline CSS) intended
for GitHub Pages deployment. Every number matches results.json exactly;
the page is presentation only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CSS = """
:root { --bg:#0e1420; --card:#171f2e; --text:#e8ebf0; --dim:#9aa3b2;
        --accent:#5dd39e; --warn:#e8865a; --bar:#3b82c4; }
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--text); margin:0;
       font:15px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:1.9em; margin:.2em 0 .1em; }
h2 { margin-top:1.8em; border-bottom:1px solid #2a3548;
     padding-bottom:.3em; }
.sub { color:var(--dim); margin-bottom:1.4em; }
.badge { display:inline-block; background:#22304a; border-radius:6px;
         padding:2px 10px; margin-right:8px; font-size:.85em;
         color:var(--dim); }
table { width:100%; border-collapse:collapse; background:var(--card);
        border-radius:10px; overflow:hidden; font-size:.92em; }
th, td { padding:9px 12px; text-align:left;
         border-bottom:1px solid #222c3e; white-space:nowrap; }
th { background:#1d2738; color:var(--dim); font-weight:600;
     font-size:.85em; text-transform:uppercase; letter-spacing:.04em; }
tr:last-child td { border-bottom:none; }
.rank { color:var(--dim); }
.score { font-weight:700; color:var(--accent); font-size:1.06em; }
.ci { color:var(--dim); font-size:.85em; }
.scorebar { background:#22304a; border-radius:4px; height:8px;
            width:140px; display:inline-block; vertical-align:middle;
            margin-left:8px; }
.scorebar i { display:block; height:100%; border-radius:4px;
              background:var(--bar); }
.agent { font-weight:600; }
footer { color:var(--dim); margin-top:3em; font-size:.85em; }
a { color:#7db8e8; }
code { background:#22304a; padding:1px 6px; border-radius:4px;
       font-size:.88em; }
.note { background:var(--card); border-left:3px solid var(--bar);
        padding:10px 16px; border-radius:0 8px 8px 0; color:var(--dim);
        margin:1em 0; }
.figs { display:flex; flex-wrap:wrap; gap:20px; margin-top:18px; }
.figs figure { background:var(--card); border-radius:10px;
               padding:14px 16px 8px; margin:0; flex:1 1 460px;
               min-width:280px; }
.figs figcaption { color:var(--dim); font-size:.85em; font-weight:600;
                   text-transform:uppercase; letter-spacing:.05em;
                   margin-bottom:8px; }
.figs svg { width:100%; height:auto; }
.tablewrap { overflow-x:auto; -webkit-overflow-scrolling:touch;
             border-radius:10px; }
.author { display:block; color:var(--dim); font-size:.8em;
          font-weight:400; }
.cards { display:grid; gap:14px;
         grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.card { background:var(--card); border-radius:10px; padding:14px 16px; }
.card h4 { margin:0 0 4px; font-size:1em; }
.card .author { margin-bottom:6px; }
.card p { margin:0; color:var(--dim); font-size:.9em; }
.fam { display:inline-block; font-size:.72em; color:#7db8e8;
       border:1px solid #2a3f5e; border-radius:99px; padding:1px 9px;
       margin-left:6px; vertical-align:middle; }
@media (max-width: 640px) {
  .wrap { padding:18px 12px 48px; }
  h1 { font-size:1.45em; }
  table { font-size:.82em; }
  th, td { padding:7px 8px; }
  .scorebar { width:70px; }
}
"""


def _rate(ci: Dict[str, float]) -> str:
    return (f"{ci['rate']:.1%} <span class='ci'>[{ci['low']:.1%}, "
            f"{ci['high']:.1%}]</span>")


def _mean(ci, digits=2) -> str:
    if not ci:
        return "–"
    return (f"{ci['mean']:.{digits}f} <span class='ci'>"
            f"[{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]</span>")


AGENT_COLORS = ["#5dd39e", "#5da9e8", "#e8c45a", "#e8865a", "#b48ce8",
                "#8ce8dd"]

SCENARIO_INFO = {
    "open_water": ("No obstacles", "Pure route following, no rule applies"),
    "head_on": ("Rule 14 — head-on", "Reciprocal courses; both vessels "
                "must alter to starboard and pass port-to-port"),
    "crossing_starboard": ("Rule 15/16 — give-way", "Traffic crosses from "
                           "starboard; own ship must take early, "
                           "substantial action to keep clear"),
    "crossing_port": ("Rule 17 — stand-on", "Traffic crosses from port; "
                      "own ship should hold course while it remains safe"),
    "overtaking": ("Rule 13 — overtaking", "Slow vessel ahead on the same "
                   "course; overtaking vessel keeps clear, either side"),
    "coastal": ("Mixed", "Landmasses forming a channel plus two traffic "
                "vessels; planning and avoidance together"),
    "multi_vessel": ("Rules 14+15 combined", "Two simultaneous conflicts "
                     "timed to overlap, plus a third vessel passing by"),
    "narrow_channel": ("Rule 14 in a fairway (Rule 9 flavor)",
                       "Oncoming vessel inside a narrow channel; limited "
                       "sea room on both sides"),
    "random": ("Mixed", "Seeded random islands and wandering traffic"),
}

VESSEL_MODEL_NOTE = (
    "All agents steer the same own-ship model: a 3-DOF surge–sway–yaw "
    "maneuvering model (first-order Nomoto yaw response, sideslip toward "
    "−k·u·r, turn-induced speed loss, first-order surge), an IMO-standard "
    "rudder actuator (35° limit, 70°-in-11-s slew rate), and a "
    "course-over-ground PD autopilot. Disturbed episodes add a "
    "scenario-seeded water current (drifting the traffic too) and wind "
    "gusts. Agents command intent (desired course and speed); the physics "
    "decides what the hull does.")
OUTCOME_COLORS = {"goal": "#3fa874", "collision": "#d9534f",
                  "grounding": "#e8965a", "out_of_bounds": "#9b6fd4",
                  "timeout": "#7d8694"}


def _svg_outcome_bars(agents_ranked, cond: str) -> str:
    """Stacked horizontal outcome bars, one row per agent."""
    bar_w, bar_h, gap, label_w = 560, 26, 14, 150
    rows = []
    outcomes = ["goal", "collision", "grounding", "out_of_bounds", "timeout"]
    for i, agent in enumerate(agents_ranked):
        episodes = agent["conditions"][cond]["episodes"]
        n = max(len(episodes), 1)
        counts = {o: sum(1 for e in episodes if e["outcome"] == o)
                  for o in outcomes}
        y = i * (bar_h + gap)
        x = label_w
        segs = [f'<text x="{label_w - 10}" y="{y + bar_h - 8}" '
                f'text-anchor="end" fill="#e8ebf0" font-size="13" '
                f'font-weight="600">{agent["name"]}</text>']
        for o in outcomes:
            frac = counts[o] / n
            w = frac * bar_w
            if w <= 0:
                continue
            segs.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" '
                f'height="{bar_h}" rx="3" fill="{OUTCOME_COLORS[o]}">'
                f'<title>{o}: {counts[o]}/{n} ({frac:.0%})</title></rect>')
            if frac >= 0.07:
                segs.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + bar_h - 8}" '
                    f'text-anchor="middle" fill="#0e1420" font-size="11" '
                    f'font-weight="700">{frac:.0%}</text>')
            x += w
        rows.append("".join(segs))
    height = len(agents_ranked) * (bar_h + gap)
    legend = "".join(
        f'<rect x="{label_w + j * 118}" y="{height + 6}" width="12" '
        f'height="12" rx="2" fill="{OUTCOME_COLORS[o]}"/>'
        f'<text x="{label_w + j * 118 + 17}" y="{height + 17}" '
        f'fill="#9aa3b2" font-size="12">{o.replace("_", " ")}</text>'
        for j, o in enumerate(outcomes))
    return (f'<svg viewBox="0 0 {label_w + bar_w + 10} {height + 30}" '
            f'role="img" aria-label="Episode outcomes by agent">'
            f'{"".join(rows)}{legend}</svg>')


def _svg_safety_speed(agents_ranked, cond: str) -> str:
    """Scatter: mean time to goal (x, lower=better) vs success rate (y)."""
    width, height, pad_l, pad_b, pad_t = 560, 300, 60, 44, 16
    pts = []
    for i, agent in enumerate(agents_ranked):
        c = agent["conditions"][cond]
        if not c.get("duration_s"):
            continue
        pts.append((agent["name"], c["duration_s"]["mean"],
                    c["success"]["rate"], AGENT_COLORS[i % len(AGENT_COLORS)]))
    if not pts:
        return ""
    xs = [p[1] for p in pts]
    x_min, x_max = min(xs) * 0.85, max(xs) * 1.12
    def X(v): return pad_l + (v - x_min) / (x_max - x_min) * (width - pad_l - 12)
    def Y(v): return pad_t + (1.0 - v) * (height - pad_t - pad_b)
    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = Y(frac)
        grid.append(f'<line x1="{pad_l}" y1="{gy:.0f}" x2="{width - 10}" '
                    f'y2="{gy:.0f}" stroke="#243149" stroke-width="1"/>'
                    f'<text x="{pad_l - 8}" y="{gy + 4:.0f}" '
                    f'text-anchor="end" fill="#9aa3b2" font-size="11">'
                    f'{frac:.0%}</text>')
    dots = []
    for name, x, y, color in pts:
        cx, cy = X(x), Y(y)
        dots.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="8" fill="{color}" '
            f'fill-opacity="0.9"><title>{name}: {x:.1f}s, {y:.0%} success'
            f'</title></circle>'
            f'<text x="{cx:.0f}" y="{cy - 13:.0f}" text-anchor="middle" '
            f'fill="#e8ebf0" font-size="12" font-weight="600">{name}</text>')
    axis = (f'<text x="{(pad_l + width) / 2:.0f}" y="{height - 6}" '
            f'text-anchor="middle" fill="#9aa3b2" font-size="12">'
            f'mean time to goal, s (successful episodes) — left and high is '
            f'better</text>'
            f'<text x="14" y="{height / 2:.0f}" fill="#9aa3b2" '
            f'font-size="12" transform="rotate(-90 14 {height / 2:.0f})" '
            f'text-anchor="middle">success rate</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Success rate vs time to goal">'
            f'{"".join(grid)}{"".join(dots)}{axis}</svg>')


def _svg_radar(agents_ranked, cond: str, scenarios) -> str:
    """Per-scenario success rate, one polygon per agent."""
    import math
    size, R = 460, 150
    cx, cy = size / 2, size / 2 + 8
    n_ax = len(scenarios)

    def point(axis: int, frac: float):
        ang = -math.pi / 2 + 2 * math.pi * axis / n_ax
        return (cx + R * frac * math.cos(ang),
                cy + R * frac * math.sin(ang))

    grid = []
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{point(i, ring)[0]:.0f},{point(i, ring)[1]:.0f}"
                       for i in range(n_ax))
        grid.append(f'<polygon points="{pts}" fill="none" '
                    f'stroke="#243149" stroke-width="1"/>')
    labels = []
    for i, name in enumerate(scenarios):
        lx, ly = point(i, 1.22)
        anchor = ("middle" if abs(lx - cx) < 30
                  else "start" if lx > cx else "end")
        labels.append(f'<text x="{lx:.0f}" y="{ly:.0f}" '
                      f'text-anchor="{anchor}" fill="#9aa3b2" '
                      f'font-size="11">{name.replace("_", " ")}</text>')
        ax_x, ax_y = point(i, 1.0)
        grid.append(f'<line x1="{cx}" y1="{cy}" x2="{ax_x:.0f}" '
                    f'y2="{ax_y:.0f}" stroke="#243149" stroke-width="1"/>')

    polys, legend = [], []
    for k, agent in enumerate(agents_ranked):
        episodes = agent["conditions"][cond]["episodes"]
        color = AGENT_COLORS[k % len(AGENT_COLORS)]
        fracs = []
        for name in scenarios:
            eps = [e for e in episodes if e["scenario"] == name]
            fracs.append(sum(1 for e in eps if e.get("success"))
                         / max(len(eps), 1))
        pts = " ".join(f"{point(i, f)[0]:.0f},{point(i, f)[1]:.0f}"
                       for i, f in enumerate(fracs))
        polys.append(f'<polygon points="{pts}" fill="{color}" '
                     f'fill-opacity="0.10" stroke="{color}" '
                     f'stroke-width="2"><title>{agent["name"]}</title>'
                     f'</polygon>')
        ly = 14 + k * 17
        legend.append(f'<rect x="6" y="{ly - 9}" width="11" height="11" '
                      f'rx="2" fill="{color}"/>'
                      f'<text x="22" y="{ly + 1}" fill="#e8ebf0" '
                      f'font-size="11.5">{agent["name"]}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="Per-scenario success rate by agent">'
            f'{"".join(grid)}{"".join(labels)}{"".join(polys)}'
            f'{"".join(legend)}</svg>')


def _svg_condition_dumbbell(agents_ranked, conditions) -> str:
    """Per-agent success in each condition, connected: robustness at a
    glance (short line = robust, long line = condition-sensitive). The first
    condition is the baseline (solid marker); every other condition is a
    hollow marker, with its success-rate delta vs baseline printed inline."""
    if len(conditions) < 2:
        return ""
    base, others = conditions[0], conditions[1:]
    label_w, plot_w, row_h = 170, 420, 34
    width = label_w + plot_w + 110
    def X(v): return label_w + v * plot_w
    rows = []
    for i, agent in enumerate(agents_ranked):
        y = 24 + i * row_h
        rates = {c: agent["conditions"][c]["success"]["rate"] for c in conditions}
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        lo, hi = min(rates.values()), max(rates.values())
        marks = (f'<circle cx="{X(rates[base]):.0f}" cy="{y}" r="6" '
                 f'fill="{color}"><title>{base}: {rates[base]:.0%}</title></circle>')
        marks += "".join(
            f'<circle cx="{X(rates[c]):.0f}" cy="{y}" r="6" fill="{color}" '
            f'fill-opacity="0.45" stroke="{color}" stroke-width="2">'
            f'<title>{c}: {rates[c]:.0%}</title></circle>' for c in others)
        deltas = " · ".join(
            f'{c[:4]} {rates[c] - rates[base]:+.0%}' for c in others)
        rows.append(
            f'<text x="{label_w - 10}" y="{y + 4}" text-anchor="end" '
            f'fill="#e8ebf0" font-size="12.5" font-weight="600">'
            f'{agent["name"]}</text>'
            f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" stroke-opacity="0.55"/>'
            f'{marks}'
            f'<text x="{X(hi) + 12:.0f}" y="{y + 4}" '
            f'fill="#9aa3b2" font-size="11">{deltas}</text>')
    height = 24 + len(agents_ranked) * row_h + 26
    axis = "".join(
        f'<line x1="{X(v):.0f}" y1="14" x2="{X(v):.0f}" '
        f'y2="{height - 34}" stroke="#243149"/>'
        f'<text x="{X(v):.0f}" y="{height - 20}" text-anchor="middle" '
        f'fill="#9aa3b2" font-size="11">{v:.0%}</text>'
        for v in (0.25, 0.5, 0.75, 1.0))
    legend = (f'<circle cx="{label_w}" cy="{height - 4}" r="5" '
              f'fill="#9aa3b2"/><text x="{label_w + 10}" y="{height}" '
              f'fill="#9aa3b2" font-size="11">{base} (baseline, solid)</text>'
              f'<circle cx="{label_w + 160}" cy="{height - 4}" r="5" '
              f'fill="none" stroke="#9aa3b2" stroke-width="2"/>'
              f'<text x="{label_w + 170}" y="{height}" fill="#9aa3b2" '
              f'font-size="11">{", ".join(others)} (hollow)</text>')
    return (f'<svg viewBox="0 0 {width} {height + 8}" role="img" '
            f'aria-label="Success rate per condition by agent">'
            f'{axis}{"".join(rows)}{legend}</svg>')


def _compliance_cell(score: float, n: int) -> str:
    # red (0) -> amber (0.5) -> green (1)
    r = int(217 - score * (217 - 63))
    g = int(83 + score * (168 - 83))
    b = int(79 + score * 37)
    return (f'<td style="background:rgba({r},{g},{b},.28)">'
            f'{score:.2f} <span class="ci">(n={n})</span></td>')


def render_html(results: Dict[str, Any], nav_link: str = "") -> str:
    suite = results["suite"]
    agents = results["agents"]
    conditions = list(suite["conditions"])

    parts = [f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{suite['name']} v{suite['version']} — Leaderboard</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{suite['name']} <span style="color:var(--dim)">v{suite['version']}</span></h1>
{nav_link}
<p class="sub">{suite.get('description', '')}</p>
<p>
<span class="badge">generated {results['generated']}</span>
<span class="badge">{len(suite['scenarios'])} scenarios ×
{suite['episodes_per_scenario']} seeded episodes × {len(conditions)}
conditions</span>
{''.join(f'<span class="badge">{c}: <code>{h}</code></span>'
         for c, h in results['config_hash'].items())}
</p>
<div class="note"><b>Benchmark score</b> = 100 × (0.6 × success rate +
0.25 × COLREGs compliance + 0.15 × path efficiency). Rates carry Wilson
95% CIs, means carry seeded bootstrap 95% CIs; all metrics are computed
from ground-truth episode logs. Scores are comparable only between runs
with identical config hashes.</div>"""]

    # ------------------------------------------------ about the benchmark
    agent_cards = []
    for agent in agents.values():
        about = agent.get("about", {})
        agent_cards.append(
            f'<div class="card"><h4>{agent["name"]}'
            f'<span class="fam">{about.get("family", "")}</span></h4>'
            f'<span class="author">by {about.get("author", "unknown")}'
            f'</span><p>{about.get("summary", "")}</p></div>')
    scenario_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{SCENARIO_INFO[name][0]}"
        f"</td><td>{SCENARIO_INFO[name][1]}</td></tr>"
        for name in suite["scenarios"] if name in SCENARIO_INFO)
    parts.append(f"""<h2>The agents</h2>
<div class="cards">{''.join(agent_cards)}</div>
<h2>What is being tested</h2>
<p class="sub">{VESSEL_MODEL_NOTE}</p>
<div class="tablewrap"><table><thead><tr><th>Scenario</th>
<th>COLREGs rule</th><th>Situation</th></tr></thead>
<tbody>{scenario_rows}</tbody></table></div>
<p class="sub" style="margin-top:10px">Each scenario runs
{suite['episodes_per_scenario']} seeded episodes per condition;
<b>calm</b> has no disturbances, <b>disturbed</b> adds a scenario-seeded
random current (up to 0.3 cells/s) and wind gusts. Every agent faces the
identical episodes.</p>""")

    for cond in conditions:
        ranked = sorted(agents.values(),
                        key=lambda a: a["conditions"][cond]["benchmark_score"],
                        reverse=True)
        rows = []
        for rank, agent in enumerate(ranked, 1):
            c = agent["conditions"][cond]
            author = agent.get("about", {}).get("author", "")
            colregs = c["colregs"]
            colregs_str = (f"{colregs['mean_score']:.2f} "
                           f"<span class='ci'>(n={colregs['encounters']})"
                           f"</span>"
                           if colregs.get("encounters") else "–")
            bar = int(min(max(c["benchmark_score"], 0), 100) * 1.4)
            rows.append(f"""<tr>
<td class="rank">{rank}</td>
<td class="agent">{agent['name']}
    <span class="author">{author}</span></td>
<td><span class="score">{c['benchmark_score']}</span>
    <span class="scorebar"><i style="width:{bar}px"></i></span></td>
<td>{_rate(c['success'])}</td>
<td>{_rate(c['collision'])}</td>
<td>{_rate(c['grounding'])}</td>
<td>{colregs_str}</td>
<td>{_mean(c['duration_s'], 1)}</td>
<td>{_mean(c['min_separation'])}</td>
</tr>""")
        encounter_types = sorted({
            enc for a in ranked
            for enc in a["conditions"][cond]["colregs"]
            .get("by_encounter", {})})
        enc_rows = []
        for agent in ranked:
            by_enc = agent["conditions"][cond]["colregs"].get(
                "by_encounter", {})
            cells = "".join(
                _compliance_cell(by_enc[e]["mean_score"], by_enc[e]["n"])
                if e in by_enc else "<td>–</td>"
                for e in encounter_types)
            enc_rows.append(f"<tr><td class='agent'>{agent['name']}</td>"
                            f"{cells}</tr>")

        parts.append(f"""<h2>Condition: {cond}</h2>
<div class="tablewrap">
<table><thead><tr><th>#</th><th>Agent</th><th>Score</th><th>Success</th>
<th>Collision</th><th>Grounding</th><th>COLREGs</th>
<th>Time to goal (s)</th><th>Min separation</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="figs">
<figure><figcaption>Episode outcomes</figcaption>
{_svg_outcome_bars(ranked, cond)}</figure>
<figure><figcaption>Safety vs speed trade-off</figcaption>
{_svg_safety_speed(ranked, cond)}</figure>
<figure><figcaption>Success rate by scenario</figcaption>
{_svg_radar(ranked, cond, suite['scenarios'])}</figure>
</div>
<h3 style="color:var(--dim)">COLREGs compliance by encounter type</h3>
<div class="tablewrap">
<table><thead><tr><th>Agent</th>
{''.join(f'<th>{e}</th>' for e in encounter_types)}</tr></thead>
<tbody>{''.join(enc_rows)}</tbody></table></div>""")

    first_cond_rank = sorted(
        agents.values(),
        key=lambda a: a["conditions"][conditions[0]]["benchmark_score"],
        reverse=True)
    if len(conditions) >= 2:
        others = ", ".join(conditions[1:])
        parts.append(f"""<h2>Robustness: {conditions[0]} vs {others}</h2>
<p class="sub">How much does each model lose (or gain) when conditions get
harder? Solid = {conditions[0]} (baseline), hollow = {others}; the printed
delta is the success-rate change vs baseline.</p>
<div class="figs"><figure><figcaption>Success rate by condition</figcaption>
{_svg_condition_dumbbell(first_cond_rank, conditions)}</figure></div>""")

    parts.append(f"""<h2>Submit your model</h2>
<p>Implement the <code>Agent</code> contract
(<code>reset(obs)</code> / <code>decide(obs) → Decision</code>) and run:</p>
<pre><code>python main.py benchmark --suite benchmarks/v{suite['version']}.yaml \\
    --agent your_package.your_module:YourAgent</code></pre>
<p>See the
<a href="https://github.com/captv89/autonomous-vessel-navigation">repository</a>
and <code>docs/SUBMITTING.md</code>. Every leaderboard number is backed by
a replayable per-step episode log.</p>
<footer>VesselNav-Bench — transparent benchmarking for autonomous vessel
navigation.</footer>
</div></body></html>""")
    return "".join(parts)


def write_html(results_path: str | Path, out_path: str | Path,
               nav_link: str = "") -> Path:
    results = json.loads(Path(results_path).read_text())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(results, nav_link=nav_link))
    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/v1/benchmark-v1/results.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/v1/benchmark-v1/index.html"
    print(write_html(src, dst))
