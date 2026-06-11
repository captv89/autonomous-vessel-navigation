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
"""


def _rate(ci: Dict[str, float]) -> str:
    return (f"{ci['rate']:.1%} <span class='ci'>[{ci['low']:.1%}, "
            f"{ci['high']:.1%}]</span>")


def _mean(ci, digits=2) -> str:
    if not ci:
        return "–"
    return (f"{ci['mean']:.{digits}f} <span class='ci'>"
            f"[{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]</span>")


def render_html(results: Dict[str, Any]) -> str:
    suite = results["suite"]
    agents = results["agents"]
    conditions = list(suite["conditions"])

    parts = [f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{suite['name']} v{suite['version']} — Leaderboard</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{suite['name']} <span style="color:var(--dim)">v{suite['version']}</span></h1>
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

    for cond in conditions:
        ranked = sorted(agents.values(),
                        key=lambda a: a["conditions"][cond]["benchmark_score"],
                        reverse=True)
        rows = []
        for rank, agent in enumerate(ranked, 1):
            c = agent["conditions"][cond]
            colregs = c["colregs"]
            colregs_str = (f"{colregs['mean_score']:.2f} "
                           f"<span class='ci'>(n={colregs['encounters']})"
                           f"</span>"
                           if colregs.get("encounters") else "–")
            bar = int(min(max(c["benchmark_score"], 0), 100) * 1.4)
            rows.append(f"""<tr>
<td class="rank">{rank}</td>
<td class="agent">{agent['name']}</td>
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
                f"<td>{by_enc[e]['mean_score']:.2f} "
                f"<span class='ci'>(n={by_enc[e]['n']})</span></td>"
                if e in by_enc else "<td>–</td>"
                for e in encounter_types)
            enc_rows.append(f"<tr><td class='agent'>{agent['name']}</td>"
                            f"{cells}</tr>")

        parts.append(f"""<h2>Condition: {cond}</h2>
<table><thead><tr><th>#</th><th>Agent</th><th>Score</th><th>Success</th>
<th>Collision</th><th>Grounding</th><th>COLREGs</th>
<th>Time to goal (s)</th><th>Min separation</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h3 style="color:var(--dim)">COLREGs compliance by encounter type</h3>
<table><thead><tr><th>Agent</th>
{''.join(f'<th>{e}</th>' for e in encounter_types)}</tr></thead>
<tbody>{''.join(enc_rows)}</tbody></table>""")

    parts.append("""<h2>Submit your model</h2>
<p>Implement the <code>Agent</code> contract
(<code>reset(obs)</code> / <code>decide(obs) → Decision</code>) and run:</p>
<pre><code>python main.py benchmark --suite benchmarks/v1.yaml \\
    --agent your_package.your_module:YourAgent</code></pre>
<p>See the
<a href="https://github.com/captv89/autonomous-vessel-navigation">repository</a>
and <code>docs/SUBMITTING.md</code>. Every leaderboard number is backed by
a replayable per-step episode log.</p>
<footer>VesselNav-Bench — transparent benchmarking for autonomous vessel
navigation.</footer>
</div></body></html>""")
    return "".join(parts)


def write_html(results_path: str | Path, out_path: str | Path) -> Path:
    results = json.loads(Path(results_path).read_text())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(results))
    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/benchmark-v1/results.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/benchmark-v1/index.html"
    print(write_html(src, dst))
