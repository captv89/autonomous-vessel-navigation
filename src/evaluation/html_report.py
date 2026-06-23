"""
Static HTML leaderboard generated from benchmark results.json.

Dependency-free output for the data itself (every number is rendered into
the DOM straight from results.json and matches it exactly; the page is
presentation only). The page is progressively enhanced with a cinematic
3D ocean hero rendered via three.js loaded from a pinned, SRI-verified CDN
build. If the CDN or WebGL is unavailable — or JavaScript is disabled —
the hero falls back to a static CSS ocean gradient and the full leaderboard
still renders. No enhancement is required to read any benchmark number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Pinned three.js build (UMD global `THREE`) with Subresource Integrity so the
# byte content is verified by the browser; `defer` keeps it from blocking the
# initial paint and the hero init runs on `window.load` once it is available.
THREE_CDN = (
    '<script defer '
    'src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" '
    'integrity="sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu" '
    'crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)

CSS = """
:root { --bg:#0a0f1a; --card:#171f2e; --text:#e8ebf0; --dim:#9aa3b2;
        --accent:#5dd39e; --warn:#e8865a; --bar:#3b82c4;
        --gold:#e8c45a; --silver:#c9d2e0; --bronze:#d8966a;
        --glass:rgba(23,31,46,.62); --line:rgba(255,255,255,.08); }
* { box-sizing: border-box; }
html { scroll-behavior:smooth; }
body { background:
        radial-gradient(110% 60% at 50% -10%, #102234 0%, transparent 60%),
        linear-gradient(180deg, #0a0f1a 0%, #0b1320 100%) fixed;
       color:var(--text); margin:0;
       font:15px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:1.9em; margin:.2em 0 .1em; }
h2 { margin-top:1.8em; border-bottom:1px solid #2a3548;
     padding-bottom:.3em; scroll-margin-top:18px; }
.sub { color:var(--dim); margin-bottom:1.4em; }
.badge { display:inline-block; background:#22304a; border-radius:6px;
         padding:2px 10px; margin-right:8px; font-size:.85em;
         color:var(--dim); }

/* ---------------------------------------------------------------- 3D hero */
.hero { position:relative; height:100vh; min-height:560px; width:100%;
        overflow:hidden; display:flex; align-items:center;
        justify-content:center;
        background:linear-gradient(180deg,#0a1020 0%,#0e2438 32%,
                   #114049 52%,#0c1f2b 72%,#0a0f1a 100%); }
.hero canvas { position:absolute; inset:0; width:100%; height:100%;
               display:block; opacity:0; transition:opacity 1.4s ease; }
.hero canvas.on { opacity:1; }
.hero::after { content:""; position:absolute; inset:0; pointer-events:none;
   background:
     radial-gradient(120% 75% at 50% 16%, rgba(93,211,158,.12), transparent 60%),
     linear-gradient(180deg, rgba(10,15,26,0) 52%, rgba(10,15,26,.9) 100%); }
.hero-inner { position:relative; z-index:2; text-align:center; padding:0 22px;
   max-width:920px; animation:floatIn 1.2s cubic-bezier(.2,.7,.2,1) both; }
.hero-eyebrow { color:var(--accent); font-size:.82rem; font-weight:600;
   text-transform:uppercase; letter-spacing:.22em; margin-bottom:14px; }
.hero-title { font-size:clamp(2.4rem,7vw,5rem); font-weight:800;
   line-height:1.02; margin:0; letter-spacing:-.02em;
   background:linear-gradient(180deg,#ffffff 0%,#c7ecda 62%,#5dd39e 100%);
   -webkit-background-clip:text; background-clip:text; color:transparent;
   filter:drop-shadow(0 4px 34px rgba(93,211,158,.25)); }
.hero-title .v { -webkit-text-fill-color:#7d8aa0; color:#7d8aa0;
   font-weight:600; }
.hero-tag { color:#d2dae6; font-size:clamp(1rem,2.1vw,1.25rem);
   margin:1em auto 0; max-width:660px; }
.hero-stat { margin-top:1.7em; display:inline-flex; flex-wrap:wrap;
   justify-content:center; }
.hero-stat span { padding:4px 22px; border-right:1px solid rgba(255,255,255,.15); }
.hero-stat span:last-child { border-right:none; }
.hero-stat b { display:block; font-size:1.8rem; font-weight:700;
   color:var(--accent); line-height:1.1; }
.hero-stat i { font-style:normal; color:var(--dim); font-size:.72rem;
   text-transform:uppercase; letter-spacing:.1em; }
.scroll-cue { position:absolute; bottom:26px; left:50%; z-index:2;
   transform:translateX(-50%); color:var(--dim); text-decoration:none;
   font-size:.72rem; text-transform:uppercase; letter-spacing:.18em;
   text-align:center; animation:bob 2.2s ease-in-out infinite; }
.scroll-cue::after { content:"\\2193"; display:block; font-size:1.3rem;
   margin-top:8px; }
@keyframes floatIn { from{opacity:0;transform:translateY(26px)}
   to{opacity:1;transform:none} }
@keyframes bob { 0%,100%{transform:translateX(-50%) translateY(0)}
   50%{transform:translateX(-50%) translateY(8px)} }

/* ------------------------------------------------------------------ tables */
table { width:100%; border-collapse:collapse; background:var(--card);
        border-radius:10px; overflow:hidden; font-size:.92em; }
th, td { padding:9px 12px; text-align:left;
         border-bottom:1px solid #222c3e; white-space:nowrap; }
th { background:#1d2738; color:var(--dim); font-weight:600;
     font-size:.85em; text-transform:uppercase; letter-spacing:.04em; }
tr:last-child td { border-bottom:none; }
tbody tr { transition:background .2s ease; }
tbody tr:hover { background:#1c2740; }
tr.top1 td:first-child { box-shadow:inset 3px 0 0 var(--gold); }
tr.top2 td:first-child { box-shadow:inset 3px 0 0 var(--silver); }
tr.top3 td:first-child { box-shadow:inset 3px 0 0 var(--bronze); }
.rank { color:var(--dim); }
.score { font-weight:700; color:var(--accent); font-size:1.06em; }
.ci { color:var(--dim); font-size:.85em; }
.scorebar { background:#22304a; border-radius:4px; height:8px;
            width:140px; display:inline-block; vertical-align:middle;
            margin-left:8px; }
.scorebar i { display:block; height:100%; border-radius:4px;
              background:linear-gradient(90deg,var(--bar),var(--accent)); }
.agent { font-weight:600; }
footer { color:var(--dim); margin-top:3em; font-size:.85em; }
a { color:#7db8e8; }
code { background:#22304a; padding:1px 6px; border-radius:4px;
       font-size:.88em; }
.note { background:var(--glass);
        backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
        border-left:3px solid var(--bar);
        padding:10px 16px; border-radius:0 8px 8px 0; color:var(--dim);
        margin:1em 0; }

/* ----------------------------------------------------------------- podium */
.podium-wrap { scroll-margin-top:18px; }
.podium { display:grid; grid-template-columns:repeat(3,1fr); gap:16px;
          align-items:end; margin-top:18px; }
.pcard { background:linear-gradient(180deg, rgba(36,48,74,.5),
                                     rgba(23,31,46,.66));
         backdrop-filter:blur(9px); -webkit-backdrop-filter:blur(9px);
         border:1px solid var(--line); border-radius:16px;
         padding:22px 18px; text-align:center; position:relative; }
.pcard.gold { border-color:rgba(232,196,90,.55); transform:translateY(-14px);
              box-shadow:0 20px 54px rgba(232,196,90,.16); }
.pcard.silver { border-color:rgba(201,210,224,.42); }
.pcard.bronze { border-color:rgba(216,150,106,.42); }
.pmedal { width:34px; height:34px; line-height:34px; border-radius:50%;
          margin:0 auto 10px; font-weight:700; color:#0e1420; }
.gold .pmedal { background:var(--gold); }
.silver .pmedal { background:var(--silver); }
.bronze .pmedal { background:var(--bronze); }
.pname { font-weight:700; font-size:1.1rem; }
.pscore { font-size:2.4rem; font-weight:800; color:var(--accent);
          line-height:1.1; margin-top:6px; }
.plabel { color:var(--dim); font-size:.7rem; text-transform:uppercase;
          letter-spacing:.08em; }
.psub { color:var(--dim); font-size:.85rem; margin-top:8px; }

/* ----------------------------------------------------------------- charts */
.figs { display:flex; flex-wrap:wrap; gap:20px; margin-top:18px; }
.figs figure { background:var(--glass);
               backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
               border:1px solid var(--line); border-radius:14px;
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
.card { background:var(--glass);
        backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
        border:1px solid var(--line); border-radius:14px;
        padding:14px 16px; transition:transform .2s ease, border-color .2s ease; }
.card:hover { transform:translateY(-3px); border-color:rgba(93,211,158,.4); }
.card h4 { margin:0 0 4px; font-size:1em; }
.card .author { margin-bottom:6px; }
.card p { margin:0; color:var(--dim); font-size:.9em; }
.fam { display:inline-block; font-size:.72em; color:#7db8e8;
       border:1px solid #2a3f5e; border-radius:99px; padding:1px 9px;
       margin-left:6px; vertical-align:middle; }

/* reveal-on-scroll (added by JS only, so no-JS shows everything) */
.reveal { opacity:0; transform:translateY(20px);
          transition:opacity .7s ease, transform .7s ease; }
.reveal.in { opacity:1; transform:none; }

@media (max-width: 640px) {
  .wrap { padding:18px 12px 48px; }
  h1 { font-size:1.45em; }
  table { font-size:.82em; }
  th, td { padding:7px 8px; }
  .scorebar { width:70px; }
  .podium { grid-template-columns:1fr; gap:12px; }
  .pcard.gold { transform:none; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior:auto; }
  .hero-inner, .scroll-cue { animation:none; }
  .reveal { opacity:1 !important; transform:none !important; transition:none; }
}
"""

# Cinematic ocean hero. Runs on `window.load` so the deferred three.js build is
# already parsed; bails cleanly to the CSS gradient if THREE or WebGL is absent
# or the user prefers reduced motion. The whole scene is decorative — no
# benchmark data flows through it.
HERO_JS = r"""
window.addEventListener('load', function () {
  var canvas = document.getElementById('sea');
  if (!canvas) return;
  var reduce = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function hasWebGL() {
    try {
      var c = document.createElement('canvas');
      return !!(window.WebGLRenderingContext &&
        (c.getContext('webgl') || c.getContext('experimental-webgl')));
    } catch (e) { return false; }
  }
  if (!window.THREE || !hasWebGL()) return;   // CSS ocean fallback stays

  var THREE = window.THREE;
  var renderer;
  try {
    // Context creation can still fail after hasWebGL() passes (e.g. browser
    // hardware acceleration disabled or GPU blocklisted). Degrade to the CSS
    // ocean rather than throwing an uncaught exception.
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  } catch (e) { return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
  camera.position.set(0, 17, 46);

  scene.add(new THREE.AmbientLight(0x33415c, 0.85));
  var sun = new THREE.DirectionalLight(0xfff0d0, 1.1);
  sun.position.set(-40, 60, 24);
  scene.add(sun);

  var uniforms = {
    uTime:    { value: 0 },
    uDeep:    { value: new THREE.Color(0x0a1422) },
    uShallow: { value: new THREE.Color(0x1d6070) },
    uCrest:   { value: new THREE.Color(0x5dd39e) },
    uFog:     { value: new THREE.Color(0x0c1a28) },
    uSun:     { value: new THREE.Vector3(-0.5, 0.7, 0.35) }
  };
  var vert = [
    'uniform float uTime;',
    'varying float vH; varying vec3 vN; varying vec3 vWorld; varying float vDist;',
    'float wave(vec2 p){',
    '  float h = 0.0;',
    '  h += 1.70*sin(dot(vec2(0.86,0.51),p)*0.09 + uTime*1.05);',
    '  h += 1.05*sin(dot(vec2(-0.57,0.82),p)*0.15 + uTime*0.85);',
    '  h += 0.55*sin(dot(vec2(0.40,-0.92),p)*0.24 + uTime*1.55);',
    '  h += 0.28*sin(dot(vec2(0.99,0.10),p)*0.37 + uTime*2.10);',
    '  return h;',
    '}',
    'void main(){',
    '  vec3 pos = position;',
    '  vec2 p = pos.xz;',
    '  float h = wave(p);',
    '  pos.y += h; vH = h;',
    '  float e = 0.75;',
    '  float hR = wave(p+vec2(e,0.0)); float hL = wave(p+vec2(-e,0.0));',
    '  float hU = wave(p+vec2(0.0,e)); float hD = wave(p+vec2(0.0,-e));',
    '  vN = normalize(vec3(hL-hR, 2.0*e, hD-hU));',
    '  vec4 world = modelMatrix*vec4(pos,1.0); vWorld = world.xyz;',
    '  vec4 mv = modelViewMatrix*vec4(pos,1.0); vDist = -mv.z;',
    '  gl_Position = projectionMatrix*mv;',
    '}'
  ].join('\n');
  var frag = [
    'uniform vec3 uDeep; uniform vec3 uShallow; uniform vec3 uCrest;',
    'uniform vec3 uFog; uniform vec3 uSun;',
    'varying float vH; varying vec3 vN; varying vec3 vWorld; varying float vDist;',
    'void main(){',
    '  vec3 n = normalize(vN);',
    '  vec3 viewDir = normalize(cameraPosition - vWorld);',
    '  vec3 sun = normalize(uSun);',
    '  float hN = clamp(vH*0.22 + 0.5, 0.0, 1.0);',
    '  vec3 col = mix(uDeep, uShallow, hN);',
    '  col = mix(col, uCrest, clamp(vH-0.6,0.0,1.0)*0.55);',
    '  float diff = max(dot(n,sun),0.0);',
    '  col += diff*0.10*vec3(0.6,0.8,0.7);',
    '  vec3 r = reflect(-sun,n);',
    '  float spec = pow(max(dot(r,viewDir),0.0),80.0);',
    '  col += spec*vec3(1.0,0.93,0.75)*0.9;',
    '  float fres = pow(1.0-max(dot(n,viewDir),0.0),4.0);',
    '  col = mix(col, uCrest*0.6+uShallow*0.4, fres*0.35);',
    '  float fog = smoothstep(70.0,340.0,vDist);',
    '  col = mix(col, uFog, fog);',
    '  gl_FragColor = vec4(col,1.0);',
    '}'
  ].join('\n');

  var geo = new THREE.PlaneGeometry(800, 800, 180, 180);
  geo.rotateX(-Math.PI/2);
  var ocean = new THREE.Mesh(geo, new THREE.ShaderMaterial({
    uniforms: uniforms, vertexShader: vert, fragmentShader: frag
  }));
  scene.add(ocean);

  function makeVessel() {
    var g = new THREE.Group();
    var s = new THREE.Shape();
    s.moveTo(0, 3.2); s.lineTo(1.1, 1.0); s.lineTo(1.1, -2.6);
    s.lineTo(-1.1, -2.6); s.lineTo(-1.1, 1.0); s.lineTo(0, 3.2);
    var hull = new THREE.Mesh(
      new THREE.ExtrudeGeometry(s, { depth: 1.4, bevelEnabled: true,
        bevelThickness: 0.25, bevelSize: 0.25, bevelSegments: 2 }),
      new THREE.MeshStandardMaterial({ color: 0x1b2433, roughness: 0.6, metalness: 0.2 }));
    hull.rotation.x = -Math.PI/2; hull.position.y = 0.2;
    g.add(hull);
    var cabin = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.7, 1.6),
      new THREE.MeshStandardMaterial({ color: 0x5dd39e, roughness: 0.4,
        emissive: 0x0c4030, emissiveIntensity: 0.4 }));
    cabin.position.set(0, 1.0, -0.3); g.add(cabin);
    g.scale.set(1.6, 1.6, 1.6);
    return g;
  }
  var vessel = makeVessel();
  scene.add(vessel);

  function path(t) {                       // gentle S-course away from camera
    return new THREE.Vector3(Math.sin(t*Math.PI*2.0)*16.0, 0, 60 - t*200);
  }
  var pts = [];
  for (var i = 0; i <= 120; i++) pts.push(path(i/120));
  var curve = new THREE.CatmullRomCurve3(pts);
  var line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(curve.getPoints(220)),
    new THREE.LineBasicMaterial({ color: 0x5dd39e, transparent: true, opacity: 0.32 }));
  line.position.y = 0.2;
  scene.add(line);

  var clock = new THREE.Clock();
  function placeVessel(time) {
    var t = (time * 0.025) % 1;
    var p = path(t), p2 = path((t + 0.002) % 1);
    vessel.position.set(p.x, 0.35 + Math.sin(time*1.2)*0.22, p.z);
    vessel.rotation.y = Math.atan2(p2.x - p.x, p2.z - p.z);
    vessel.rotation.z = Math.sin(time*0.9)*0.06;
    vessel.rotation.x = Math.sin(time*0.7)*0.04;
  }
  function render() {
    var time = clock.getElapsedTime();
    uniforms.uTime.value = time;
    placeVessel(time);
    camera.position.x = Math.sin(time*0.08)*7;
    camera.position.y = 17 + Math.sin(time*0.15)*1.2;
    camera.lookAt(0, 1.5, -26);
    renderer.render(scene, camera);
  }
  function resize() {
    var w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();
  canvas.classList.add('on');

  if (reduce) { uniforms.uTime.value = 1.8; placeVessel(1.8);
    camera.lookAt(0, 1.5, -26); renderer.render(scene, camera); return; }

  var visible = true, inView = true;
  document.addEventListener('visibilitychange', function () {
    visible = !document.hidden;
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (es) {
      inView = es[0].isIntersecting;
    }, { threshold: 0 }).observe(canvas);
  }
  (function loop() {
    requestAnimationFrame(loop);
    if (visible && inView) render();
  })();
});
"""

# Progressive enhancement: fade sections in on scroll and count scores up.
# Adds classes only via JS, so with JS disabled everything is fully visible and
# every score shows its exact final value immediately.
REVEAL_JS = r"""
window.addEventListener('DOMContentLoaded', function () {
  if (!('IntersectionObserver' in window)) return;
  var reduce = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.08 });
    document.querySelectorAll('h2,.cards,.figs,.tablewrap,.note,.podium')
      .forEach(function (el) { el.classList.add('reveal'); io.observe(el); });
  }

  var cio = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      var el = e.target; cio.unobserve(el);
      var orig = el.getAttribute('data-val');
      var target = parseFloat(orig); if (isNaN(target)) return;
      if (reduce) { el.textContent = orig; return; }
      var dot = orig.indexOf('.');
      var dec = dot >= 0 ? orig.length - dot - 1 : 0;
      var dur = 900, t0 = null;
      var step = function (ts) {
        if (!t0) t0 = ts;
        var k = Math.min((ts - t0) / dur, 1);
        var eased = 0.5 - 0.5 * Math.cos(Math.PI * k);
        el.textContent = (target * eased).toFixed(dec);
        if (k < 1) requestAnimationFrame(step); else el.textContent = orig;
      };
      requestAnimationFrame(step);
    });
  }, { threshold: 0.6 });
  document.querySelectorAll('.score,.pscore').forEach(function (el) {
    var v = parseFloat(el.textContent); if (isNaN(v)) return;
    el.setAttribute('data-val', el.textContent.trim());
    cio.observe(el);
  });
});
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


def _hero(suite: Dict[str, Any], agents: Dict[str, Any],
          conditions, scenarios) -> str:
    """Full-viewport cinematic ocean hero. Decorative only — the title, tagline
    and headline stats are real DOM so they survive any JS/WebGL failure."""
    n_agents = len(agents)
    n_scn = len(scenarios)
    total_eps = n_scn * suite.get("episodes_per_scenario", 0) * len(conditions)
    return (
        '<section class="hero">'
        '<canvas id="sea"></canvas>'
        '<div class="hero-inner">'
        '<div class="hero-eyebrow">Autonomous Vessel Navigation · '
        'COLREGs Benchmark</div>'
        f'<h1 class="hero-title">{suite["name"]}'
        f'<span class="v"> v{suite["version"]}</span></h1>'
        '<p class="hero-tag">A transparent, reproducible benchmark where '
        'rule-based, optimization and learning agents navigate the same seas '
        'under the same collision-avoidance rules — every number backed by a '
        'replayable episode log.</p>'
        '<div class="hero-stat">'
        f'<span><b>{n_agents}</b><i>agents</i></span>'
        f'<span><b>{n_scn}</b><i>scenarios</i></span>'
        f'<span><b>{total_eps}</b><i>seeded episodes</i></span>'
        '</div></div>'
        '<a class="scroll-cue" href="#leaderboard">View the leaderboard</a>'
        '</section>')


def _podium(agents: Dict[str, Any], cond: str) -> str:
    """Top-3 glass podium for a condition, built from the same ranked data the
    leaderboard table uses (gold centered and lifted, silver left, bronze right)."""
    ranked = sorted(agents.values(),
                    key=lambda a: a["conditions"][cond]["benchmark_score"],
                    reverse=True)[:3]
    klass = ["gold", "silver", "bronze"]
    medal = ["1", "2", "3"]
    visual_order = [1, 0, 2]  # silver, gold, bronze
    cards = []
    for v in visual_order:
        if v >= len(ranked):
            continue
        agent = ranked[v]
        c = agent["conditions"][cond]
        cards.append(
            f'<div class="pcard {klass[v]}">'
            f'<div class="pmedal">{medal[v]}</div>'
            f'<div class="pname">{agent["name"]}</div>'
            f'<div class="pscore">{c["benchmark_score"]}</div>'
            f'<div class="plabel">benchmark score</div>'
            f'<div class="psub">{c["success"]["rate"]:.0%} success · '
            f'{cond}</div></div>')
    return (f'<section id="leaderboard" class="podium-wrap">'
            f'<h2>Podium — {cond}</h2>'
            f'<div class="podium">{"".join(cards)}</div></section>')


def render_html(results: Dict[str, Any], nav_link: str = "") -> str:
    suite = results["suite"]
    agents = results["agents"]
    conditions = list(suite["conditions"])
    scenarios = suite["scenarios"]

    parts = [f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{suite['name']} v{suite['version']} — Leaderboard</title>
{THREE_CDN}
<style>{CSS}</style></head><body>
{_hero(suite, agents, conditions, scenarios)}
<div class="wrap">
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

    # Top-3 podium for the first condition, doubling as the scroll-cue anchor.
    parts.append(_podium(agents, conditions[0]))

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
            tr_cls = f' class="top{rank}"' if rank <= 3 else ""
            rows.append(f"""<tr{tr_cls}>
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
</div>""")
    # Progressive enhancement scripts live at the end of <body>; with JS or
    # WebGL unavailable the page above is already complete and correct.
    parts.append("<script>" + HERO_JS + "\n" + REVEAL_JS
                 + "</script></body></html>")
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
