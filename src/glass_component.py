"""Liquid Glass header for Streamlit — pastel light theme edition.

The WebGL glass effect (genuine html2canvas + Container shader) is preserved exactly.
Only the iframe backdrop colours are updated to match the new pastel palette.

Architecture notes (unchanged):
* WebGL refraction runs inside an <iframe> and can only refract the iframe's OWN background.
* One WebGL context per page keeps GPU load low on laptops.
* All other glass panels (sidebar, KPI cards, etc.) use CSS backdrop-filter in theme.css.
* If WebGL or html2canvas is unavailable the header falls back to a CSS frosted-glass look.
"""
from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "liquid-glass"

# Pastel-themed icon (sky-blue strokes, soft mint ring)
ICON_SVG = (
    '<svg viewBox="0 0 32 32" width="34" height="34" aria-hidden="true" '
    'fill="none" stroke="#4A90D9" stroke-width="1.6">'
    '<circle cx="16" cy="16" r="4.2"/>'
    '<path d="M16 3v5M16 24v5M3 16h5M24 16h5'
    'M7 7l3.5 3.5M21.5 21.5L25 25M25 7l-3.5 3.5M10.5 21.5L7 25"/>'
    '<circle cx="16" cy="16" r="11" stroke="#52B788" stroke-opacity=".65" stroke-dasharray="2 3"/>'
    '</svg>'
)


@lru_cache(maxsize=None)
def _asset(name: str) -> str:
    return (ASSET_DIR / name).read_text(encoding="utf-8").replace("</script", "<\\/script")


def build_header_html(
    title: str,
    subtitle: str,
    status: str = "Local mode — no cloud services",
    height: int = 132,
) -> str:
    t, s, st = html.escape(title), html.escape(subtitle), html.escape(status)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<style>{_asset('glass.css')}</style>
<style>
html,body{{height:100%;margin:0;overflow:hidden;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;}}

/* Pastel soft backdrop that the WebGL glass refracts */
body{{background:
  radial-gradient(400px 220px at 10% 30%,  rgba(74,144,217,.35), transparent 70%),
  radial-gradient(340px 200px at 88% 70%,  rgba(82,183,136,.22), transparent 70%),
  radial-gradient(280px 180px at 60% 10%,  rgba(124,92,191,.18), transparent 70%),
  repeating-linear-gradient(0deg,  rgba(180,210,240,.08) 0 1px, transparent 1px 22px),
  repeating-linear-gradient(90deg, rgba(180,210,240,.08) 0 1px, transparent 1px 22px),
  linear-gradient(135deg, #C8E6FF 0%, #D8F0E6 50%, #E4DCFF 100%);}}

#wrap{{position:absolute;inset:0;padding:10px 4px 6px 4px;box-sizing:border-box}}
.glass-container{{width:100%;height:calc(100% - 4px);padding:0;box-sizing:border-box;
  justify-content:flex-start;
  border:1px solid rgba(255,255,255,.70);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.80);
  background:rgba(255,255,255,.25);backdrop-filter:blur(4px)}}
body.fallback .glass-container{{
  background:rgba(255,255,255,.60);
  backdrop-filter:blur(18px) saturate(160%);
  -webkit-backdrop-filter:blur(18px) saturate(160%)}}

.inner{{display:flex;align-items:center;gap:16px;width:100%;
  padding:0 26px;box-sizing:border-box;color:#263449;position:relative;z-index:2}}
.inner h1{{margin:0;font-size:24px;font-weight:700;letter-spacing:.2px;
  color:#1A2E44;text-shadow:0 1px 4px rgba(255,255,255,.70)}}
.inner .eyebrow{{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;
  color:#4A90D9;margin-bottom:3px;font-weight:650}}
.inner p{{margin:4px 0 0;font-size:13.5px;color:#68758A;
  text-shadow:0 1px 3px rgba(255,255,255,.60)}}
.grow{{flex:1;min-width:0}}
.pills{{display:flex;flex-direction:column;gap:6px;align-items:flex-end}}
.pill{{font-size:11.5px;padding:3px 12px;border-radius:999px;
  border:1px solid rgba(74,144,217,.35);
  background:rgba(255,255,255,.65);
  color:#263449;white-space:nowrap;
  backdrop-filter:blur(6px)}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:#52B788;margin-right:6px;box-shadow:0 0 6px rgba(82,183,136,.60)}}
@media (max-width:620px){{
  .pills{{display:none}}.inner h1{{font-size:19px}}.inner p{{display:none}}}}
</style></head>
<body><div id="wrap" role="banner"></div>
<script>{_asset('html2canvas.min.js')}</script>
<script>{_asset('container.js')}</script>
<script>
(function(){{
  var wrap=document.getElementById('wrap');
  var inner=document.createElement('div'); inner.className='inner';
  inner.innerHTML='<div aria-hidden="true">{ICON_SVG.replace("'", "\\'")}</div>'+
   '<div class="grow"><div class="eyebrow">Industrial Knowledge Intelligence</div>'+
   '<h1>{t}</h1><p>{s}</p></div>'+
   '<div class="pills"><span class="pill"><span class="dot"></span>{st}</span>'+
   '<span class="pill" id="mode">Glass: initialising…</span></div>';
  var modeEl=function(){{return document.getElementById('mode')}};
  function fallback(reason){{
    document.body.classList.add('fallback');
    if(!document.querySelector('.glass-container')){{
      var d=document.createElement('div');d.className='glass-container';
      d.appendChild(inner);wrap.appendChild(d);}}
    if(modeEl()) modeEl().textContent='Glass: CSS fallback';
    if(modeEl()) modeEl().title=reason||'';
  }}
  function webglOK(){{
    try{{var c=document.createElement('canvas');
      return !!(c.getContext('webgl')||c.getContext('experimental-webgl'));}}
    catch(e){{return false;}}
  }}
  try {{
    if(!webglOK()||typeof Container!=='function'||typeof html2canvas!=='function'){{
      fallback('WebGL or library unavailable'); return;
    }}
    window.glassControls={{
      edgeIntensity:0.012, rimIntensity:0.06, baseIntensity:0.008,
      edgeDistance:0.15,   rimDistance:0.8,   baseDistance:0.1,
      cornerBoost:0.02,    rippleEffect:0.08,  blurRadius:4.0, tintOpacity:0.3
    }};
    var glass=new Container({{borderRadius:26, type:'rounded', tintOpacity:0.22}});
    glass.element.appendChild(inner);
    wrap.appendChild(glass.element);
    glass.updateSizeFromDOM();
    var tid;
    window.addEventListener('resize',function(){{
      clearTimeout(tid);
      tid=setTimeout(function(){{glass.updateSizeFromDOM();}},150);
    }});
    var tries=0, iv=setInterval(function(){{
      tries++;
      if(Container.pageSnapshot&&glass.webglInitialized){{
        clearInterval(iv); modeEl().textContent='Glass: WebGL refraction';
      }} else if(tries>=16){{
        clearInterval(iv); fallback('snapshot not produced in 4 s');
      }}
    }},250);
  }} catch(err) {{ fallback(String(err)); }}
}})();
</script></body></html>"""


def render_html_frame(page_html: str, height: int) -> None:
    """Embed a self-contained HTML document in an iframe.

    Uses st.iframe (current API) and falls back to the deprecated
    components.html on older Streamlit builds.
    Only pass trusted HTML — anything derived from uploaded documents
    must be HTML-escaped first.
    """
    import streamlit as st
    if hasattr(st, "iframe"):
        st.iframe(page_html, height=height)
    else:  # pragma: no cover — older Streamlit
        import streamlit.components.v1 as components
        components.html(page_html, height=height, scrolling=False)


def render_glass_header(
    title: str,
    subtitle: str,
    status: str = "Local mode — no cloud services",
    height: int = 132,
) -> None:
    render_html_frame(build_header_html(title, subtitle, status, height), height)
