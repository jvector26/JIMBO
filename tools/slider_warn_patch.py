"""Session 56 (2026-11-20, user request): visible notice when any weight slider is off its default.
Banner under the header (every tab) + 'your sliders' tag on Tonight card JIMBO lines. Applies the same edits to the
template (work/code/dash_v2.html) and the built page (site/index.html). Run once; refuses to reapply."""
import sys
FILES = ["work/code/dash_v2.html", "site/index.html"]
EDITS = [
 # CSS
 (".fresh{font-size:13.5px;font-weight:600;color:var(--purple);margin:0 0 8px}",
  ".fresh{font-size:13.5px;font-weight:600;color:var(--purple);margin:0 0 8px}\n"
  ".slwarn{font-size:13px;background:var(--yellow);color:var(--black);border:2px solid var(--black);border-radius:6px;padding:6px 10px;margin:0 0 10px;box-shadow:3px 3px 0 var(--shadow2)}\n"
  ".slwarn button{font:inherit;font-weight:700;text-decoration:underline;background:none;border:0;padding:0 2px;color:inherit;cursor:pointer}\n"
  ".sltag{display:inline-block;font-size:11px;font-weight:700;background:var(--yellow);color:var(--black);border:1px solid var(--black);border-radius:3px;padding:0 4px;margin-left:4px}"),
 # banner element
 ('<p class="fresh" id="fresh" hidden></p>',
  '<p class="fresh" id="fresh" hidden></p>\n  <p class="slwarn" id="slwarn" role="status" hidden></p>'),
 # banner render (inside renderSetSum, which every change already calls)
 ("function renderSetSum(){ const nw=",
  "function slMoved(){ return Object.keys(DEF).filter(k=>!(PURE&&k==='bm')&&Math.abs((S[k]==null?DEF[k]:S[k])-DEF[k])>(k==='bm'?0.004:1e-9)); }\n"
  "function renderSlWarn(){ const el=$('slwarn'); if(!el) return; const mv=slMoved(); el.hidden=!mv.length; if(!mv.length){ el.innerHTML=''; return; }\n"
  "  const lab=k=>k==='bm'?'Model share':k==='age'?'Age adjustment':(LABEL[k]||k), val=k=>k==='bm'?Math.round(S[k]*100)+'%':'\\u00d7'+(S[k]/(k.startsWith('lineup')?DEF[k]:1)).toFixed(2);\n"
  "  el.innerHTML=`<b>Sliders moved:</b> ${mv.map(k=>`${esc(lab(k))} ${val(k)}`).join(', ')}. Forecasts and Tonight cards use them; the vs Vegas record uses the default model. <button type=\"button\" id=\"slwShow\">Show sliders</button>`;\n"
  "  $('slwShow').addEventListener('click',()=>{ const d=$('settings'); d.open=true; d.scrollIntoView({behavior:'smooth',block:'start'}); }); }\n"
  "function renderSetSum(){ renderSlWarn(); const nw="),
 # Tonight card tag
 ("<div class=\"note gsub\">JIMBO: ${fav} by ${Math.abs(g.sp).toFixed(1)}",
  "<div class=\"note gsub\">JIMBO${slMoved().length?'<span class=\"sltag\" title=\"Uses your moved sliders\">your sliders</span>':''}: ${fav} by ${Math.abs(g.sp).toFixed(1)}"),
]
for f in FILES:
    s = open(f).read()
    if 'id="slwarn"' in s: sys.exit(f"{f}: already patched")
    for a, b in EDITS:
        if s.count(a) != 1: sys.exit(f"{f}: anchor count {s.count(a)}: {a[:60]}")
        s = s.replace(a, b)
    open(f, "w").write(s); print("patched", f)
