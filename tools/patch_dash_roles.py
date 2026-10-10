#!/usr/bin/env python3
"""Role display + five-starter rule for the dashboard / app page (user request 2026-10-09).

- Team view (and player view / player detail) drop the static "depth chart: <tier>" note and show the player's CURRENT
  role: plain = chart default, bold purple = the user's role edit, bold italic magenta "(auto)" = moved automatically.
- Always five starters per team: engine balanceRoles/assignRoles (same rule as pipeline/nightly.py via
  model/depth_chart.py balance_starters). User roles are kept; short -> best non-starter by chart (R, then L, then N; most
  base minutes m0) moves up to S; long -> the chart starter with the fewest m0 moves down to R. An exempt (X, e.g.
  unsigned) player fills a starter slot when the chart lists fewer than five.

Usage: python3 tools/patch_dash_roles.py FILE [FILE ...]   (dashboard.html, site/index.html, or the dashboard template
in the sandbox zip; the JS is identical in all three). Idempotent: a file that already has the patch is skipped.
"""
import sys

MARK = "function balanceRoles(items)"

R = [
# --- CSS
("/* game detail page (app) */",
 "/* role labels (2026-10-09): your edit = bold purple, automatic five-starter move = bold italic magenta */\n"
 ".rl-u{color:var(--gold);font-weight:700}\n.rl-a{color:var(--magenta);font-weight:700;font-style:italic}\n"
 "/* game detail page (app) */"),
# --- editor note
('id="edDnote">Role sets the minutes the model starts from (starter, rotation, limited, out of the rotation); the team\'s minutes are re-shared around it.',
 'id="edDnote">Role sets the minutes the model starts from (starter, rotation, limited, out of the rotation); the team\'s minutes are re-shared around it. A team always has five starters: move a starter down and the next man up (by the depth chart, then projected minutes) starts in his place; move someone up and the chart starter with the fewest projected minutes goes to the rotation.'),
# --- engine: rule
("  // team minutes: overridden players are locked;",
 """  // always five starters (user 2026-10-09; same rule as model/depth_chart.py balance_starters in the nightly): user roles
  // are kept; short -> best non-starter by chart (R, then L, then N; most base minutes m0) moves up to S; long -> the chart
  // starter with the fewest m0 moves down to R. An exempt (X) player fills a starter slot when the chart lists fewer than five.
  function balanceRoles(items){ // items: {k, c: chart tier, u: user role, m0} -> Map k -> {d, u, a}
    const TRK={R:0,L:1,N:2}, out=new Map(); items.forEach(it=>out.set(it.k,{d:it.u||it.c||null, u:!!it.u, a:false}));
    const nS=items.filter(it=>it.c==='S').length, nX=items.filter(it=>it.c==='X'&&!it.u).length;
    const target=5-Math.min(nX,Math.max(0,5-nS)), m=it=>it.m0==null?0:it.m0, kc=(a,b)=>a.k<b.k?-1:a.k>b.k?1:0;
    let cur=items.filter(it=>out.get(it.k).d==='S').length;
    if(cur<target){ const c=items.filter(it=>!it.u&&TRK[it.c]!=null).sort((a,b)=>TRK[a.c]-TRK[b.c]||m(b)-m(a)||kc(a,b));
      for(const it of c){ if(cur>=target) break; out.set(it.k,{d:'S',u:false,a:true}); cur++; } }
    else if(cur>target){ const c=items.filter(it=>!it.u&&it.c==='S').sort((a,b)=>m(a)-m(b)||kc(a,b));
      for(const it of c){ if(cur<=target) break; out.set(it.k,{d:'R',u:false,a:true}); cur--; } }
    return out; }
  function assignRoles(list, chartOf){ // sets r.role (S/R/L/N/X or null), r.roleU (user edit), r.roleA (automatic move)
    const B=balanceRoles(list.map(r=>({k:r.p.k, c:chartOf(r)||null, u:r.o.d||null, m0:r.p.m0})));
    list.forEach(r=>{ const x=B.get(r.p.k); r.role=x.d; r.roleU=x.u; r.roleA=x.a; }); }
  // team minutes: overridden players are locked;"""),
# --- engine: project() roles
("parts:r.parts, o}; });\n    const byTeam={}; TEAMS.forEach(t=>byTeam[t]=[]); rows.forEach(r=>{ if(byTeam[r.t]) byTeam[r.t].push(r); });",
 "parts:r.parts, o, role:o.d||null, roleU:!!o.d, roleA:false}; });\n    const byTeam={}; TEAMS.forEach(t=>byTeam[t]=[]); rows.forEach(r=>{ if(byTeam[r.t]) byTeam[r.t].push(r); });\n"
 "    TEAMS.forEach(t=>assignRoles(byTeam[t], r=>r.o.t?null:r.p.dt));"),
("const a=allocate(list.map(r=>({k:r.p.k, mm:wantOf(r.p,r.o)})), locked);",
 "const a=allocate(list.map(r=>({k:r.p.k, mm:wantOf(r.p,(r.roleU||r.roleA)?{d:r.role}:{})})), locked);"),
("return {project, simulate, backtest, Phi, PhiInv, TEAMS, TI, rng, wantOf, allocate, poDelta};",
 "return {project, simulate, backtest, Phi, PhiInv, TEAMS, TI, rng, wantOf, allocate, poDelta, assignRoles, balanceRoles};"),
# --- in-season overlay: roles from the nightly's chart (z.dc), then the existing role re-share uses them
("if(byT[r.t]) byT[r.t].roster.push(r); });",
 "if(byT[r.t]) byT[r.t].roster.push(r); });\n"
 "  P.rows.forEach(r=>{ if(!byT[r.t]){ r.role=r.o.d||null; r.roleU=!!r.o.d; r.roleA=false; } });\n"
 "  P.team.forEach(x=>E.assignRoles(x.roster, r=>r.o.t?null:(r.z?r.z.dc:r.p.dt)));"),
("const a=r.o.d||r.z.dc, b=r.z.dt;", "const a=r.role||r.z.dc, b=r.z.dt;"),
("tier(r,r.o.d||r.z.dc||r.z.dt)", "tier(r,r.role||r.z.dc||r.z.dt)"),
("tier(r,r.z.dt||r.o.d||r.z.dc)", "tier(r,r.z.dt||r.role||r.z.dc)"),
# --- display helpers + team roster row
("function rosterHTML(x){",
 """function noteRest(n){ return String(n||'').replace(/(^|;\\s*)depth chart: [a-z ]+/g,'').replace(/^;\\s*/,'').trim(); }
function roleTag(r){ const L={S:'Starter',R:'Rotation',L:'Limited minutes',N:'Out of rotation'}[r.role]; if(!L) return '';
  if(r.roleU) return `<span class="rl rl-u" title="Your role edit">${L}</span>`;
  if(r.roleA) return `<span class="rl rl-a" title="Moved automatically to keep five starters">${L} (auto)</span>`;
  return `<span class="rl">${L}</span>`; }
function roleNote(r){ const a=roleTag(r), b=noteRest(r.p.note); return a&&b?`${a} · ${esc(b)}`:a||esc(b); }
function chNR(r){ return Object.keys(r.o).some(k=>k!=='d'); }  // edits other than role (role edits show in the role label)
function rosterHTML(x){"""),
("const row=r=>{ const ch=Object.keys(r.o).length>0, z0=!inRot(r);",
 "const row=r=>{ const ch=chNR(r), z0=!inRot(r), rn=roleNote(r);"),
("${injBadge(r)}${r.p.note?`<span class=\"tm\">${esc(r.p.note)}</span>`:''}</td>",
 "${injBadge(r)}${rn?`<span class=\"tm\">${rn}</span>`:''}</td>"),
("${x.over>0?`<p class=\"warn\">",
 "${x.roster.filter(r=>r.roleU&&r.role==='S').length>5?`<p class=\"warn\">You have set ${x.roster.filter(r=>r.roleU&&r.role==='S').length} starters for ${x.t}; a team starts five.</p>`:''}"
 "${x.over>0?`<p class=\"warn\">"),
# --- players tab detail row, player view subtitle
("(r.p.note?`<div class=\"drow\"><span></span><span>${esc(r.p.note)}</span><b></b></div>`:'')",
 "(roleNote(r)?`<div class=\"drow\"><span></span><span>${r.role&&r.role!=='X'?'Role: ':''}${roleNote(r)}</span><b></b></div>`:'')"),
("${p.note?` · ${esc(p.note)}`:''}${Object.keys(r.o).length?' · <span class=\"ov\">edited</span>':''}",
 "${roleNote(r)?` · ${roleNote(r)}`:''}${chNR(r)?' · <span class=\"ov\">edited</span>':''}"),
# --- editor: default option says what the default is, without "depth chart"
("function edFill(k){ const p=byKey.get(k), ct=chartTier(k);\n  $('edD').innerHTML=`<option value=\"\">${ct?'Depth chart: '+TN[ct]:'Model (not on a depth chart)'}</option>`",
 "function edFill(k){ const p=byKey.get(k), ct=chartTier(k), r=P.rows.find(x=>x.p.k===k);\n"
 "  const dl=r&&r.roleA?`Default: ${TN[r.role]} (auto, keeps five starters)`:ct&&ct!=='X'?'Default: '+TN[ct]:'Default: model minutes';\n"
 "  $('edD').innerHTML=`<option value=\"\">${dl}</option>`"),
]


def patch(path):
    s = open(path, encoding="utf-8").read()
    if MARK in s:
        print(f"{path}: already patched"); return
    for old, new in R:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{path}: anchor found {n} times (need 1): {old[:90]!r}")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print(f"{path}: patched ({len(R)} edits)")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        patch(f)
