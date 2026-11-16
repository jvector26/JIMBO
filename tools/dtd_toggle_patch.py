"""Day-to-day what-if toggles on the game page (session 54, 2026-11-16, user request, display only).
Applies the same edits to site/index.html and work/code/dash_v2.html. Run once; refuses if already applied."""
import sys, pathlib

R = pathlib.Path(__file__).resolve().parent.parent
FILES = [R / 'site/index.html', R / 'work/code/dash_v2.html']

EDITS = [
# CSS
(""".game[data-g]{cursor:pointer}""",
 """.game[data-g]{cursor:pointer}
.dtdrow{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;padding:10px 0;border-top:1px solid var(--line,#ddd)}
.dtdrow:first-of-type{border-top:0}
.dtdrow .who{flex:1 1 180px;min-width:0}
.dtdrow .who b{font-weight:700}
.dtdseg{display:inline-flex;border:2px solid var(--black);border-radius:6px;overflow:hidden}
.dtdseg button{font:inherit;font-size:13px;padding:5px 11px;border:0;border-left:2px solid var(--black);background:transparent;color:inherit;cursor:pointer;min-height:32px}
.dtdseg button:first-child{border-left:0}
.dtdseg button[aria-pressed="true"]{background:var(--black);color:var(--bg,#FBF8F1);font-weight:700}
.dtdswing{flex:1 1 100%;font-size:13px;color:var(--muted)}
.dtdswing b{color:var(--ink,inherit)}
.dtdwhatif{background:var(--gold);color:var(--black);font-size:12px;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:6px}"""),
# gameStrength: optional force map {player key: 1 in | 0 out}; DTD players flagged dk
("""function gameStrength(){ // per team""",
 """function gameStrength(force){ // force (game page what-if, session 54): {player key: 1 plays (minutes when playing) | 0 out}; others as below. Per team"""),
("""      if(s.startsWith('out')||s.includes('suspen')) return 0; return r.z&&r.z.mc!=null?r.z.mc:(r.me||0); });""",
 """      if(s.startsWith('out')||s.includes('suspen')) return 0; if(force&&force[r.p.k]===0) return 0; return r.z&&r.z.mc!=null?r.z.mc:(r.me||0); });"""),
("""    const dtd=R.map((r,i)=>i).filter(i=>!lock[i]&&base[i]>0&&/day/i.test(R[i].z&&R[i].z.status||'')).slice(0,8);""",
 """    const isd=R.map((r,i)=>r.o.g!==0&&!lock[i]&&/day/i.test(r.z&&r.z.status||'')&&(r.z&&r.z.mc!=null?r.z.mc:(r.me||0))>0);
    const dtd=R.map((r,i)=>i).filter(i=>isd[i]&&base[i]>0&&!(force&&force[R[i].p.k]!=null)).slice(0,8);"""),
("""    rot[x.t]=R.map((r,i)=>({r, em:em[i], ifp:dtd.includes(i)?trim240(base,lock,key)[i]:null}));""",
 """    rot[x.t]=R.map((r,i)=>({r, em:em[i], ifp:dtd.includes(i)?trim240(base,lock,key)[i]:null, dk:isd[i]}));"""),
# tonightGames / allGames pass the force map through
("""function tonightGames(list, day){""", """function tonightGames(list, day, force){"""),
("""  const G=gameStrength(), T=""", """  const G=gameStrength(force), T="""),
("""function allGames(){ return tonightGames().concat(...(SZ.upcoming||[]).map(u=>tonightGames(u.games, ymd(u.day)))); }""",
 """function allGames(force){ return tonightGames(null, null, force).concat(...(SZ.upcoming||[]).map(u=>tonightGames(u.games, ymd(u.day), force))); }
let GVF={};   // game page what-if for day-to-day players {key: 1 in | 0 out}; page only, never saved, cleared when the game page closes or changes"""),
# reset on open/close
("""  gvKey=id; document.body.classList.add('gvopen');""", """  if(gvKey!==id) GVF={}; gvKey=id; document.body.classList.add('gvopen');"""),
("""function closeGame(){ gvKey=null;""", """function closeGame(){ GVF={}; gvKey=null;"""),
# renderGV: game with toggles applied
("""  const g=allGames().find(x=>x.game===gvKey); if(!g){ closeGame(); return; }""",
 """  const g=allGames(GVF).find(x=>x.game===gvKey); if(!g){ closeGame(); return; }"""),
# DTD section above the uncertainty section
("""  // uncertainty""",
 """  h+=dtdHTML(g);
  // uncertainty"""),
# click handling
("""$('gv').addEventListener('click',e=>{ if(e.target.closest('#gvBack')){ gvPushed?history.back():closeGame(); return; }""",
 """$('gv').addEventListener('click',e=>{ if(e.target.closest('#gvBack')){ gvPushed?history.back():closeGame(); return; }
  const dd=e.target.closest('[data-dtd]'); if(dd){ const v=dd.dataset.v; if(v==='h') delete GVF[dd.dataset.dtd]; else GVF[dd.dataset.dtd]=+v; renderGV(); return; }
  if(e.target.closest('[data-dtdreset]')){ GVF={}; renderGV(); return; }"""),
# dtdHTML before renderGV
("""function renderGV(){""",
 """function dtdHTML(g){ // day-to-day what-if (session 54): each DTD player in / 50-50 / out, with the in-vs-out swing (other DTD players held as set)
  const list=[...g.A.rot.map(q=>[q,g.A]), ...g.H.rot.map(q=>[q,g.H])].filter(([q])=>q.dk); if(!list.length) return '';
  const one=f=>tonightGames([g], g.day, f)[0];
  const rows=list.map(([q,S])=>{ const r=q.r, k=r.p.k, cur=GVF[k]==null?'h':String(GVF[k]), home=S===g.H;
    const pin=one({...GVF,[k]:1}), pout=one({...GVF,[k]:0}), w=x=>home?x.ph:1-x.ph;
    const mIn=(pin.H.rot.concat(pin.A.rot).find(x=>x.r.p.k===k)||{}).em||0, d=Math.round((w(pin)-w(pout))*100);
    const b=(v,lab)=>`<button data-dtd="${esc(k)}" data-v="${v}" aria-pressed="${cur===v}">${lab}</button>`;
    return `<div class="dtdrow"><div class="who"><b>${esc(r.p.n)}</b> <span class="note">${S.t}${injWhat(r.z)?` · ${esc(injWhat(r.z))}`:''}</span></div>
      <div class="dtdseg" role="group" aria-label="${esc(r.p.n)} tonight">${b('1','In')}${b('h','50-50')}${b('0','Out')}</div>
      <div class="dtdswing">${S.t} win chance: <b>${pct(w(pin))}</b> if he plays (${Math.round(mIn)} min) · <b>${pct(w(pout))}</b> if he sits · swing <b>${d} pts</b></div></div>`; }).join('');
  const any=Object.keys(GVF).length;
  return `<div class="pvsec"><h3>Day-to-day players${any?'<span class="dtdwhatif">what-if</span>':''}</h3>
    <p class="note">JIMBO counts each day-to-day player as a coin flip. Set him in or out to see the odds if he plays (his usual minutes, teammates trimmed to fit 240) or sits (his minutes go to teammates); everything on this page updates. ${any?`Showing your what-if; the Tonight card and the vs Vegas log keep the 50-50 number. <button class="linkbtn" data-dtdreset="1">Reset</button>`:'Only this page changes; nothing is saved.'}</p>
    ${rows}</div>`;
}
function renderGV(){"""),
]

def main():
    for f in FILES:
        s = f.read_text()
        if 'function dtdHTML' in s:
            sys.exit(f'already applied: {f}')
        for old, new in EDITS:
            n = s.count(old)
            if n != 1:
                sys.exit(f'{f.name}: expected 1 match, found {n}: {old[:60]!r}')
            s = s.replace(old, new)
        f.write_text(s)
        print('patched', f)

if __name__ == '__main__':
    main()
