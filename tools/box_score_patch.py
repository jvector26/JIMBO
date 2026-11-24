"""Box score page (session 59, 2026-11-24, user request). Applied to site/index.html AND work/code/dash_v2.html.
Do not reapply (asserts each anchor occurs exactly once and the new code is absent).
Tonight tab > a past day: tap a final -> both teams' box scores (MIN PTS REB AST STL BLK TO FG 3P FT PF +/- GS),
default sort minutes (most first), tap any header to sort both tables, tap again to reverse. Team totals + shooting %,
didn't-play list. Tap a player = his page with that game open (game score detail).
Data: gamelog.json (pipeline/season_log.py): player rows, new col pf, new "n" {player_id: name} for unprojected players."""
import sys

FILES = sys.argv[1:] or ["site/index.html", "work/code/dash_v2.html"]
E = []
def rep(a, b): E.append((a, b))

# ---------- CSS
rep(""".tnbest .sc{text-align:right;font-family:var(--cond)}""",
    """.game[data-bx]{cursor:pointer} .game[data-bx]:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.bxsec{margin:18px 0 0} .bxsec h3{margin:0;font-size:17px} .bxsec h3 .note{font-weight:400;font-size:13px}
.bxwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:2px solid var(--black);background:var(--surface);margin:6px 0 4px}
.bxt{width:100%;min-width:0}
.bxt th,.bxt td{padding:6px 6px}
.bxt th button{color:#fff;cursor:pointer;font:inherit;font-weight:600;min-height:28px}
.bxt th button[aria-sort]{text-decoration:underline;text-underline-offset:3px}
.bxt td{font-family:var(--cond);font-size:16px}
.bxt .bxn{position:sticky;left:0;z-index:1;text-align:left;font-family:var(--text);font-size:14px;max-width:180px;overflow:hidden;text-overflow:ellipsis}
.bxt td.bxn{background:var(--surface);box-shadow:inset -1px 0 0 var(--line)}
.bxt th.bxn{background:var(--purple);z-index:2}
.bxt td.bxn button{background:none;border:0;padding:0;font:inherit;color:inherit;text-align:left;cursor:pointer;text-decoration:underline;text-decoration-color:var(--line);text-underline-offset:3px}
.bxt .ns{display:none} .bxt .bxst{font-size:10.5px;color:var(--muted);margin-left:4px;font-weight:600}
.bxt tfoot td{font-weight:700;border-top:2px solid var(--black);border-bottom:0}
.bxt tfoot td.bxn{font-weight:700}
.bxpct{font-size:13px;margin:4px 0 0} .bxdnp{font-size:13px;margin:2px 0 0}
.bxsum{margin:2px 0 4px}
@media (max-width:640px){ .bxt .nf{display:none} .bxt .ns{display:inline} .bxt th,.bxt td{padding:6px 4px} .bxt td{font-size:15px} .bxt .bxn{font-size:13.5px;max-width:128px} }
.tnbest .sc{text-align:right;font-family:var(--cond)}""")

# ---------- past card: tappable
rep("""  return `<div class="game past"><div class="gtop"><span class="note">Final</span>""",
    """  return `<div class="game past" data-bx="${r[8]}" tabindex="0" role="button" aria-label="Box score: ${aw} at ${hm}"><div class="gtop"><span class="note">Final</span>""")
rep("""      <span>Final</span><b>${who(a,0)}</b><span></span></div>${verdict}</div>`; }""",
    """      <span>Final</span><b>${who(a,0)}</b><span></span></div>${verdict}<div class="gdet">Box score ›</div></div>`; }""")

# ---------- Tonight: click + keyboard
rep("""  const c=e.target.closest('.game[data-g]'); if(c) openGame(+c.dataset.g); });""",
    """  const bx=e.target.closest('.game[data-bx]'); if(bx){ openBox(+bx.dataset.bx); return; }
  const c=e.target.closest('.game[data-g]'); if(c) openGame(+c.dataset.g); });""")
rep("""$('tonight')&&$('tonight').addEventListener('keydown',e=>{ const c=e.target.closest('.game[data-g]');""",
    """$('tonight')&&$('tonight').addEventListener('keydown',e=>{ const bx=e.target.closest('.game[data-bx]'); if(bx&&e.target===bx&&(e.key==='Enter'||e.key===' ')){ e.preventDefault(); openBox(+bx.dataset.bx); return; }
  const c=e.target.closest('.game[data-g]');""")

# ---------- overlay plumbing (shares #gv with the game page)
rep("""function closeGame(){ GVF={}; gvKey=null;""", """function closeGame(){ GVF={}; gvKey=null; bxKey=null;""")
rep("""function renderGV(){
  if(!gvKey||!P) return;""", """function renderGV(){
  if(bxKey!=null){ renderBox(); return; }
  if(!gvKey||!P) return;""")
rep("""window.addEventListener('popstate',()=>{ if(pvKey) closePlayer(); else if(gvKey) closeGame(); });""",
    """window.addEventListener('popstate',()=>{ if(pvKey) closePlayer(); else if(gvKey||bxKey!=null) closeGame(); });""")
rep("""} else if(gvKey){ gvPushed?history.back():closeGame(); } });""",
    """} else if(gvKey||bxKey!=null){ gvPushed?history.back():closeGame(); } });""")
rep("""    if(gvKey){ if(pvKey) closePlayer(); closeGame(); }""", """    if(gvKey||bxKey!=null){ if(pvKey) closePlayer(); closeGame(); }""")
rep("""function seasAfter(){ try{ """, """function seasAfter(){ try{ if(bxKey!=null) renderBox(); """)
rep("""$('gv').addEventListener('click',e=>{ if(e.target.closest('#gvBack')){ gvPushed?history.back():closeGame(); return; }""",
    """$('gv').addEventListener('click',e=>{ if(e.target.closest('#gvBack')){ gvPushed?history.back():closeGame(); return; }
  const bs=e.target.closest('[data-bxs]'); if(bs){ const k=bs.dataset.bxs; bxAsc=k===bxSort?!bxAsc:false; bxSort=k; renderBox(k); return; }
  const bp=e.target.closest('[data-bxp]'); if(bp){ openGameScore(bp.dataset.bxp,bxKey); return; }""")

# ---------- the page
rep("""function openGame(id){""",
    """// ---------- box score page (session 59, 2026-11-24, user request): tap a final on a past Tonight day. Data: gamelog.json
// (player rows; "n" = names for players without a projection). Both teams; default sort minutes, most first; any header sorts both.
let bxKey=null, bxSort='min', bxAsc=false, BXG=null, BXI=null, BXK=null;
const BXC=[['min','MIN','Minutes'],['pts','PTS','Points'],['reb','REB','Rebounds (offensive + defensive)'],['ast','AST','Assists'],['stl','STL','Steals'],
  ['blk','BLK','Blocks'],['tov','TO','Turnovers'],['fg','FG','Field goals made-attempted'],['fg3','3P','Threes made-attempted'],['ft','FT','Free throws made-attempted'],
  ['pf','PF','Personal fouls'],['pm','+/−','Plus-minus: team margin while he was on the floor'],['gs','GS','Box game score: points added to the margin vs an average player in the same possessions']];
function bxIndex(){ const G=SEAS.g; if(BXG===G) return BXI; const m=new Map(), kOf=new Map();
  Object.entries(G.p).forEach(([pid,rows])=>rows.forEach(x=>{ let a=m.get(x[0]); if(!a) m.set(x[0],a=[]); a.push([pid,x]); }));
  if(P) P.rows.forEach(r=>{ const id=seasPid(r.p.k); if(id!=null&&!kOf.has(id)) kOf.set(id,r.p.k); });
  BXG=G; BXK=kOf; return (BXI=m); }
function openBox(id){
  if(!gvKey&&bxKey==null){ gvScroll=window.scrollY; try{ history.pushState({bx:id},''); gvPushed=true; }catch(e){ gvPushed=false; } }
  bxKey=id; bxSort='min'; bxAsc=false; document.body.classList.add('gvopen'); $('gv').hidden=false; renderBox();
  const top=$('gv').getBoundingClientRect().top+window.scrollY-10; window.scrollTo(0,Math.max(0,top)); $('gvBack')&&$('gvBack').focus({preventScroll:true}); }
function renderBox(focusKey){
  const id=bxKey; if(id==null) return;
  const sl=[...$('gv').querySelectorAll('.bxwrap')].map(w=>w.scrollLeft);   // keep the sideways scroll when re-sorting
  const vr=((APP&&APP.vg&&APP.vg.games)||[]).find(r=>r[8]===id), G=SEAS.st==='ok'?SEAS.g:null, g=G&&G.g[id];
  const date=g?g[0]:vr?ymd(vr[0]):'', hm=g?g[1]:vr?vr[1]:'', aw=g?g[2]:vr?vr[2]:'', hp=g?g[3]:vr?vr[6]:null, ap=g?g[4]:vr?vr[7]:null;
  const ds=date?new Date(date+'T12:00:00').toLocaleDateString(undefined,{weekday:'long', month:'long', day:'numeric'}):'';
  const hw=hp!=null&&ap!=null&&hp>ap;
  let h=`<div class="pvtop"><button class="back" id="gvBack">← Back</button><span class="note">${ds} · Final</span></div>
  <div class="gvh"><div class="gside"><span class="gpct" style="${hw?'color:var(--muted)':''}">${ap??''}</span><span class="gvteam">${aw}</span><span class="note">away</span></div>
    <span class="at">at</span><div class="gside"><span class="gpct" style="${hw?'':'color:var(--muted)'}">${hp??''}</span><span class="gvteam">${hm}</span><span class="note">home</span></div></div>`;
  if(vr){ const [,,,m,v,a]=vr, who=(x,k)=>Math.abs(x)<0.05?'even':`${x>0?hm:aw} by ${Math.abs(x).toFixed(k)}`;
    h+=`<div class="note gsub pgrid bxsum"><span>JIMBO</span><b>${who(m,1)}</b><span></span><span>Line</span><b>${v!=null?who(v,1):'–'}</b><span></span><span>Final</span><b>${who(a,0)}</b><span></span></div>`; }
  if(!G){ seasLoad(); $('gv').innerHTML=h+`<p class="note">${SEAS.st==='err'?"Couldn't load the box score. Check your connection.":'Loading the box score…'}</p>`; return; }
  if(!g){ $('gv').innerHTML=h+`<p class="note">This game's box score isn't in yet. It arrives with the next nightly update.</p>`; return; }
  const ix=Object.fromEntries(G.cols.map((c,i)=>[c,i])), L=bxIndex().get(id)||[], NM=G.n||{};
  const nameOf=(pid,k)=>{ const r=k&&byKey.get(k); return (r&&(r.n||r.p&&r.p.n))||NM[pid]||`Player ${pid}`; };
  const short=n=>{ const w=n.split(' '); return w.length>1?`${w[0][0]}. ${w.slice(1).join(' ')}`:n; };
  const v=(x,c)=>ix[c]!=null&&x[ix[c]]!=null?x[ix[c]]:null;
  const rowOf=(pid,x)=>{ const k=BXK.get(pid)||null, gso=v(x,'gso'), gsd=v(x,'gsd');
    return {pid, k, n:nameOf(pid,k), st:x[2]===1, min:v(x,'min')||0, pts:v(x,'pts'), reb:v(x,'oreb')+v(x,'dreb'), ast:v(x,'ast'), stl:v(x,'stl'), blk:v(x,'blk'),
      tov:v(x,'tov'), fgm:v(x,'fgm'), fga:v(x,'fga'), f3m:v(x,'fg3m'), f3a:v(x,'fg3a'), ftm:v(x,'ftm'), fta:v(x,'fta'), pf:v(x,'pf'), pm:v(x,'pm'),
      gs:gso==null||gsd==null?null:Math.round((gso+gsd)*10)/10}; };
  const sv=(r,c)=>c==='fg'?r.fgm+r.fga/1000:c==='fg3'?r.f3m+r.f3a/1000:c==='ft'?r.ftm+r.fta/1000:r[c];
  const ma=(m,a)=>`${m}-${a}`, pc=(m,a)=>a?`${(100*m/a).toFixed(1)}%`:'–';
  const head=tm=>`<thead><tr><th class="bxn">${tm}</th>${BXC.map(([c,lab,t])=>`<th><button data-bxs="${c}" title="${t}" ${c===bxSort?`aria-sort="${bxAsc?'ascending':'descending'}"`:''}>${lab}</button></th>`).join('')}</tr></thead>`;
  const cell=(r,c)=>{ switch(c){ case 'min': return Math.round(r.min); case 'fg': return ma(r.fgm,r.fga); case 'fg3': return ma(r.f3m,r.f3a); case 'ft': return ma(r.ftm,r.fta);
      case 'pm': return `<span class="${cls(r.pm)}">${fmt(r.pm,0)}</span>`; case 'gs': return r.gs==null?'<span class="muted">–</span>':`<b class="${cls(r.gs)}">${fmt(r.gs)}</b>`;
      default: return r[c]==null?'–':r[c]; } };
  const side=home=>{ const tm=home?hm:aw, pts=home?hp:ap, op=home?ap:hp, mine=L.filter(([,x])=>!!x[1]===home);
    const pl=mine.filter(([,x])=>x[2]>=0).map(([pid,x])=>rowOf(pid,x)), dnp=mine.filter(([,x])=>x[2]<0);
    pl.sort((a,b)=>{ const x=sv(a,bxSort), y=sv(b,bxSort); if(x==null||y==null) return (x==null)-(y==null)||b.min-a.min;
      return (bxAsc?x-y:y-x)||b.min-a.min; });
    const T={}; ['min','pts','reb','ast','stl','blk','tov','fgm','fga','f3m','f3a','ftm','fta','pf'].forEach(c=>T[c]=pl.reduce((s,r)=>s+(r[c]||0),0));
    const gsT=pl.some(r=>r.gs!=null)?Math.round(pl.reduce((s,r)=>s+(r.gs||0),0)*10)/10:null;
    const body=pl.map(r=>{ const nm=`<span class="nf">${esc(r.n)}</span><span class="ns">${esc(short(r.n))}</span>${r.st?'<span class="bxst" title="Starter">S</span>':''}`;
      return `<tr><td class="bxn">${r.k?`<button data-bxp="${esc(r.k)}" aria-label="${esc(r.n)}: open his game log">${nm}</button>`:nm}</td>${BXC.map(([c])=>`<td>${cell(r,c)}</td>`).join('')}</tr>`; }).join('');
    const foot=`<tfoot><tr><td class="bxn">Team</td><td>${Math.round(T.min)}</td><td>${T.pts}</td><td>${T.reb}</td><td>${T.ast}</td><td>${T.stl}</td><td>${T.blk}</td><td>${T.tov}</td>
      <td>${ma(T.fgm,T.fga)}</td><td>${ma(T.f3m,T.f3a)}</td><td>${ma(T.ftm,T.fta)}</td><td>${ix.pf!=null?T.pf:'–'}</td><td></td><td>${gsT==null?'':`<span class="${cls(gsT)}">${fmt(gsT)}</span>`}</td></tr></tfoot>`;
    const dn=dnp.map(([pid,x])=>`${esc(nameOf(pid,BXK.get(pid)))} (${esc(String(x[3]||'').toLowerCase())})`).join(', ');
    return `<div class="bxsec"><h3>${tm} <span class="note">${pts!=null&&op!=null?`${pts>op?'W':'L'} ${pts}-${op}`:''}</span></h3>
      <div class="bxwrap"><table class="bxt">${head(tm)}<tbody>${body}</tbody>${foot}</table></div>
      <p class="note bxpct">FG ${pc(T.fgm,T.fga)} · 3P ${pc(T.f3m,T.f3a)} · FT ${pc(T.ftm,T.fta)}</p>${dn?`<p class="note bxdnp">Didn't play: ${dn}</p>`:''}</div>`; };
  h+=side(false)+side(true);
  h+=`<p class="note" style="margin-top:14px">Sorted by ${(BXC.find(c=>c[0]===bxSort)||BXC[0])[2].split(':')[0].toLowerCase()}, ${bxAsc?'lowest':'most'} first. Tap a header to sort both teams, tap again to reverse. S = starter. GS = box game score: points added to the margin vs an average player in the same possessions (offense + defense). Tap a player for his game log.</p>`;
  $('gv').innerHTML=h;
  $('gv').querySelectorAll('.bxwrap').forEach((w,i)=>{ if(sl[i]) w.scrollLeft=sl[i]; });
  if(focusKey){ const b=$('gv').querySelector(`[data-bxs="${focusKey}"]`); b&&b.focus({preventScroll:true}); }
}
function openGame(id){""")


for f in FILES:
    s = open(f).read()
    assert "function renderBox(" not in s, f"{f}: already applied"
    for a, b in E:
        n = s.count(a)
        assert n == 1, f"{f}: anchor found {n}x: {a[:70]!r}"
        s = s.replace(a, b)
    open(f, "w").write(s)
    print("patched", f)
