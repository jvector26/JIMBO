"""Box game score in the app (session 58, 2026-11-23, user request). Applied to site/index.html AND work/code/dash_v2.html.
Do not reapply (asserts each anchor occurs exactly once and the new code is absent).
  1. Player page game log: GS column (offense + defense game score, points) + O / D in the tap-open detail.
  2. Players tab: third view "Top games" (every game this season, sortable by game score / offense / defense, reversible).
  3. Tonight tab: "Last night's best" (top 5 game scores of the latest night) on the Tonight view; "Best games" on past days.
Data: gamelog.json cols gso / gsd (pipeline/season_log.py, model/jimbo_update.game_scores)."""
import sys

FILES = sys.argv[1:] or ["site/index.html", "work/code/dash_v2.html"]
E = []
def rep(a, b): E.append((a, b))

# ---------- CSS
rep(""".glh,.glr{display:grid;grid-template-columns:minmax(0,1fr) 78px 34px 32px 32px 32px 38px;gap:2px;align-items:center;text-align:right}""",
    """.glh,.glr{display:grid;grid-template-columns:minmax(0,1fr) 78px 34px 32px 32px 32px 38px 42px;gap:2px;align-items:center;text-align:right}
.glr .gsc{font-weight:700}
.tgtab td .tm{display:block}
.tgtab .line{display:block;font-size:13px;color:var(--muted);font-family:var(--cond);white-space:nowrap}
.tgtab th button[aria-sort]{text-decoration:underline;text-underline-offset:3px}
.tgtab .pa{font-family:var(--cond);font-size:19px;font-weight:700}
.tgtab td.l{white-space:normal}
@media (max-width:640px){ .tgtab td,.tgtab th{padding-left:3px;padding-right:3px} .tgtab td.num{font-size:15px} .tgtab .pa{font-size:17px} }
.tnbest{background:var(--surface);border:2px solid var(--black);box-shadow:4px 4px 0 var(--shadow2);padding:10px 12px 8px;margin:0 0 16px}
.tnbest h3{margin:0;font-size:16px}
.tnbest ol{list-style:none;margin:6px 0 0;padding:0}
.tnbest li{border-top:1px solid var(--line)}
.tnbest li:first-child{border-top:0}
.tnbest button{display:grid;grid-template-columns:20px minmax(0,1fr) auto;gap:8px;align-items:center;width:100%;background:none;border:0;padding:7px 0;min-height:48px;text-align:left;font:inherit;color:inherit;cursor:pointer}
.tnbest button:hover{background:#FFF7CF}
.tnbest .rk{font-family:var(--cond);font-weight:700;color:var(--muted)}
.tnbest .who b{display:block;font-size:15px} .tnbest .who small{display:block;font-size:12px;color:var(--muted);font-family:var(--cond)}
.tnbest .sc{text-align:right;font-family:var(--cond)} .tnbest .sc b{display:block;font-size:20px} .tnbest .sc small{font-size:11.5px;color:var(--muted);white-space:nowrap}""")
rep(""".glh,.glr{grid-template-columns:minmax(0,1fr) 70px 30px 28px 28px 28px 34px;padding-left:6px;padding-right:6px}""",
    """.glh,.glr{grid-template-columns:minmax(0,1fr) 66px 24px 26px 26px 24px 32px 40px;gap:3px;padding-left:6px;padding-right:6px}
  .glr{font-size:15px} .glr .res{font-size:13px} .glr .gsc{font-size:16px}""")

# ---------- Players tab: view switch, table, note
rep("""<button data-plv="season" aria-pressed="false">This season</button></div>""",
    """<button data-plv="season" aria-pressed="false">This season</button><button data-plv="games" aria-pressed="false">Top games</button></div>""")
rep("""      <p class="note" id="sNote" hidden>""",
    """      <div class="tbl" id="gTbl" hidden>
        <table class="stab tgtab">
          <thead><tr>
            <th class="l">#</th><th class="l">Player · game<span class="sub">pts · reb · ast · min</span></th>
            <th><button data-gsort="o" title="Box game score, offense: points added vs an average player in the same possessions">Off</button></th>
            <th><button data-gsort="d" title="Box game score, defense: steals, blocks, defensive rebounds, fouls and opponents' shots while he was on the floor">Def</button></th>
            <th><button data-gsort="t" aria-sort="descending" title="Box game score: offense + defense, points added to the margin vs an average player">Game<br>score</button></th>
          </tr></thead>
          <tbody id="grows"></tbody>
        </table>
      </div>
      <p class="note" id="gNote" hidden>Game score = what one game's box score adds to his team's margin, in points, compared with an average player in the same minutes. It uses the same box-score weights as the player ratings (shooting is scored as makes above the league rate on his attempts), so a player's game scores add up to his season box rating × possessions ÷ 100. Lineup results aren't in it. Off and Def are the two halves; box defense is thin in a single game (mostly steals, blocks, defensive rebounds and opponents' shots while he was on the floor), so the defensive list is noisier. Tap a header to sort, tap again to reverse. Team = the team he played for that night. Updated nightly.</p>
      <p class="note" id="sNote" hidden>""")

# ---------- Players tab: render + state
rep("""function renderPlayers(){
  if(!P) return;
  if(SEASON&&plView==='season'){ renderSeasonPlayers(); return; }""",
    """// ---------- box game score (session 58, 2026-11-23): gamelog.json gso / gsd, points vs an average player in the same possessions
let gSort='t', gAsc=false, GTOP=null;
function gameScores(){ // every played game with a score: {k, pid, gid, date, tm, opp, home, w, score, x, o, d, t}
  if(GTOP) return GTOP; if(SEAS.st!=='ok') return null;
  const G=SEAS.g, ix=Object.fromEntries(G.cols.map((c,i)=>[c,i])); if(ix.gso==null) return (GTOP=[]);
  const kOf=new Map(); P.rows.forEach(r=>{ const id=seasPid(r.p.k); if(id!=null&&!kOf.has(id)) kOf.set(id,r.p.k); });
  const L=[]; Object.entries(G.p).forEach(([pid,rows])=>{ const k=kOf.get(pid); if(!k) return;
    rows.forEach(x=>{ if(x[2]<0||x[ix.gso]==null) return; const g=G.g[x[0]]; if(!g) return; const home=!!x[1], mp=home?g[3]:g[4], op=home?g[4]:g[3];
      L.push({k, pid, gid:x[0], date:g[0], tm:home?g[1]:g[2], opp:(home?'vs ':'@ ')+(home?g[2]:g[1]), w:mp>op, score:`${mp}-${op}`, x, ix,
        o:x[ix.gso], d:x[ix.gsd], t:Math.round((x[ix.gso]+x[ix.gsd])*10)/10}); }); });
  return (GTOP=L); }
const gsLine=e=>{ const v=c=>e.x[e.ix[c]]; return `${v('pts')} pts · ${v('oreb')+v('dreb')} reb · ${v('ast')} ast · ${Math.round(v('min'))} min`; };
function openGameScore(k,gid){ if(!byKey.has(k)) return; openPlayer(k); pvTab='season'; pvSea.split='all'; pvSea.open=String(gid); pvSea.all=true; renderPV();
  const r=$('pv').querySelector(`.glr[data-v="${gid}"]`); if(r) r.scrollIntoView({block:'center'}); }
function renderTopGames(){
  const L=gameScores();
  if(!L){ seasLoad(); $('count').textContent=''; $('grows').innerHTML=`<tr><td colspan="5" class="l note">${SEAS.st==='err'?"Couldn't load this season's games. Check your connection.":'Loading this season’s games…'}</td></tr>`; $('more').hidden=true; return; }
  const q=$('q').value.trim().toLowerCase(), team=tf.value, pf=$('posf').value;
  const posOk=r=>!pf||('GWB'.includes(pf)?POS3(r.p.ps)===pf:POS5(r.p.ps)===pf);
  const list=L.filter(e=>{ const r=byKey.get(e.k); return r&&(!team||e.tm===team)&&posOk({p:r})&&(!q||r.n.toLowerCase().includes(q)); })
    .sort((a,b)=>gAsc?a[gSort]-b[gSort]:b[gSort]-a[gSort]);
  $('count').textContent=`${list.length.toLocaleString()} games`;
  $('grows').innerHTML=list.slice(0,shown).map((e,i)=>{ const p=byKey.get(e.k);
    return `<tr class="p" tabindex="0" data-k="${esc(e.k)}" data-g="${e.gid}"><td class="l rk">${i+1}</td>
      <td class="l"><span class="nm pvlink">${esc(p.n)}</span><span class="tm">${e.tm} ${e.opp} · ${seaDate(e.date).replace(/^\\w+, /,'')} · <span style="white-space:nowrap">${e.w?'W':'L'} ${e.score}</span></span><span class="line">${gsLine(e)}</span></td>
      <td class="num ${cls(e.o)}">${fmt(e.o)}</td><td class="num ${cls(e.d)}">${fmt(e.d)}</td><td><span class="pa ${cls(e.t)}">${fmt(e.t)}</span></td></tr>`; }).join('');
  $('more').hidden=list.length<=shown;
}
function bestNightHTML(date,title){ // top 5 game scores on one night (Tonight tab)
  const L=gameScores(); if(!L){ seasLoad(); return ''; }
  const d=date||L.reduce((m,e)=>e.date>m?e.date:m,''); if(!d) return '';
  const top=L.filter(e=>e.date===d).sort((a,b)=>b.t-a.t).slice(0,5); if(!top.length) return '';
  const ttl=title||(()=>{ const y=new Date(new Date(SZ.asof+'T12:00:00')-864e5).toISOString().slice(0,10); return d===y?"Last night's best":`Best games · ${seaDate(d)}`; })();
  return `<div class="tnbest"><div class="sea-hd"><h3>${ttl}</h3><span class="note">box game score, points</span></div><ol>${top.map((e,i)=>{ const p=byKey.get(e.k);
    return `<li><button data-bgk="${esc(e.k)}" data-bgg="${e.gid}"><span class="rk">${i+1}</span><span class="who"><b>${esc(p.n)}</b><small>${e.tm} ${e.opp} · ${gsLine(e)}</small></span>
      <span class="sc"><b class="${cls(e.t)}">${fmt(e.t)}</b><small>O ${fmt(e.o)} · D ${fmt(e.d)}</small></span></button></li>`; }).join('')}</ol></div>`; }
function renderPlayers(){
  if(!P) return;
  if(SEASON&&plView==='games'){ renderTopGames(); return; }
  if(SEASON&&plView==='season'){ renderSeasonPlayers(); return; }""")
rep("""    const s_=plView==='season'; $('sTbl').hidden=!s_; $('sNote').hidden=!s_; document.querySelector('#p-pl .tbl:not(#sTbl)').hidden=s_;
    $('minf').hidden=s_; renderPlayers(); });""",
    """    const s_=plView==='season', g_=plView==='games'; $('sTbl').hidden=!s_; $('sNote').hidden=!s_; $('gTbl').hidden=!g_; $('gNote').hidden=!g_;
    document.querySelector('#p-pl .tbl:not(#sTbl):not(#gTbl)').hidden=s_||g_; $('minf').hidden=s_||g_; renderPlayers(); });
  document.querySelectorAll('th button[data-gsort]').forEach(b=>b.addEventListener('click',()=>{ const k=b.dataset.gsort; gAsc=k===gSort?!gAsc:false; gSort=k; shown=50;
    document.querySelectorAll('th button[data-gsort]').forEach(x=>x.removeAttribute('aria-sort')); b.setAttribute('aria-sort',gAsc?'ascending':'descending'); renderPlayers(); }));
  $('grows').addEventListener('click',e=>{ const tr=e.target.closest('tr.p'); if(tr) openGameScore(tr.dataset.k,tr.dataset.g); });
  $('grows').addEventListener('keydown',e=>{ const tr=e.target.closest('tr.p'); if(tr&&e.target===tr&&(e.key==='Enter'||e.key===' ')){ e.preventDefault(); tr.click(); } });""")

# ---------- lazy load: repaint the views that use the game log
rep(""".then(([g,rh,pv])=>{ SEAS.g=g; SEAS.rh=rh; SEAS.pv=pv; SEAS.st='ok'; seasPaint(); })
    .catch(()=>{ SEAS.st='err'; seasPaint(); }); }""",
    """.then(([g,rh,pv])=>{ SEAS.g=g; SEAS.rh=rh; SEAS.pv=pv; SEAS.st='ok'; GTOP=null; seasPaint(); seasAfter(); })
    .catch(()=>{ SEAS.st='err'; seasPaint(); seasAfter(); }); }
function seasAfter(){ try{ if(plView==='games') renderPlayers(); renderTonight(); }catch(e){} }""")

# ---------- Tonight tab
rep("""    h+='<div class="games">'+rows.map(pastCard).join('')+'</div>';""",
    """    h+='<div class="games">'+rows.map(pastCard).join('')+'</div>';
    h+=bestNightHTML(D0.d,'Best games that night');""")
rep("""    if(!gs.length) h+='<p class="note">No games scheduled.</p>';""",
    """    if(D0.k==='now') h+=bestNightHTML();
    if(!gs.length) h+='<p class="note">No games scheduled.</p>';""")
rep("""$('tonight')&&$('tonight').addEventListener('click',e=>{ const n=e.target.closest('[data-tnd]');""",
    """$('tonight')&&$('tonight').addEventListener('click',e=>{ const bg=e.target.closest('[data-bgk]'); if(bg){ openGameScore(bg.dataset.bgk,bg.dataset.bgg); return; } const n=e.target.closest('[data-tnd]');""")

# ---------- player page game log
rep("""    const op=s.open===id, v=c=>x[ix[c]], pm=v('pm'), t=v('fga')+0.44*v('fta');""",
    """    const op=s.open===id, v=c=>x[ix[c]], pm=v('pm'), t=v('fga')+0.44*v('fta'), go=ix.gso!=null?v('gso'):null, gd=ix.gsd!=null?v('gsd'):null, gt=go==null?null:Math.round((go+gd)*10)/10;""")
rep("""<span class="${cls(pm)}">${pm>0?'+':pm<0?'−':''}${Math.abs(pm)}</span></button>""",
    """<span class="${cls(pm)}">${pm>0?'+':pm<0?'−':''}${Math.abs(pm)}</span><span class="gsc ${gt==null?'muted':cls(gt)}">${gt==null?'–':fmt(gt)}</span></button>""")
rep("""['Off reb',v('oreb')],['Def reb',v('dreb')],['Stl / Blk',`${v('stl')} / ${v('blk')}`],['Turnovers',v('tov')]].map(""",
    """['Off reb',v('oreb')],['Def reb',v('dreb')],['Stl / Blk',`${v('stl')} / ${v('blk')}`],['Turnovers',v('tov')],
        ...(gt==null?[]:[['Game score',fmt(gt)],['GS offense',fmt(go)],['GS defense',fmt(gd)]])].map(""")
rep("""<span>Game</span><span>Result</span><span>MIN</span><span>PTS</span><span>REB</span><span>AST</span><span>+/−</span></div>${lg}""",
    """<span>Game</span><span>Result</span><span>MIN</span><span>PTS</span><span>REB</span><span>AST</span><span>+/−</span><span title="Box game score: points added vs an average player in the same minutes">GS</span></div>${lg}""")
rep("""Box lines built from play-by-play; points checked against the official box score.</p>`;""",
    """Box lines built from play-by-play; points checked against the official box score. GS = box game score: points that game's box score added to the margin vs an average player in the same minutes (same weights as the ratings; his games add up to his season box rating). Players tab › Top games ranks every game.</p>`;""")
# footnote
rep("""<span style="opacity:.6">Build 11-22.</span>""", """<span style="opacity:.6">Build 11-23.</span>""")

for fn in FILES:
    s = open(fn).read()
    assert "function gameScores()" not in s, f"{fn}: already patched"
    for a, b in E:
        n = s.count(a); assert n == 1, f"{fn}: anchor found {n}x: {a[:90]!r}"
        s = s.replace(a, b)
    open(fn, "w").write(s); print(f"patched {fn}: {len(E)} edits")
