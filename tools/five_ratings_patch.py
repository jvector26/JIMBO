"""Session 57 (2026-11-22, user request, display only): FIVE RATINGS wherever a player is summarised:
Overall, Box offense, Box defense, Lineup offense, Lineup defense.
Box O / Box D = the box-score part (hustle counts as box defense), including this season's box update split O/D by the
nightly (latest.json players[].u = [box O, box D, lineup net]). Lineup O / D = the preseason lineup parts; the in-season
lineup update is one net number (the in-season RAPM fits one number per player), shown in the "not in the four" line
with age and league re-centering. Rookies / newcomers / in-season signings (fx) have one baseline only: four parts '-'.
Places: player page cards; Players tab Projection table (Off/Def columns replaced, sortable; phones: line under the name);
team view roster (four parts on wide screens only). Applies the same edits to the template (work/code/dash_v2.html) and
the built page (site/index.html). Run once; refuses to reapply."""
import sys
FILES = ["work/code/dash_v2.html", "site/index.html"]
EDITS = [
 # CSS
 (".ptab .skline{display:none;font-size:12px;color:var(--muted);margin-top:1px}",
  ".ptab .skline{display:none;font-size:12px;color:var(--muted);margin-top:1px}\n"
  ".fivenote{font-size:12.5px;color:var(--muted);margin:-8px 0 16px}"),
 # rawRating: box-only offense / defense (hustle = box defense)
 ("    off+=lo+ag/2+rt/2; def+=ld+ag/2+rt/2+hu;\n    return {off, def, parts, dreb, dstop};",
  "    const bo=off, bd=def+hu;  // five ratings (2026-11-22): box offense / box defense (hustle counts as box defense)\n"
  "    off+=lo+ag/2+rt/2; def+=ld+ag/2+rt/2+hu;\n    return {off, def, parts, dreb, dstop, bo, bd};"),
 ("if(p.fx!=null) return {off:p.fx/2, def:p.fx/2, parts:{base:p.fx}, dreb:null, dstop:null};",
  "if(p.fx!=null) return {off:p.fx/2, def:p.fx/2, parts:{base:p.fx}, dreb:null, dstop:null, bo:null, bd:null};"),
 ("return {p, t:o.t||p.t, raw:r.off+r.def, off:r.off, def:r.def, dreb:r.dreb,",
  "return {p, t:o.t||p.t, raw:r.off+r.def, off:r.off, def:r.def, bo:r.bo, bd:r.bd, lo:r.bo==null?null:r.parts.lineup_o, ld:r.bo==null?null:r.parts.lineup_d, lu:null, dreb:r.dreb,"),
 # in season: this season's box update split O/D, lineup update net (nightly field u)
 ("    if(Math.abs(d)>1e-9){ r.parts.season=d; r.mtot+=d; if(r.o.r==null){ r.tot+=d; r.off+=d/2; r.def+=d/2; } }",
  "    if(Math.abs(d)>1e-9){ r.parts.season=d; r.mtot+=d; if(r.o.r==null){ r.tot+=d; r.off+=d/2; r.def+=d/2; } }\n"
  "    if(z&&z.u&&r.bo!=null){ r.bo+=z.u[0]; r.bd+=z.u[1]; r.lu=z.u[2]; }  // five ratings: this season's box update O/D; lineup update stays net"),
 # helpers (next to roleTag)
 ("function roleTag(r){",
  "// five ratings (2026-11-22): what the overall holds beyond box O/D + lineup O/D\n"
  "function fiveRest(r){ if(r.bo==null) return null; const pa=r.parts, age=(pa.aging||0)+(pa.returner||0), lu=r.lu||0;\n"
  "  const rest=r.tot-(r.bo+r.bd+r.lo+r.ld), other=rest-age-lu, out=[];\n"
  "  if(Math.abs(age)>=0.005) out.push(`age ${fmt(age)}`);\n"
  "  if(r.lu!=null) out.push(`lineup results this season ${fmt(lu)} (one net number, not split into offense and defense)`);\n"
  "  if(Math.abs(other)>=0.05) out.push(r.o.r!=null?`your rating edit ${fmt(other)}`:(r.lu==null&&pa.season!=null?`this season's games ${fmt(other)} (not split yet)`:`league re-centering ${fmt(other)}`));\n"
  "  return out; }\n"
  "function fiveLine(r){ return r.bo==null?'':`Box ${fmt(r.bo)} / ${fmt(r.bd)} · Lineup ${fmt(r.lo)} / ${fmt(r.ld)}`; }\n"
  "function fiveTd(v, extra){ return v==null?`<td class=\"num muted ${extra||''}\">–</td>`:`<td class=\"num ${cls(v)} ${extra||''}\">${fmt(v)}</td>`; }\n"
  "function roleTag(r){"),
 # Players table header: Off / Def -> four parts; Rating -> Overall
 ('<th class="hide-s"><button data-sort="off" title="Offense, points per 100 vs an average player">Off</button></th><th class="hide-s"><button data-sort="def" title="Defense, points per 100 vs an average player">Def</button></th>',
  '<th class="hide-s"><button data-sort="bo" title="Box score, offense: points per 100 vs average (this season\'s box score included)">Box O</button></th><th class="hide-s"><button data-sort="bd" title="Box score, defense (hustle stats included): points per 100 vs average">Box D</button></th>'
  '<th class="hide-s"><button data-sort="lo" title="Lineup results beyond the box score, offense (preseason)">Lin O</button></th><th class="hide-s"><button data-sort="ld" title="Lineup results beyond the box score, defense (preseason)">Lin D</button></th>'),
 ('<th><button data-sort="tot" aria-sort="descending" title="2026-27 projected rating (offense + defense)">Rating</button></th>',
  '<th><button data-sort="tot" aria-sort="descending" title="Overall 2026-27 rating: box + lineup, plus age, this season\'s lineup results and league re-centering">Overall</button></th>'),
 ('<table class="ptab" style="min-width:720px">', '<table class="ptab" style="min-width:800px">'),
 # Players table rows
 ("${r.tw[0]?`<span class=\"skline\">Off ${fmt(r.off)} · Def ${fmt(r.def)} · best ${SKS[r.tw[0][0]]}, worst ${SKS[r.tw[1][0]]}</span>`:`<span class=\"skline\">Off ${fmt(r.off)} · Def ${fmt(r.def)}</span>`}",
  "${r.bo!=null?`<span class=\"skline\">${fiveLine(r)}${r.tw[0]?` · best ${SKS[r.tw[0][0]]}, worst ${SKS[r.tw[1][0]]}`:''}</span>`:''}"),
 ("      <td class=\"num hide-s ${cls(r.off)}\">${fmt(r.off)}</td><td class=\"num hide-s ${cls(r.def)}\">${fmt(r.def)}</td>",
  "      ${fiveTd(r.bo,'hide-s')}${fiveTd(r.bd,'hide-s')}${fiveTd(r.lo,'hide-s')}${fiveTd(r.ld,'hide-s')}"),
 # player page cards: Offense / Defense -> four parts + the 'not in the four' line
 ("<div class=\"card\"><div class=\"k\">2026-27 rating</div>",
  "<div class=\"card\"><div class=\"k\">Overall</div>"),
 ("    <div class=\"card\"><div class=\"k\">Offense</div><div class=\"v ${cls(r.off)}\">${fmt(r.off)}</div><div class=\"s\">pts/100 vs average</div></div>\n"
  "    <div class=\"card\"><div class=\"k\">Defense</div><div class=\"v ${cls(r.def)}\">${fmt(r.def)}</div><div class=\"s\">pts/100 vs average</div></div>\n",
  "    ${[['Box offense',r.bo,r.lu!=null?'incl. this season':'box score'],['Box defense',r.bd,r.lu!=null?'incl. hustle, this season':'incl. hustle'],['Lineup offense',r.lo,'preseason'],['Lineup defense',r.ld,'preseason']].map(([k_,v_,s_])=>\n"
  "      `<div class=\"card\"><div class=\"k\">${k_}</div><div class=\"v ${v_==null?'muted':cls(v_)}\">${v_==null?'–':fmt(v_)}</div><div class=\"s\">${v_==null?(D.rook[k]?'rookie: one baseline':'one baseline value'):s_}</div></div>`).join('')}\n"),
 ("    ${r.t!=='FA'?`<div class=\"card\"><div class=\"k\">Adds to team</div><div class=\"v ${cls(gv.add)}\">${fmt(gv.add,2)}</div><div class=\"s\">net rating points</div></div>`:''}\n  </div>`;",
  "    ${r.t!=='FA'?`<div class=\"card\"><div class=\"k\">Adds to team</div><div class=\"v ${cls(gv.add)}\">${fmt(gv.add,2)}</div><div class=\"s\">net rating points</div></div>`:''}\n  </div>"
  "${(()=>{ const fr=fiveRest(r); return fr==null?`<p class=\"fivenote\">${D.rook[k]?'Rookie':'Newcomer'}: the overall is one baseline value, so there is no box / lineup split yet.</p>`:fr.length?`<p class=\"fivenote\">Overall also includes: ${fr.join(' · ')}.</p>`:''; })()}`;"),
 # team roster: four parts on wide screens
 ("    <td class=\"num ${r.o.r!=null?'ov':cls(r.tot)}\">${fmt(r.tot)}</td><td class=\"num ${z0?'muted':cls(gmv(r).add)}\">${fmt(gmv(r).add,2)}</td>",
  "    <td class=\"num ${r.o.r!=null?'ov':cls(r.tot)}\">${fmt(r.tot)}</td>${fiveTd(r.bo,'hide-m')}${fiveTd(r.bd,'hide-m')}${fiveTd(r.lo,'hide-m')}${fiveTd(r.ld,'hide-m')}<td class=\"num ${z0?'muted':cls(gmv(r).add)}\">${fmt(gmv(r).add,2)}</td>"),
 ("<th>Rating</th><th title=\"Rating times share of team minutes\">Adds to team</th><th></th></tr></thead><tbody>${rows}</tbody></table>",
  "<th title=\"Overall rating\">Overall</th><th class=\"hide-m\" title=\"Box score, offense\">Box O</th><th class=\"hide-m\" title=\"Box score + hustle, defense\">Box D</th><th class=\"hide-m\" title=\"Lineup results beyond the box score, offense (preseason)\">Lin O</th><th class=\"hide-m\" title=\"Lineup results beyond the box score, defense (preseason)\">Lin D</th><th title=\"Rating times share of team minutes\">Adds to team</th><th></th></tr></thead><tbody>${rows}</tbody></table>"),
 ("<tr class=\"benchhd\"><td colspan=\"8\">", "<tr class=\"benchhd\"><td colspan=\"12\">"),
]
for f in FILES:
    s = open(f).read()
    if "function fiveRest(" in s: sys.exit(f"{f}: already patched")
    for a, b in EDITS:
        if s.count(a) != 1: sys.exit(f"{f}: anchor count {s.count(a)}: {a[:70]}")
        s = s.replace(a, b)
    open(f, "w").write(s); print("patched", f)
