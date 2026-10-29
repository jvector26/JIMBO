// Checks the page engine reproduces project.py (team wins) and simulate.py (blend sim) at default settings.
const fs=require('fs');
const html=fs.readFileSync('dashboard.html','utf8');
const data=JSON.parse(html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const eng=html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
global.atob=s=>Buffer.from(s,'base64').toString('binary');
require('vm').runInThisContext(eng+';globalThis.makeEngine=makeEngine;');
const E=makeEngine(data);
const DEF={}; data.meta.components.forEach(c=>DEF[c]=1); DEF.lineup_o=0.6; DEF.lineup_d=0.8; 
let t0=Date.now(); const P=E.project(DEF,{}); console.log('project ms',Date.now()-t0);
const py=Object.fromEntries(fs.readFileSync('data/teams_proj_2026.csv','utf8').trim().split('\n').slice(1).map(l=>{const a=l.split(',');return [a[0],+a[5]];}));
let md=0; P.team.forEach(x=>md=Math.max(md,Math.abs(x.mw-py[x.t]))); console.log('max |model wins - project.py|',md.toFixed(4));
t0=Date.now(); const S=E.simulate(P.team.map(x=>x.bmargin),data.meta.blend.tau,10000); console.log('sim ms',Date.now()-t0);
let dw=0,dp=0,dt=0; P.team.forEach((x,i)=>{const q=data.meta.py_sim[x.t]; dw=Math.max(dw,Math.abs(S[i].mean-q.w)); dp=Math.max(dp,Math.abs(S[i].po-q.po)); dt=Math.max(dt,Math.abs(S[i].title-q.title));});
console.log('max diff vs simulate.py blend: wins',dw.toFixed(2),'playoffs',dp.toFixed(3),'title',dt.toFixed(3));
console.log('sum wins',S.reduce((a,s)=>a+s.mean,0).toFixed(1));
P.team.slice().sort((a,b)=>b.bw-a.bw).slice(0,5).forEach(x=>console.log(x.t,x.bw.toFixed(1),S[x.i].mean.toFixed(1),S[x.i].po.toFixed(3),S[x.i].title.toFixed(3),data.meta.py_sim[x.t]));
const bt=E.backtest(DEF); console.log('backtest',bt);
// override test: Haliburton 70 g x 33 min; Herro rating +2; Duren -> FA
const k=n=>data.players.find(p=>p.n===n).k;
const OV={}; OV[k('Tyrese Haliburton')]={g:70,mpg:33}; OV[k('Tyler Herro')]={r:2}; OV[k('Jalen Duren')]={t:'FA'};
const P2=E.project(DEF,OV); const T2=Object.fromEntries(P2.team.map(x=>[x.t,x])), T1=Object.fromEntries(P.team.map(x=>[x.t,x]));
['IND','MIL','DET'].forEach(t=>{ const mins=T2[t].roster.reduce((a,r)=>a+r.min,0); console.log(t,'model wins',T1[t].mw.toFixed(1),'->',T2[t].mw.toFixed(1),'forecast',T1[t].bw.toFixed(1),'->',T2[t].bw.toFixed(1),'team min',mins.toFixed(0)); });
console.log('sum model wins after overrides',P2.team.reduce((a,x)=>a+x.mw,0).toFixed(1));
