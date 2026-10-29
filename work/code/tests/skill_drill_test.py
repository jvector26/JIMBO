"""Skill drill-down UI test (2026-10-18): open players, tap skills, check per-stat projection rows sum to the bar value,
no page errors; screenshots on phone + desktop. usage: python3 code/tests/skill_drill_test.py /home/claude/dashboard.html OUTDIR"""
import asyncio, sys, os
from playwright.async_api import async_playwright
MOCK = "window.claude={use:(n)=>new Promise(r=>setTimeout(()=>r(n==='db'?{doc:()=>({get:async()=>({exists:false,data:()=>undefined}),set:async()=>{}})}:null),100))};"
CHECK = """(c)=>{ const li=document.querySelector(`.skr[data-sk="${c}"]`); const bar=li.querySelector('.skv').textContent;
  const d=document.querySelector(`.skd[data-skd="${c}"]`); if(!d) return {c, open:false};
  const rows=[...d.querySelectorAll('tbody tr:not(.sr):not(.tot)')].map(tr=>tr.children[1].textContent);
  const tot=d.querySelector('tr.tot'); const num=s=>parseFloat(s.replace('−','-').replace('+',''));
  return {c, open:true, bar, rows, sum:rows.reduce((a,s)=>a+num(s),0).toFixed(2), tot:tot?tot.children[1].textContent:null,
          last:[...d.querySelectorAll('tbody tr:not(.sr)')].map(tr=>tr.children[2]?tr.children[2].textContent:''),
          raw:[...d.querySelectorAll('td.skst')].map(td=>td.textContent).filter(Boolean)}; }"""
async def main(path, out):
    os.makedirs(out, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for dev, ctxargs in (("phone", p.devices["iPhone 13"]), ("desktop", {"viewport": {"width": 1280, "height": 900}})):
            ctx = await b.new_context(**ctxargs); pg = await ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.add_init_script(MOCK); await pg.goto(path if path.startswith("http") else "file://" + path); await pg.wait_for_timeout(4000)
            for name in (["Jalen Brunson", "Rudy Gobert", "Victor Wembanyama"] if dev == "phone" else ["Jalen Brunson"]):
                k = await pg.evaluate(f"(async()=>{{const e=document.getElementById('data').textContent; const d=e.trim()?JSON.parse(e):await (await fetch('data/dash.json')).json(); return d;}})().then(d=>(d.players.find(x=>x.n==={name!r})||{{}}).k)")
                await (pg.locator('[data-go="t-pl"]') if await pg.locator('[data-go="t-pl"]').is_visible() else pg.locator("#t-pl")).click(); await pg.wait_for_timeout(400)
                await pg.locator("#q").fill(name); await pg.wait_for_timeout(400)
                await pg.locator(f'#rows tr[data-k="{k}"] .pvlink').first.click(); await pg.wait_for_timeout(600)
                if await pg.locator('[data-pvtab="proj"]').count(): await pg.locator('[data-pvtab="proj"]').click(); await pg.wait_for_timeout(200)
                for c in ["playmaking", "three_pt", "rim_protection", "perimeter_def", "hustle"]:
                    li = pg.locator(f'.skr[data-sk="{c}"]')
                    if await li.count() == 0: print(dev, name, c, "not tappable/absent"); continue
                    await li.click(); await pg.wait_for_timeout(120)
                    print(dev, name, await pg.evaluate(CHECK, c))
                el = pg.locator(".sk").first
                await el.scroll_into_view_if_needed(); await el.screenshot(path=f"{out}/{dev}_{name.split()[-1]}.png")
                # toggle closed + persistence across a re-render
                await pg.locator('.skr[data-sk="playmaking"]').click(); await pg.wait_for_timeout(100)
                if await pg.locator('[data-pvtab="career"]').count(): await pg.locator('[data-pvtab="career"]').click(); await pg.wait_for_timeout(200)
                await pg.locator("#pvMode").select_option(index=1); await pg.wait_for_timeout(300)  # change event -> renderPV()
                print(dev, name, "after close+rerender open:", await pg.evaluate("[...document.querySelectorAll('.skd')].map(d=>d.dataset.skd)"))
                await pg.locator("#pvBack").click(); await pg.wait_for_timeout(300)
            print(dev, "page errors:", errs[:3]); await ctx.close()
        await b.close()
asyncio.run(main(sys.argv[1], sys.argv[2]))
