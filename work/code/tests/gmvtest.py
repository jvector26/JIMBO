import asyncio, subprocess, time
from playwright.async_api import async_playwright
srv = subprocess.Popen(["python3","-m","http.server","8803","-d","/tmp/site_c"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(1)
CHK = """(()=>{ const P=window.__jimbo.P; let mx=0, bad=0, worst=null; P.rows.filter(r=>r.t!=='FA').forEach(r=>{ const e=document; });
  return null; })()"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**p.devices["iPhone 13"]); await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs=[]; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8803/index.html"); await pg.wait_for_timeout(4000)
        async def row(k):
            await pg.locator('[data-go="t-pl"]').tap(); await pg.wait_for_timeout(300)
            await pg.fill("#q", k); await pg.wait_for_timeout(500)
            cells = await pg.locator("#rows tr").first.locator("td").all_text_contents(); return cells
        async def edit(name, key, role=None, mpg=None):
            await row(name); await pg.locator(f'#rows [data-edit="{key}"]').first.tap(); await pg.wait_for_timeout(400)
            before=(await pg.input_value("#edG"), await pg.input_value("#edM"))
            if role is not None: await pg.select_option("#edD", role); await pg.wait_for_timeout(400)
            if mpg is not None: await pg.fill("#edM", str(mpg))
            after=(await pg.input_value("#edG"), await pg.input_value("#edM"))
            await pg.locator("#edSave").tap(); await pg.wait_for_timeout(1500); return before, after
        for nm,k,role in [("Jakucionis","p1642857","S"),("Porzingis","p204001","L"),("Jakucionis","p1642857","N")]:
            print(nm, role, "editor (g,mpg) before/after:", await edit(nm,k,role=role))
            r = await pg.evaluate(f"(()=>{{const r=window.__jimbo.P.rows.find(x=>x.p.k==='{k}'); return [Math.round(r.min), r.z&&r.z.av, r.z&&r.z.miss]}})()")
            print("   min, av, miss:", r, " row:", [c.strip() for c in (await row(nm))][:8])
        # mpg lock round trip
        print("Curry mpg 30:", await edit("Curry","p201939",mpg=30), await pg.evaluate("window.__jimbo.OV['p201939']"),
              await pg.evaluate("Math.round(window.__jimbo.P.rows.find(x=>x.p.k==='p201939').min)"))
        # sanity over everyone: g <= 82 and g x mpg ~ minutes (via page gmv not exposed: use roster table of all teams)
        await pg.locator('[data-go="t-st"]').tap(); await pg.wait_for_timeout(400)
        print("errors", errs); await b.close()
asyncio.run(main()); srv.terminate()
