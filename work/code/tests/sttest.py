import asyncio, subprocess, time
from playwright.async_api import async_playwright
srv = subprocess.Popen(["python3","-m","http.server","8804","-d","/tmp/site_c"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(1)
async def teams(pg, tbl=0):
    return await pg.evaluate(f"[...document.querySelectorAll('#standings table.st')[{tbl}].querySelectorAll('tbody tr.p')].map(r=>r.dataset.team)")
async def col(pg, lab):
    return await pg.evaluate(f"""(()=>{{ const t=document.querySelector('#standings table.st'); const ths=[...t.querySelectorAll('thead th')];
      const i=ths.findIndex(h=>h.textContent.trim().startsWith('{lab}')); const vis=getComputedStyle(ths[i]).display!=='none';
      return [vis, [...t.querySelectorAll('tbody tr.p')].slice(0,6).map(r=>r.children[i].textContent.trim())]; }})()""")
async def run(dev, page, label):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**(p.devices["iPhone 13"] if dev else {"viewport":{"width":1400,"height":1000}})); await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs=[]; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(f"http://localhost:8804/{page}"); await pg.wait_for_timeout(4500)
        print(f"--- {label}: default E", (await teams(pg))[:6], "cut rows", await pg.evaluate("document.querySelectorAll('#standings tr.cut').length"))
        await pg.select_option("#stview","all"); await pg.select_option("#stsort","off"); await pg.wait_for_timeout(300)
        print(" all teams by Off:", len(await teams(pg)), await col(pg,"Off rtg"))
        await pg.click("#stdir"); await pg.wait_for_timeout(300); print(" reversed:", (await col(pg,"Off rtg"))[1][:4])
        await pg.select_option("#stsort","agea"); await pg.wait_for_timeout(300); print(" by age adj:", (await teams(pg))[:5], (await col(pg,"Age adj"))[1][:5])
        await pg.screenshot(path=f"/tmp/st_{label}.png")
        # header tap: Playoffs, then tap again to reverse
        await pg.locator('[data-stsort="po"]').first.click(); await pg.wait_for_timeout(300); a=(await col(pg,"Playoffs"))[1][:3]
        await pg.locator('[data-stsort="po"]').first.click(); await pg.wait_for_timeout(300); z=(await col(pg,"Playoffs"))[1][:3]
        print(" header Playoffs desc/asc:", a, z)
        await pg.select_option("#stview","conf"); await pg.wait_for_timeout(300)
        # expand a team row still works
        await pg.locator('#standings tr.p').first.click(); await pg.wait_for_timeout(500); print(" expand ok:", await pg.evaluate("!!document.querySelector('#standings tr.detail')"))
        await pg.reload(); await pg.wait_for_timeout(4500); print(" after reload:", await pg.input_value("#stview"), await pg.input_value("#stsort"), await pg.text_content("#stdir")); await pg.select_option("#stview","all"); await pg.select_option("#stsort","agea"); await pg.wait_for_timeout(300)
        await pg.screenshot(path=f"/tmp/st_{label}_2.png")
        print(" errors:", errs); await b.close()
async def main():
    await run(True, "index.html", "phone_app"); await run(False, "artifact.html", "desktop_artifact")
asyncio.run(main()); srv.terminate()
