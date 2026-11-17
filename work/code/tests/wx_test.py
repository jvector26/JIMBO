"""Box wins vs proj column (session 55): Players > This season. Usage: python3 code/tests/wx_test.py SITE_DIR"""
import asyncio, subprocess, sys, time
from playwright.async_api import async_playwright
SITE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/site1"
srv = subprocess.Popen(["python3", "-m", "http.server", "8811", "-d", SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
ROWS = """[...document.querySelectorAll('#srows tr.p')].slice(0,N).map(r=>[...r.children].map(c=>c.textContent.trim().replace(/\\s+/g,' ')))"""
async def run(dev, label):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**(p.devices["iPhone 13"] if dev else {"viewport": {"width": 1400, "height": 1000}}))
        await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort()); await ctx.route("**/fonts.g*/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8811/index.html"); await pg.wait_for_timeout(5000)
        await pg.evaluate("document.getElementById('t-pl').click()"); await pg.wait_for_timeout(400); await pg.click('[data-plv="season"]'); await pg.wait_for_timeout(500)
        print(f"--- {label} default sort (pa):", await pg.evaluate(ROWS.replace("N", "3")))
        await pg.click('[data-ssort="wx"]'); await pg.wait_for_timeout(400)
        print(" wx sort top:", await pg.evaluate(ROWS.replace("N", "6")), await pg.text_content("#count"))
        ov = await pg.evaluate("[document.documentElement.scrollWidth, innerWidth, document.querySelector('#sTbl').scrollWidth, document.querySelector('#sTbl').clientWidth]")
        print(" page/table widths:", ov)
        await pg.screenshot(path=f"/tmp/wx_{label}.png")
        await pg.select_option("#tf", "SAC") if await pg.locator("#tf").count() else None
        await pg.click('[data-ssort="pa"]'); await pg.wait_for_timeout(300)
        rk = await pg.evaluate("""[...document.querySelectorAll('#srows tr.p')].map(r=>[r.children[1].querySelector('.nm').textContent, r.lastElementChild.textContent, r.lastElementChild.className])""")
        print(" team view, blanks/greys:", [x for x in rk if x[1] == '–' or 'muted' in x[2]][:6], len(rk))
        await pg.locator('#srows tr.p').first.click(); await pg.wait_for_timeout(1500)
        print(" player summary:", (await pg.text_content(".sea-sum")) if await pg.locator(".sea-sum").count() else None)
        await pg.screenshot(path=f"/tmp/wx_{label}_pv.png")
        print(" errors:", errs); await b.close()
async def main():
    await run(True, "phone"); await run(False, "desktop")
asyncio.run(main()); srv.terminate()
