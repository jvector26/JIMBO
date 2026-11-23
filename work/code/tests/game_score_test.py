"""Box game score (session 58): game log GS column, Players > Top games, Tonight "Last night's best".
Usage: python3 code/tests/game_score_test.py SITE_DIR"""
import asyncio, subprocess, sys, time
from playwright.async_api import async_playwright
SITE = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/gh/site"
srv = subprocess.Popen(["python3", "-m", "http.server", "8812", "-d", SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
ROWS = """[...document.querySelectorAll('#grows tr.p')].slice(0,N).map(r=>[...r.children].map(c=>c.textContent.trim().replace(/\\s+/g,' ')))"""
WID = "[document.documentElement.scrollWidth, innerWidth]"
async def run(dev, label):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**(p.devices["iPhone 13"] if dev else {"viewport": {"width": 1400, "height": 1000}}))
        await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort()); await ctx.route("**/fonts.g*/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8812/index.html"); await pg.wait_for_timeout(5000)
        # Tonight
        await pg.evaluate("document.getElementById('t-tn').click()"); await pg.wait_for_timeout(1500)
        print(f"--- {label} tonight best:", (await pg.text_content(".tnbest")) if await pg.locator(".tnbest").count() else None)
        print(" widths:", await pg.evaluate(WID)); await pg.screenshot(path=f"/tmp/gs_{label}_tn.png")
        # previous night
        await pg.click('[data-tnd]:not([disabled])'); await pg.wait_for_timeout(500)
        print(" past day best:", (await pg.text_content(".tnbest h3")) if await pg.locator(".tnbest").count() else None)
        await pg.screenshot(path=f"/tmp/gs_{label}_tnpast.png", full_page=True)
        # Players > Top games
        await pg.evaluate("document.getElementById('t-pl').click()"); await pg.wait_for_timeout(400); await pg.click('[data-plv="games"]'); await pg.wait_for_timeout(800)
        print(" top games:", await pg.evaluate(ROWS.replace("N", "5")), await pg.text_content("#count"))
        print(" widths:", await pg.evaluate(WID + ".concat([document.querySelector('#gTbl').scrollWidth, document.querySelector('#gTbl').clientWidth])"))
        await pg.screenshot(path=f"/tmp/gs_{label}_top.png")
        await pg.click('[data-gsort="d"]'); await pg.wait_for_timeout(300)
        print(" def sort:", await pg.evaluate(ROWS.replace("N", "3")))
        await pg.click('[data-gsort="t"]'); await pg.wait_for_timeout(200); await pg.click('[data-gsort="t"]'); await pg.wait_for_timeout(300)
        print(" worst games:", await pg.evaluate(ROWS.replace("N", "3")))
        await pg.select_option("#tf", "MIL"); await pg.wait_for_timeout(300)
        print(" MIL worst:", await pg.evaluate(ROWS.replace("N", "2")), await pg.text_content("#count"))
        await pg.select_option("#tf", ""); await pg.click('[data-gsort="t"]'); await pg.wait_for_timeout(300)
        # back to the other views still work
        await pg.click('[data-plv="season"]'); await pg.wait_for_timeout(300)
        vis = await pg.evaluate("[document.getElementById('sTbl').hidden, document.getElementById('gTbl').hidden, document.querySelector('#p-pl .tbl:not(#sTbl):not(#gTbl)').hidden]")
        print(" season view hidden flags (sTbl,gTbl,proj):", vis)
        await pg.click('[data-plv="games"]'); await pg.wait_for_timeout(300)
        await pg.locator('#grows tr.p').first.click(); await pg.wait_for_timeout(1500)
        print(" game log head:", await pg.text_content(".glh"))
        op = await pg.evaluate("(()=>{const b=document.querySelector('.glr[aria-expanded=\"true\"]'); return b?[b.textContent.replace(/\\s+/g,' '), b.nextElementSibling&&b.nextElementSibling.textContent.replace(/\\s+/g,' ')]:null})()")
        print(" opened row:", op)
        print(" widths:", await pg.evaluate(WID), " glr fits:", await pg.evaluate("[...document.querySelectorAll('.glr')].every(b=>b.scrollWidth<=b.clientWidth+1)"))
        await pg.screenshot(path=f"/tmp/gs_{label}_pv.png")
        print(" errors:", errs); await b.close()
async def main():
    await run(True, "phone"); await run(False, "desktop")
asyncio.run(main()); srv.terminate()
