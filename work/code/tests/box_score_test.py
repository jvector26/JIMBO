"""Box score page (session 59): Tonight > past day > tap a final.
Usage: python3 code/tests/box_score_test.py SITE_DIR"""
import asyncio, subprocess, sys, time
from playwright.async_api import async_playwright
SITE = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/jimbo/site"
srv = subprocess.Popen(["python3", "-m", "http.server", "8813", "-d", SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
WID = "[document.documentElement.scrollWidth, innerWidth]"
ROWS = """([i,n])=>[...document.querySelectorAll('.bxt')[i].querySelectorAll('tbody tr')].slice(0,n).map(r=>[...r.children].map(c=>c.textContent.trim()).join(' | '))"""
async def run(dev, label):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**(p.devices["iPhone 13"] if dev else {"viewport": {"width": 1400, "height": 1000}}))
        await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort()); await ctx.route("**/fonts.g*/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8813/index.html"); await pg.wait_for_timeout(5000)
        await pg.evaluate("document.getElementById('t-tn').click()"); await pg.wait_for_timeout(1200)
        await pg.click('[data-tnd]:not([disabled])'); await pg.wait_for_timeout(600)
        print(f"--- {label} past cards:", await pg.locator('.game[data-bx]').count())
        await pg.locator('.game[data-bx]').first.click(); await pg.wait_for_timeout(800)
        print(" head:", (await pg.text_content('#gv .gvh')).split(), (await pg.text_content('#gv .bxsum')))
        print(" away top:", await pg.evaluate(ROWS, [0, 4])); print(" home top:", await pg.evaluate(ROWS, [1, 3]))
        print(" foot:", await pg.evaluate("[...document.querySelectorAll('.bxt tfoot')].map(t=>t.textContent.replace(/\\s+/g,' '))"))
        print(" pct/dnp:", await pg.evaluate("[...document.querySelectorAll('.bxpct,.bxdnp')].map(t=>t.textContent)"))
        print(" widths:", await pg.evaluate(WID), await pg.evaluate("[...document.querySelectorAll('.bxwrap')].map(w=>[w.scrollWidth,w.clientWidth])"))
        await pg.screenshot(path=f"/tmp/bx_{label}.png", full_page=True)
        await pg.click('.bxt [data-bxs="pts"] >> nth=0'); await pg.wait_for_timeout(300)
        print(" pts sort:", await pg.evaluate(ROWS, [0, 3]), await pg.evaluate(ROWS, [1, 2]))
        await pg.evaluate("document.querySelector('.bxwrap').scrollLeft=200; document.querySelector('.bxt [data-bxs=pts]').click()"); await pg.wait_for_timeout(300)
        print(" pts asc:", await pg.evaluate(ROWS, [0, 2]), "scrollLeft kept:", await pg.evaluate("document.querySelector('.bxwrap').scrollLeft"))
        await pg.click('.bxt [data-bxs="gs"] >> nth=1', force=True); await pg.wait_for_timeout(300)
        print(" gs sort home:", await pg.evaluate(ROWS, [1, 2]))
        await pg.screenshot(path=f"/tmp/bx_{label}_gs.png")
        await pg.locator('.bxt td.bxn button').first.click(); await pg.wait_for_timeout(1200)
        op = await pg.evaluate("(()=>{const b=document.querySelector('.glr[aria-expanded=\"true\"]'); return b?b.textContent.replace(/\\s+/g,' '):null})()")
        print(" player opened row:", op)
        await pg.go_back(); await pg.wait_for_timeout(600)
        print(" back to box?", await pg.evaluate("!document.getElementById('gv').hidden && !!document.querySelector('.bxt')"))
        await pg.go_back(); await pg.wait_for_timeout(600)
        print(" back to tonight?", await pg.evaluate("document.getElementById('gv').hidden"), await pg.text_content('.tnlab h2'))
        # a regular (upcoming) game page still works
        print(" errors:", errs); await b.close()
async def main():
    await run(True, "phone"); await run(False, "desktop")
asyncio.run(main()); srv.terminate()
