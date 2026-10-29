import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**p.devices["iPhone 13"])
        pg = await ctx.new_page(); logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}")); pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
        await pg.goto("file:///home/claude/dashboard.html"); await pg.wait_for_timeout(2500)
        btns = pg.locator("[data-edit]"); n = await btns.count(); print("edit buttons visible in DOM:", n)
        # open first visible team rotation if needed
        await pg.locator("#t-pl").tap(); await pg.wait_for_timeout(800)
        vis = [i for i in range(await btns.count()) if await btns.nth(i).is_visible()]
        print("visible edit buttons:", len(vis))
        k = await btns.nth(vis[0]).get_attribute("data-edit"); await btns.nth(vis[0]).tap(); await pg.wait_for_timeout(300)
        print("editing", k, await pg.locator("#edName").text_content(), "| open:", await pg.locator("#ed").evaluate("d=>d.open"))
        await pg.locator("#edG").fill("0"); await pg.locator("#edSave").tap(); await pg.wait_for_timeout(1500)
        print("dialog open after save:", await pg.locator("#ed").evaluate("d=>d.open"))
        print("OV:", await pg.evaluate("localStorage.getItem('nba27-dash-state')")[:0] if False else "")
        await pg.evaluate("document.getElementById('settings').open=true")
        print("changes header:", await pg.locator("#chgH").text_content(), "| list:", await pg.locator("#chgList").inner_text())
        print("sync:", await pg.locator("#sync").text_content())
        print("\n".join(logs[-10:]))
        await b.close()
asyncio.run(main())
