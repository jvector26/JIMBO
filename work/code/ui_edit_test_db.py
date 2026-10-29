"""Edit/Save test with a mock of the hosted db capability (frozen snapshots, like runtime 0.2.52)."""
import asyncio, sys
from playwright.async_api import async_playwright
MOCK = """
(()=>{ const deepFreeze=o=>{ if(o&&typeof o==='object'){ Object.values(o).forEach(deepFreeze); Object.freeze(o);} return o; };
 let store={'settings/current':{players:{},sliders:{lineup:0.6,def_rebounding:1},updated:'x'}}; window.__writes=0;
 const db={ doc:(p)=>({ get: async()=>{ const d=store[p]; return deepFreeze({exists:!!d, data:()=>d?deepFreeze(JSON.parse(JSON.stringify(d))):undefined}); },
                        set: async(v)=>{ store[p]=JSON.parse(JSON.stringify(v)); window.__writes++; } }) };
 window.claude={ use: (n)=> new Promise(r=>setTimeout(()=>r(n==='db'?Object.freeze(db):null),200)) };
 window.__store=store; })();
"""
async def main(path):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**p.devices["iPhone 13"]); pg = await ctx.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.add_init_script(MOCK); await pg.goto("file://" + path); await pg.wait_for_timeout(2500)
        print("sync:", await pg.locator("#sync").text_content())
        await pg.locator("#t-pl").tap(); await pg.wait_for_timeout(600)
        btn = pg.locator("#rows [data-edit]").first; await btn.tap(); await pg.wait_for_timeout(300)
        print("editing", await pg.locator("#edName").text_content())
        await pg.locator("#edG").fill("0"); await pg.locator("#edSave").tap(); await pg.wait_for_timeout(1500)
        print("dialog still open:", await pg.locator("#ed").evaluate("d=>d.open"))
        await pg.evaluate("document.getElementById('settings').open=true")
        print("changes:", (await pg.locator("#chgH").text_content()), "|", (await pg.locator("#chgList").inner_text()).replace("\n", " "))
        print("db writes:", await pg.evaluate("window.__writes"), "| stored players:", await pg.evaluate("JSON.stringify(window.__store['settings/current'].players)"))
        # second edit + remove, to check delete path
        await pg.locator("#chgList [data-del]").first.tap(); await pg.wait_for_timeout(1200)
        print("after remove:", await pg.locator("#chgH").text_content(), "| stored:", await pg.evaluate("JSON.stringify(window.__store['settings/current'].players)"))
        print("page errors:", errs[:3]); await b.close()
asyncio.run(main(sys.argv[1]))
