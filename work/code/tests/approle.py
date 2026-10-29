# Setup: nightly outputs -> python3 pipeline/nightly.py --season 2026 --out /tmp/n1 --no-history --sims 2000 (no overrides) and /tmp/n2 (overrides.json players p1642857 d S, p204001 d L); /tmp/site_a, /tmp/site_b = copies of gh/site with n1/n2 latest.json
import asyncio, json, subprocess, time
from playwright.async_api import async_playwright
LS = 'nba27-dash-state'
OVR = {"p1642857": {"d": "S"}, "p204001": {"d": "L"}}
srv = [subprocess.Popen(["python3", "-m", "http.server", str(port), "-d", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
       for port, d in ((8801, "/tmp/site_a"), (8802, "/tmp/site_b"))]
time.sleep(1)
async def run(port, ov, nightly, label, ui=False):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(**p.devices["iPhone 13"]); await ctx.route("**/raw.githubusercontent.com/**", lambda r: r.abort())
        await ctx.route("**/api.github.com/**", lambda r: r.abort())
        pg = await ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        if ov is not None:
            await pg.add_init_script(f"localStorage.setItem('{LS}', JSON.stringify({{sliders:null, players:{json.dumps(ov)}, updated:'2099'}}));")
        await pg.goto(f"http://localhost:{port}/index.html"); await pg.wait_for_timeout(4000)
        T = await pg.evaluate("window.__jimbo.P.team.map(x=>({t:x.t, m:x.margin, w:x.mw}))")
        N = {t["t"]: t for t in json.load(open(nightly))["teams"]}
        d = max(abs(x["m"] - N[x["t"]]["model_margin"]) for x in T)
        pick = {x["t"]: (round(x["m"], 2), N[x["t"]]["model_margin"]) for x in T if x["t"] in ("MIL", "GSW")}
        print(f"{label}: max |app margin - nightly model_margin| {d:.3f}  MIL/GSW (app, nightly) {pick}  errors {errs}")
        if ui:
            await pg.locator('[data-go="t-pl"]').tap(); await pg.wait_for_timeout(400)
            await pg.fill("#q", "Jakucionis"); await pg.wait_for_timeout(500)
            await pg.locator('#rows [data-edit="p1642857"]').first.tap(); await pg.wait_for_timeout(400)
            opts = await pg.eval_on_selector_all("#edD option", "els=>els.map(e=>e.textContent)")
            before = (await pg.input_value("#edG"), await pg.input_value("#edM"))
            await pg.select_option("#edD", "S"); await pg.wait_for_timeout(400)
            after = (await pg.input_value("#edG"), await pg.input_value("#edM"))
            await pg.locator("#edSave").tap(); await pg.wait_for_timeout(1200)
            ov = await pg.evaluate("window.__jimbo.OV"); chg = await pg.locator("#chgList").text_content()
            await pg.screenshot(path="/tmp/role_after.png")
            # reopen: select shows S; set back to chart default -> override removed
            await pg.locator('#rows [data-edit="p1642857"]').first.tap(); await pg.wait_for_timeout(400)
            sel = await pg.input_value("#edD"); await pg.screenshot(path="/tmp/role_editor.png")
            await pg.select_option("#edD", ""); await pg.locator("#edSave").tap(); await pg.wait_for_timeout(800)
            ov2 = await pg.evaluate("window.__jimbo.OV")
            print("options", opts, "| g/mpg before", before, "after role S", after, "| OV", ov, "| changes:", chg.strip(), "| reopen select", sel, "| after reset", ov2, "| errors", errs)
        await b.close()
async def main():
    await run(8801, None, "/tmp/n1/latest.json", "A no edits (parity)")
    await run(8801, OVR, "/tmp/n2/latest.json", "B role edits, nightly not yet applied (approx)")
    await run(8802, OVR, "/tmp/n2/latest.json", "C role edits, nightly applied (exact)")
    await run(8802, {}, "/tmp/n1/latest.json", "D edits removed, nightly still has them")
    await run(8801, None, "/tmp/n1/latest.json", "UI", ui=True)
asyncio.run(main())
for s in srv: s.terminate()
