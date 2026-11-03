import asyncio, pathlib
from playwright.async_api import async_playwright
root = pathlib.Path(__file__).resolve().parent.parent
async def main():
    async with async_playwright() as p:
        try:
            b = await p.chromium.launch()
        except Exception:
            b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto((root / "pitch.html").as_uri())
        await pg.wait_for_timeout(800)
        n = await pg.evaluate("document.querySelectorAll('.slide').length")
        for i in range(n):
            await pg.evaluate(f"show({i})")
            await pg.wait_for_timeout(300)
            await pg.screenshot(path=str(root / f"video/slide{i+1}.png"))
        print(n, "slides rendered")
        await b.close()
asyncio.run(main())
