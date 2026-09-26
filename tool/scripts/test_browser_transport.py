#!/usr/bin/env python3
"""Fault-test the generated production transport module in browser CacheStorage.

Serve the repository on loopback and build tool/web/out as in docs/motion.md.
The synthetic network part is intercepted by Playwright; no model is required.
"""
import argparse
import asyncio
import hashlib
import json
import pathlib
from playwright.async_api import async_playwright


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--chrome", default="/usr/bin/google-chrome-stable")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--wasm-out", type=pathlib.Path, default=pathlib.Path("tool/web/out"))
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    modules = list((root / args.wasm_out / "snippets").glob("burn_human_inference-*/inline0.js"))
    if len(modules) != 1:
        raise RuntimeError("Build a single current wasm-bindgen output in tool/web/out first")
    module = modules[0]
    url = args.base.rstrip("/") + "/__transport_test_part.bin"
    fixture = b"verified browser artifact"
    network = dict(requests=0, body=fixture)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.chrome, headless=True)
        page = await browser.new_page()

        async def response(route):
            network["requests"] += 1
            await route.fulfill(status=200, body=network["body"],
                                headers={"content-type": "application/octet-stream"})

        await page.route(url, response)
        await page.goto(args.base)
        await page.evaluate("""async c=>{
            window.transport=await import(c.module);
            window.part=c;
            await caches.delete('burn-human-motion-parts-v1');
            window.readPart=()=>transport.motion_part(c.url,c.size,c.digest);
        }""", dict(module=args.base.rstrip("/") + "/" + module.relative_to(root).as_posix(),
                   url=url, size=len(fixture), digest=hashlib.sha256(fixture).hexdigest()))

        async def read():
            data = await page.evaluate("async()=>Array.from(await readPart())")
            assert bytes(data) == fixture

        await read()
        assert network["requests"] == 1
        await read()
        assert network["requests"] == 1, "warm cache fetched the network"
        await page.evaluate("""async()=>{
            const c=await caches.open('burn-human-motion-parts-v1');
            await c.put(part.url,new Response('corrupt'));
        }""")
        await read()
        assert network["requests"] == 2
        for cls, method in [("Cache", "match"), ("CacheStorage", "open"), ("Cache", "put")]:
            await page.evaluate("""async()=>{
                const c=await caches.open('burn-human-motion-parts-v1');
                await c.delete(part.url);
            }""")
            before = network["requests"]
            await page.evaluate("""([cls,method])=>{
                window.savedMethod=globalThis[cls].prototype[method];
                globalThis[cls].prototype[method]=async()=>{throw Error('Injected storage failure');};
            }""", [cls, method])
            try:
                await read()
                assert network["requests"] == before + 1
            finally:
                await page.evaluate("([cls,method])=>{globalThis[cls].prototype[method]=savedMethod;}", [cls, method])

        assert await page.evaluate("""async()=>{
            try {await transport.motion_read(part.url,part.size-1);return false;}
            catch(_){return true;}
        }"""), "oversize response was accepted"
        await page.evaluate("async()=>{await caches.delete('burn-human-motion-parts-v1');}")
        network["body"] = b"untrusted bytes"
        assert await page.evaluate("""async()=>{
            try {await readPart();return false;}catch(_){return true;}
        }"""), "corrupt network response was accepted"
        report = dict(browser=browser.version, transport_js_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                      checks={name: "passed" for name in ["cold_read", "warm_cache_no_network", "corrupt_cache_repair",
                                                         "cache_match_failure", "cache_open_failure", "cache_write_failure",
                                                         "oversize_response_rejected", "corrupt_network_rejected"]})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
