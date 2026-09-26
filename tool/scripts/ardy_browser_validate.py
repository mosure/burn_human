#!/usr/bin/env python3
"""Run identical real-checkpoint hooks in Chrome WebGPU, with network evidence.

Serve the checkout on loopback and pass --url with bundle/reference/digest query
parameters. Requires Playwright and a WebGPU-capable Chrome/Vulkan installation.
Missing artifacts, fallback adapters, browser errors and parity failures fail.
"""
import argparse
import asyncio
import json
import pathlib
from playwright.async_api import async_playwright


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome-stable")
    ap.add_argument("--paged-text", action="store_true", help="Validate only the vocabulary pages used by the text fixtures")
    ap.add_argument("--suite", action="store_true", help="The bundle URL identifies a GEM suite with multiple pinned manifests")
    args = ap.parse_args()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.chrome, headless=True, args=[
            "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=vulkan",
            "--enable-features=Vulkan,VulkanFromANGLE", "--disable-vulkan-surface", "--disable-gpu-sandbox",
            "--enable-dawn-features=allow_unsafe_apis",
        ])
        page = await browser.new_page(viewport={"width": 1400, "height": 1000})
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: print(msg.type, msg.text[:2000], flush=True))
        page.on("response", lambda response: network.append(dict(url=response.url, status=response.status)))
        reports = []
        expected_parts = None
        corrupted_url = None
        for run in ["cold_cache", "warm_cache", "corrupt_cache"]:
            if run == "corrupt_cache":
                corrupted_url = await page.evaluate("""async()=>{
                    const cache=await caches.open('burn-human-motion-parts-v1');
                    const keys=await cache.keys();
                    if(!keys.length) throw Error('No cached parts to corrupt');
                    const key=keys[0];
                    await cache.put(key,new Response(new Uint8Array([0,1,2]),{headers:{'content-length':'3'}}));
                    return key.url;
                }""")
            network.clear()
            await page.goto(args.url)
            adapter = await page.evaluate("""async()=>{
                const a=await navigator.gpu.requestAdapter();
                if(!a) throw Error('No WebGPU adapter');
                return {vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,fallback:a.info.isFallbackAdapter};
            }""")
            if adapter.get("fallback") or "swiftshader" in str(adapter).lower():
                raise RuntimeError(f"Software adapter cannot qualify GPU performance: {adapter}")
            await page.wait_for_function("window.ardyValidation !== undefined", timeout=600000)
            result = await page.evaluate("window.ardyValidation")
            if not result["ok"] or errors:
                raise RuntimeError(dict(result=result, errors=errors))
            parts = [r for r in network if "/parts/" in r["url"] and r["url"].endswith(".bin")]
            if run == "cold_cache":
                manifests = await page.evaluate("""async(suite)=>{
                    const bundle=new URL(location.href).searchParams.get('bundle');
                    if(suite) {
                        const url=new URL(bundle,location.href);
                        const artifacts=await (await fetch(url)).json();
                        return Promise.all(Object.values(artifacts).map(async a=>{
                            const base=new URL(a.base.replace(/\\/$/,'')+'/',url);
                            return (await fetch(new URL('manifest.json',base))).json();
                        }));
                    }
                    return [(await fetch(bundle.replace(/\\/$/,'')+'/manifest.json')).json()];
                }""", args.suite)
                # CacheStorage keys include each bundle's URL, even if two
                # distinct models happen to have byte-identical zero tensors.
                expected_parts = sum(len({p["sha256"] for o in m["objects"] for p in o["parts"]}) for m in manifests)
                if args.paged_text:
                    manifest = manifests[0]
                    reference = await page.evaluate("""async()=>{
                        return (await fetch(new URL(location.href).searchParams.get('reference'))).json();
                    }""")
                    pages = {token // 512 for records in reference["modes"].values()
                             for record in records for token in record["tokens"]["input_ids"]}
                    expected_parts = len({p["sha256"] for o in manifest["objects"]
                        if not o["stage"].startswith("model.embed_tokens.page.") or int(o["stage"].rsplit(".",1)[1]) in pages
                        for p in o["parts"]})
                if len(parts) != expected_parts:
                    raise RuntimeError(f"Cold cache fetched {len(parts)} parts, expected {expected_parts}")
            elif run == "warm_cache" and parts:
                raise RuntimeError(f"Warm cache unexpectedly fetched {len(parts)} parts")
            elif run == "corrupt_cache" and [p["url"] for p in parts] != [corrupted_url]:
                raise RuntimeError(f"Corrupt cache did not repair exactly the damaged part: {parts}")
            reports.append(dict(run=run, adapter=adapter, result=result["report"],
                                part_requests=len(parts), expected_parts=expected_parts,
                                corrupted_url=corrupted_url, network=list(network)))
            print(run, result["report"]["performance"], flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(browser=browser.version, reports=reports), indent=2) + "\n")
        await page.screenshot(path=str(args.out.with_suffix(".png")), full_page=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
