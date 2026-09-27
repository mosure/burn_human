#!/usr/bin/env python3
"""Profile the existing real-checkpoint WebGPU validator, including event-loop gaps.

Pass the generated module URL, exported validation function, and its arguments
as a JSON array. Event-loop timing includes model loading and cold compilation;
the validator separately reports synchronized warm inference. No software GPU
fallback is accepted. Requires Playwright and a WebGPU-capable Chromium.
"""
import argparse
import asyncio
import json
import pathlib
from playwright.async_api import async_playwright


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['chrome', 'origin', 'module', 'function', 'args', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cpu-profile', help='Optional Chrome CPU profile; diagnostic timings include profiler overhead')
    args = parser.parse_args()
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.chrome, headless=True, args=[
            '--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis',
            '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE',
            '--disable-vulkan-surface', '--ignore-gpu-blocklist'])
        page = await browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: print(message.type, message.text[:1600], flush=True))
        await page.goto(args.origin)
        profiler = None
        if args.cpu_profile:
            profiler = await page.context.new_cdp_session(page)
            await profiler.send('Performance.enable')
            await profiler.send('Profiler.enable')
            await profiler.send('Profiler.start')
        result = await page.evaluate('''async({module, func, args, trace}) => {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter || adapter.info.isFallbackAdapter) throw Error('Hardware WebGPU required');
            const api = await import(module);
            const wasm = await api.default();
            const frames = [], frameIntervals = [], tasks = [], taskIntervals = [], gpuCalls = [];
            if (trace) {
                for (const [type, methods] of [[GPUDevice, ['createBuffer', 'createShaderModule', 'createComputePipeline']],
                                               [GPUQueue, ['writeBuffer', 'submit']]]) {
                    for (const method of methods) {
                        const original = type.prototype[method];
                        type.prototype[method] = function(...args) {
                            const start = performance.now();
                            try { return original.apply(this, args); }
                            finally {
                                const duration = performance.now() - start;
                                if (duration >= 2) gpuCalls.push({method, start, duration});
                            }
                        };
                    }
                }
            }
            let running = true, previous = performance.now(), peak = wasm.memory.buffer.byteLength;
            function tick(t) {
                if (!running) return;
                frames.push(t - previous); frameIntervals.push({start:previous, end:t, duration:t-previous});
                previous = t; requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
            const observer = new PerformanceObserver(list => { for (const entry of list.getEntries()) {
                tasks.push(entry.duration); taskIntervals.push({start:entry.startTime, duration:entry.duration});
            }});
            observer.observe({type: 'longtask', buffered: false});
            const timer = setInterval(() => { peak = Math.max(peak, wasm.memory.buffer.byteLength); }, 25);
            const start = performance.now();
            const report = JSON.parse(await api[func](...args));
            await new Promise(resolve => setTimeout(resolve, 30));
            running = false; clearInterval(timer); observer.disconnect();
            frames.shift(); frames.sort((a,b) => a-b);
            const percentile = p => frames[Math.min(frames.length-1, Math.floor(frames.length*p))];
            const phases = report.phases?.map(phase => {
                const from = start + phase.start_ms, to = start + phase.end_ms;
                const gaps = frameIntervals.slice(1).filter(x => x.end >= from && x.start <= to).map(x => x.duration).sort((a,b) => a-b);
                const longTasks = taskIntervals.filter(x => x.start+x.duration >= from && x.start <= to);
                return {...phase, max_gap_ms:gaps.at(-1) ?? null,
                    p95_gap_ms:gaps[Math.floor(gaps.length*.95)] ?? null,
                    long_task_count:longTasks.length, max_task_ms:Math.max(0,...longTasks.map(x=>x.duration))};
            });
            return {report, elapsed_ms: performance.now()-start, adapter: {vendor:adapter.info.vendor, architecture:adapter.info.architecture, fallback:adapter.info.isFallbackAdapter},
              event_loop: {scope:report.scope ?? 'load, cold compilation, parity, warm inference', frames:frames.length, p50_ms:percentile(.5), p95_ms:percentile(.95), p99_ms:percentile(.99), max_ms:frames.at(-1), long_tasks_ms:tasks},
              event_loop_by_phase: phases,
              diagnostics: trace ? {cpu_profile:true, gpu_call_scope:'synchronous host API time, not GPU execution', gpu_calls:gpuCalls, task_intervals:taskIntervals} : undefined,
              wasm_linear_memory_peak_bytes:Math.max(peak,wasm.memory.buffer.byteLength)};
        }''', {'module': args.module, 'func': args.function, 'args': json.loads(args.args), 'trace': bool(args.cpu_profile)})
        if profiler:
            profile = await profiler.send('Profiler.stop')
            result['diagnostics']['performance_metrics'] = (await profiler.send('Performance.getMetrics'))['metrics']
            pathlib.Path(args.cpu_profile).write_text(json.dumps(profile['profile']) + '\n')
        if errors or result['report'].get('passed') is False:
            raise RuntimeError(dict(errors=errors, result=result))
        result.update(browser=browser.version, module=args.module, function=args.function, args=json.loads(args.args))
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
        event_loop = result['event_loop']
        summary = {k: v for k, v in event_loop.items() if k != 'long_tasks_ms'}
        summary['long_task_count'] = len(event_loop['long_tasks_ms'])
        print('PASS', args.function, result['elapsed_ms'], summary, flush=True)
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
