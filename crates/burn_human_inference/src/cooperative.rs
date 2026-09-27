//! Keep browser input and rendering responsive between inference submissions.
//! This yields to the event loop, without waiting for GPU completion.

/// Yield one browser task. Native callers already run on background workers.
/// A resolved Promise would only yield a microtask and starve input/animation.
pub async fn yield_to_browser() {
    #[cfg(all(target_arch = "wasm32", feature = "cooperative"))]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen(inline_js = "
let channel;
const pending = [];
export function inferenceYield() {
    if (globalThis.scheduler?.yield) return globalThis.scheduler.yield();
    return new Promise(resolve => {
        if (!channel) {
            channel = new MessageChannel();
            channel.port1.onmessage = () => pending.shift()();
        }
        pending.push(resolve);
        channel.port2.postMessage(0);
    });
}
")]
        extern "C" {
            #[wasm_bindgen(js_name = inferenceYield)]
            fn inference_yield() -> js_sys::Promise;
        }
        let _ = wasm_bindgen_futures::JsFuture::from(inference_yield()).await;
    }
}
