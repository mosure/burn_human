//! Keep browser input and rendering responsive between inference submissions.
//! This yields to the event loop, without waiting for GPU completion.

#[cfg(all(target_arch = "wasm32", feature = "cooperative"))]
mod browser {
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
export function inferenceYieldCallback() { return inferenceYield; }
")]
    extern "C" {
        #[wasm_bindgen(js_name = inferenceYield)]
        pub fn inference_yield() -> js_sys::Promise;
        #[wasm_bindgen(js_name = inferenceYieldCallback)]
        pub fn inference_yield_callback() -> js_sys::Function;
    }
}

/// Yield one browser task. Native callers already run on background workers.
/// A resolved Promise would only yield a microtask and starve input/animation.
pub async fn yield_to_browser() {
    #[cfg(all(target_arch = "wasm32", feature = "cooperative"))]
    {
        let _ = wasm_bindgen_futures::JsFuture::from(browser::inference_yield()).await;
    }
}

#[cfg(all(target_arch = "wasm32", feature = "transport"))]
pub(crate) fn yield_callback() -> js_sys::Function {
    browser::inference_yield_callback()
}
