//! Synchronous compatibility for the explicitly non-cooperative inference path.
use std::{
    future::Future,
    task::{Context, Poll, Waker},
};

// Native jobs already run on workers. Compile out browser suspension states
// there, rather than constructing and polling a no-op yield at every stage.
pub(crate) const COOPERATIVE: bool = cfg!(target_arch = "wasm32");

/// Only used with `COOPERATIVE = false`: these futures contain tensor dispatches
/// and no suspension points. Browser applications use the yielding async APIs.
pub(crate) fn immediate<T>(future: impl Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("non-cooperative inference cannot suspend"),
    }
}
