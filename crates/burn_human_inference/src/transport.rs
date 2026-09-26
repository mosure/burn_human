//! Bounded native file/HTTP and browser fetch/CacheStorage transports.
use anyhow::{Result, ensure};
use burn_human_motion::artifacts::{
    MAX_MANIFEST_BYTES, MAX_PART_BYTES, Manifest, Part, PartReader,
};

pub struct ModelSource {
    pub base: String,
    #[cfg(not(target_arch = "wasm32"))]
    pub cache_hits: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pub downloaded_bytes: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pub cache: Option<std::path::PathBuf>,
}

impl ModelSource {
    /// Use the bounded shared disk cache for native HTTP bundles. Browser
    /// sources always use authenticated CacheStorage, with the same 8 GiB budget.
    pub fn cached(base: String) -> Self {
        let source = Self::new(base);
        #[cfg(not(target_arch = "wasm32"))]
        let source = {
            let mut source = source;
            if source.base.starts_with("https://") || source.base.starts_with("http://") {
                source.cache = std::env::var_os("XDG_CACHE_HOME")
                    .map(std::path::PathBuf::from)
                    .or_else(|| {
                        std::env::var_os("HOME").map(|p| std::path::PathBuf::from(p).join(".cache"))
                    })
                    .map(|p| p.join("burn-human"));
            }
            source
        };
        source
    }
    pub fn new(base: String) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            cache_hits: 0,
            #[cfg(not(target_arch = "wasm32"))]
            downloaded_bytes: 0,
            #[cfg(not(target_arch = "wasm32"))]
            cache: None,
        }
    }
    pub async fn manifest(&self, digest: Option<&str>) -> Result<Manifest> {
        let bytes =
            read_bounded(&format!("{}/manifest.json", self.base), MAX_MANIFEST_BYTES).await?;
        Manifest::from_bytes(&bytes, digest)
    }
    pub async fn asset(&self, asset: &burn_human_motion::artifacts::Asset) -> Result<Vec<u8>> {
        let bytes = read_bounded(&format!("{}/{}", self.base, asset.path), asset.size).await?;
        asset.verify(&bytes)?;
        Ok(bytes)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read_bounded(location: &str, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    if location.starts_with("https://") || location.starts_with("http://") {
        let mut response = ureq::get(location).call()?;
        if let Some(size) = response.headers().get("content-length") {
            ensure!(
                size.to_str()?.parse::<u64>()? <= limit as u64,
                "response exceeds byte limit"
            );
        }
        response
            .body_mut()
            .as_reader()
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)?;
    } else {
        let file = std::fs::File::open(location)?;
        ensure!(
            file.metadata()?.len() <= limit as u64,
            "file exceeds byte limit"
        );
        file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    }
    ensure!(bytes.len() <= limit, "response exceeds byte limit");
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
impl PartReader for ModelSource {
    async fn read_part(&mut self, part: &Part) -> Result<Vec<u8>> {
        ensure!(part.size <= MAX_PART_BYTES, "part exceeds byte limit");
        let cached = self.cache.as_ref().map(|p| p.join(part.path()));
        if let Some(path) = &cached {
            if let Ok(bytes) = read_bounded(&path.to_string_lossy(), part.size).await
                && part.verify(&bytes).is_ok()
            {
                self.cache_hits += 1;
                return Ok(bytes);
            }
            if path.exists() {
                let _ = std::fs::remove_file(path);
            }
        }
        let bytes = read_bounded(&format!("{}/{}", self.base, part.path()), part.size).await?;
        part.verify(&bytes)?;
        self.downloaded_bytes += bytes.len();
        if let Some(path) = cached {
            // A cache is an optimization: unavailable storage must not reject
            // bytes that already passed the artifact's integrity checks.
            let _ = cache_part(&path, &bytes);
        }
        Ok(bytes)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cache_part(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir)?;
    // A fixed 8 GiB disk budget. Evict oldest parts before committing a new one.
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok().map(|m| (e.path(), m)))
        .filter(|(p, m)| m.is_file() && p.extension().is_some_and(|v| v == "bin"))
        .collect();
    entries.sort_by_key(|(_, m)| m.modified().ok());
    let mut total: u64 = entries.iter().map(|(_, m)| m.len()).sum();
    for (old, meta) in entries {
        if total + bytes.len() as u64 <= 8 * 1024 * 1024 * 1024 {
            break;
        }
        std::fs::remove_file(old)?;
        total -= meta.len();
    }
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_extension(format!("{}.{unique}.tmp", std::process::id()));
    let result = std::fs::write(&temporary, bytes).and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;
    pub fn error_message(error: JsValue) -> String {
        error
            .as_string()
            .or_else(|| {
                js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                    .ok()
                    .and_then(|v| v.as_string())
            })
            .unwrap_or_else(|| "Browser request failed".into())
    }

    #[wasm_bindgen(inline_js = r#"
async function bounded(response, limit) {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const length = Number(response.headers.get('content-length'));
  if (length > limit) throw new Error('Response exceeds byte limit');
  if (!response.body) throw new Error('Streaming fetch unavailable');
  const reader = response.body.getReader();
  const chunks = []; let size = 0;
  try {
    while (true) {
      const {done,value} = await reader.read(); if (done) break;
      size += value.length;
      if (size > limit) throw new Error('Response exceeds byte limit');
      chunks.push(value);
    }
  } finally { await reader.cancel(); reader.releaseLock(); }
  const data = new Uint8Array(size); let offset = 0;
  for (const c of chunks) {data.set(c,offset);offset+=c.length;}
  return data;
}
async function authentic(data, digest, size) {
  if (data.length !== size) return false;
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256',data));
  return Array.from(hash,b=>b.toString(16).padStart(2,'0')).join('') === digest;
}
export async function motion_read(url, limit) {
  return bounded(await fetch(url),limit);
}
let partIndex = null;
let partBytes = 0;
const cacheBudget = 8 * 1024 * 1024 * 1024;
export async function motion_part(url, size, digest) {
  let cache = null;
  try {cache = await caches.open('burn-human-motion-parts-v1');} catch (_) {}
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        try {const b=await bounded(hit,size);if(await authentic(b,digest,size))return b;}catch(_){}
        await cache.delete(url);
      }
    } catch (_) {cache = null;}
  }
  const data = await motion_read(url,size);
  if (!await authentic(data,digest,size)) throw new Error('Artifact digest mismatch');
  if (cache) {
    try {
      // Build the FIFO inventory once per page, not once per large part.
      // Quota/storage errors leave authenticated inference available.
      if (!partIndex) {
        partIndex = new Map(); partBytes = 0;
        for (const key of await cache.keys()) {
          const response = await cache.match(key);
          const n = Number(response?.headers.get('content-length')) || 0;
          partIndex.set(key.url,n); partBytes += n;
        }
      }
      const key = new URL(url,globalThis.location.href).href;
      partBytes -= partIndex.get(key) || 0; partIndex.delete(key);
      for (const [old,n] of partIndex) {
        if (partBytes + data.length <= cacheBudget) break;
        await cache.delete(old); partIndex.delete(old); partBytes -= n;
      }
      await cache.put(url,new Response(data,{headers:{'content-length':String(data.length)}}));
      partIndex.set(key,data.length); partBytes += data.length;
    } catch (_) {}
  }
  return data;
}
"#)]
    extern "C" {
        #[wasm_bindgen(catch)]
        pub async fn motion_read(url: &str, limit: usize) -> Result<js_sys::Uint8Array, JsValue>;
        #[wasm_bindgen(catch)]
        pub async fn motion_part(
            url: &str,
            size: usize,
            digest: &str,
        ) -> Result<js_sys::Uint8Array, JsValue>;
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn read_bounded(location: &str, limit: usize) -> Result<Vec<u8>> {
    Ok(browser::motion_read(location, limit)
        .await
        .map_err(|e| anyhow::anyhow!("fetch: {}", browser::error_message(e)))?
        .to_vec())
}
#[cfg(target_arch = "wasm32")]
impl PartReader for ModelSource {
    async fn read_part(&mut self, part: &Part) -> Result<Vec<u8>> {
        ensure!(part.size <= MAX_PART_BYTES, "part exceeds byte limit");
        let bytes = browser::motion_part(
            &format!("{}/{}", self.base, part.path()),
            part.size,
            &part.sha256,
        )
        .await
        .map_err(|e| anyhow::anyhow!("artifact fetch: {}", browser::error_message(e)))?
        .to_vec();
        part.verify(&bytes)?;
        Ok(bytes)
    }
}
