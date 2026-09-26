//! Bounded local file selection and clip export for the browser viewer.
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
export async function select_motion_file(accept, limit) {
  return new Promise((resolve,reject)=>{
    const input=document.createElement('input');input.type='file';input.accept=accept;
    input.oncancel=()=>resolve(null);
    input.onchange=async()=>{
      try {
        const file=input.files[0];if(!file){resolve(null);return;}
        if(file.size>limit)throw Error('Selected file exceeds size limit');
        resolve(new Uint8Array(await file.arrayBuffer()));
      }catch(error){reject(error);}
    };
    input.click();
  });
}
export function download_motion_clip(bytes) {
  download_motion_artifact(bytes,'motion.json');
}
export function download_motion_artifact(bytes,name) {
  const url=URL.createObjectURL(new Blob([new Uint8Array(bytes)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
"#)]
extern "C" {
    #[wasm_bindgen(catch)]
    pub async fn select_motion_file(accept: &str, limit: usize) -> Result<JsValue, JsValue>;
    pub fn download_motion_clip(bytes: &[u8]);
    pub fn download_motion_artifact(bytes: &[u8], name: &str);
}
