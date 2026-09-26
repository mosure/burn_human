
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
