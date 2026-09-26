# Explore the human studio

Open the [WebGPU studio](https://mosure.github.io/burn_human/) or run from the
repository root:

```sh
cargo run --locked -p bevy_burn_human
```

The native application finds the repository's `assets/model` directory.
For a packaged executable, put `assets` beside it or set `BEVY_ASSET_ROOT` to
its parent directory. The studio and camera start independently of Anny asset
loading. Native file selection and saving use the platform dialog; Linux
needs an XDG desktop portal or Zenity. URL/path imports remain available.

Models download only when requested. Inference runs locally in Rust/Burn on
Bevy's GPU device, in both native WGPU and browser WebGPU. There is no hosted
text or image inference service. The default CDN sources and manifest
identities are already configured; custom sources live under **Advanced**.

## Choose a mode

| Mode | Start here | Explore next |
| --- | --- | --- |
| Motion | **Load motion models**, enter a prompt, **Generate motion** | Action examples, duration, timed world-space paths, playback speed, scrubbing, joint offsets and JSON export/import |
| SOMA controls | **Load SOMA body** | T pose / Relaxed / Wave, live joint rotations, identity components, bone proportions and correctives |
| Image pose | **Choose image**, **Load image pose models**, **Estimate pose** | Crop positioning, image camera calibration, detected keypoints, editable inferred SOMA identity and pose export |
| Anny | The bundled body needs no large-model download | Body shape, individual bone rotations, procedural motion and random poses with **R** |

ARDY generates Anny/Core motion; it does not generate SOMA motion directly.
GEM-X currently estimates one person from one image. Inferred MHR identities
can be adjusted in **SOMA controls**; **Start a native SOMA identity** switches
to canonical SOMA identity controls and pose correctives.

## Camera and world paths

| Input | Action |
| --- | --- |
| Left drag | Orbit the subject |
| Right drag | Pan |
| Scroll wheel | Zoom |
| F | Frame the current body or motion path |
| Home / Front / Side / Top | Reset or choose a useful camera angle |
| Edit path in scene | Left-click ground to add a point; drag an existing marker to move it |
| Middle drag while editing a path | Orbit without moving waypoints |
| Delete while editing a path | Remove the selected point |
| Escape | Finish path editing |

Camera input is suppressed while using UI controls or typing. During path
editing, right-drag and scroll still pan and zoom. The selected waypoint is
white; requested paths are gold and generated paths are blue.

Try **Straight**, **Turn** or **Loop**, then drag the markers. Waypoint timing
is shown in seconds. Select a point to adjust world X/Z in metres, facing
direction or root height. Adding points or changing duration distributes
their times across the clip; subsequent timing edits remain ordered and
inside its duration. Path constraints are soft conditions, so the generated
trajectory may deviate. Clear the path to explore text conditioning alone.

## Bodies and image conditions

SOMA's **Live preview** coalesces edits while evaluation is in progress. Turn
it off to make several changes, then **Apply body controls**. Search joint
names to find hands, fingers, head or legs. Rotation sliders are local
axis-angle components, rather than independent Euler angles.

For GEM-X, choose a PNG/JPEG with one clearly visible person (up to 8 MiB and
4096 × 4096 pixels). Drag the preview to center the person crop and adjust
**Crop size**. The gold rectangle is the crop. Image camera calibration is
separate from the 3D scene camera; the default focal length is the image
diagonal, or you can enter a calibrated value.

After estimation, green dots show confident image keypoints and the 3D view
shows the fitted SOMA body. Changing the image, crop or image camera makes the
old result stale: estimate again before exporting it as the current image
result. Open **SOMA controls** to adjust its identity or joints. Native export
opens a save dialog; browser export downloads JSON.

## Loading and hardware

The complete catalog contains 14.1 GB of weights; each mode loads only its
required components. Authenticated disk/CacheStorage caches have an 8 GiB
budget, so switching between large models can evict older parts. A warm load
still reads and verifies the model and uploads its tensors to the GPU.

Sharding bounds transport and WASM memory; it does not remove GPU memory
requirements. Qualification uses an RTX PRO 6000 Blackwell with 96 GiB VRAM,
not a mobile or minimum-memory GPU. See [CDN loading](cdn.md) and the
[numerical/model guide](motion.md) for artifact contracts and model scope.
