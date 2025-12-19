# bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria

Forked from https://github.com/mosure/burn_human.git

Bevy port of the CodePen https://codepen.io/Dave-Horner/pen/zxqgOOo “absurdia -stuck in an endless loop in mcdonalds playground slide” visualization, using `bevy_burn_human` as the moving subject.

<p align="center">
	<img alt="Screenshot" src="https://raw.githubusercontent.com/davehorner/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/main/crates/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/screenshot.jpg" />
</p>

## Run (native)

```cmd
task mcbaise:native
```

## Run (web)

This repo uses go-task (https://taskfile.dev/) to build and serve the wasm demo:

```cmd
task mcbaise:wasm:serve
```

Open the printed URL (served from repo root so `/assets/...` is available).

Notes:
- The published crates.io package name for this demo is `bevy_mcbaise_fantasmagoria`.
- The Taskfile runs `wasm-bindgen` and writes output to `crates/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/www/pkg`.

## GitHub Pages (wasm hosting)

The repo root includes a GitHub Actions workflow that builds and publishes this demo to GitHub Pages:

- Workflow: `.github/workflows/deploy.yml`
- Pages URL: `https://davehorner.github.io/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/`

Enable Pages (create if missing; uses GitHub Actions as the build source):

```cmd
gh api -X POST repos/davehorner/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```

Or update an existing Pages config:

```cmd
gh api -X PUT repos/davehorner/bevy_Mcbaise_Palee_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```
