import {mkdir, cp, readFile, writeFile, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const target = root + 'public/assets/mediapipe/';
await mkdir(target, {recursive:true});
await mkdir(root+'public/assets/licenses', {recursive:true});
for (const font of ['noto-sans-kr','noto-serif-kr','noto-serif-tc']) {
  await cp(root+`node_modules/@fontsource/${font}/LICENSE`, root+`public/assets/licenses/${font}.txt`);
}
await cp(root + 'node_modules/@mediapipe/tasks-vision/wasm', target + 'wasm', {recursive:true});
const url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
try {await stat(target+'face_landmarker.task');}
catch {const response = await fetch(url); if (!response.ok) throw Error(`Model asset download HTTP ${response.status}`); await writeFile(target+'face_landmarker.task', new Uint8Array(await response.arrayBuffer()));}
await build({entryPoints:[root+'src/vision.worker.ts'], outfile:target+'vision-worker.js', bundle:true, format:'iife', platform:'browser', target:'es2022', minify:true});
const bytes = await readFile(target+'face_landmarker.task');
await writeFile(target+'manifest.json', JSON.stringify({model_source:url, model_sha256:createHash('sha256').update(bytes).digest('hex'), tasks_vision:'0.10.32', delegate:'CPU', runtime_network:'same-origin only'},null,2)+'\n');
console.log('Local MediaPipe assets ready: /assets/mediapipe/');
