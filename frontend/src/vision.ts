import type {Conditions, Direction} from './types';
export interface VisionPort {evaluate(bitmap: ImageBitmap, direction: Direction, mode?: 'camera' | 'upload'): Promise<Conditions>; close(): void}
declare global {interface Window {__FACETELLER_TEST__?: {evaluate?: (direction: Direction) => Conditions | Promise<Conditions>; fixture?: boolean}}}
export function createVision(): VisionPort {
  // Compile-time test mode only: dead-code eliminated from production builds.
  if (import.meta.env.MODE === 'test' && window.__FACETELLER_TEST__?.evaluate) {
    const test = window.__FACETELLER_TEST__;
    return {evaluate: async (bitmap, direction) => {bitmap.close(); return await test.evaluate!(direction);}, close() {}};
  }
  const worker = new Worker('/assets/mediapipe/vision-worker.js');
  let seq = 0, closed = false;
  const pending = new Map<number, {resolve: (value: Conditions) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout>}>();
  const close = () => {closed = true; worker.terminate(); for (const p of pending.values()) {clearTimeout(p.timer); p.reject(new Error('얼굴 검사가 종료되었습니다.'));} pending.clear();};
  worker.onmessage = ({data}) => {const p = pending.get(data.id); if (!p) return; pending.delete(data.id); clearTimeout(p.timer); if (data.error) p.reject(new Error(data.error)); else p.resolve(data.result);};
  worker.onerror = () => close();
  return {close, evaluate(bitmap, direction, mode = 'camera') {
    if (closed) {bitmap.close(); return Promise.reject(new Error('얼굴 검사기가 종료되었습니다.'));}
    const id = ++seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {pending.delete(id); reject(new Error('얼굴 검사 응답이 지연되었습니다. 다시 시도해 주세요.')); close();}, 25000);
      pending.set(id, {resolve, reject, timer});
      worker.postMessage({id, bitmap, direction, mode}, [bitmap]);
    });
  }};
}
export async function decodeFile(file: File): Promise<ImageBitmap> {
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error('JPG, PNG, WebP 파일을 선택해 주세요.');
  if (!file.size || file.size > 10*1024*1024) throw new Error('사진 한 장은 0바이트보다 크고 10 MiB 이하여야 합니다.');
  let bitmap: ImageBitmap;
  try {bitmap = await createImageBitmap(file, {imageOrientation: 'from-image'});} catch {throw new Error('이미지를 읽을 수 없습니다. 다른 파일을 선택해 주세요.');}
  if (bitmap.width * bitmap.height > 24_000_000 || Math.min(bitmap.width,bitmap.height) < 160) {bitmap.close(); throw new Error('사진은 가로·세로 160px 이상, 2,400만 화소 이하여야 합니다.');}
  if (Math.max(bitmap.width,bitmap.height) > 1600) {
    const scale = 1600 / Math.max(bitmap.width,bitmap.height);
    const small = await createImageBitmap(bitmap, {resizeWidth: Math.round(bitmap.width*scale), resizeHeight: Math.round(bitmap.height*scale)});
    bitmap.close(); return small;
  }
  return bitmap;
}
