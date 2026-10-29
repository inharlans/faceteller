import {FaceLandmarker, FilesetResolver} from '@mediapipe/tasks-vision';
import {eulerXYZ, guideFit, quality} from './capture';
import type {Direction} from './types';

let model: Promise<FaceLandmarker> | undefined;
const load = () => model ||= FilesetResolver.forVisionTasks('/assets/mediapipe/wasm').then(files => FaceLandmarker.createFromOptions(files, {
  baseOptions: {modelAssetPath: '/assets/mediapipe/face_landmarker.task', delegate: 'CPU'}, runningMode: 'IMAGE', numFaces: 2,
  outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  minFaceDetectionConfidence: .5, minFacePresenceConfidence: .5, minTrackingConfidence: .5,
}));
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<{id: number; bitmap: ImageBitmap; direction: Direction; mode?: 'camera' | 'upload'}>) => {
  const {id, bitmap, direction, mode = 'camera'} = event.data;
  queue = queue.then(async () => {
    try {
      const detector = await load();
      const r = detector.detect(bitmap);
      const landmarks = r.faceLandmarks[0] || [];
      let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
      for (const p of landmarks) {x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);}
      x0 = Math.max(0, x0); x1 = Math.min(1, x1); y0 = Math.max(0, y0); y1 = Math.min(1, y1);
      let brightness = 0;
      if (x1 > x0 && y1 > y0) {
        const canvas = new OffscreenCanvas(48, 48), ctx = canvas.getContext('2d', {willReadFrequently: true});
        if (!ctx) throw new Error('brightness canvas');
        ctx.drawImage(bitmap, x0 * bitmap.width, y0 * bitmap.height, (x1-x0)*bitmap.width, (y1-y0)*bitmap.height, 0, 0, 48, 48);
        const pixels = ctx.getImageData(0,0,48,48).data;
        for (let i = 0; i < pixels.length; i += 4) brightness += .2126*pixels[i] + .7152*pixels[i+1] + .0722*pixels[i+2];
        brightness /= 48*48; canvas.width = 0; canvas.height = 0;
      }
      const categories = r.faceBlendshapes[0]?.categories || [];
      const blend = (name: string) => categories.find(c => c.categoryName === name)?.score ?? Infinity;
      const result = quality(eulerXYZ(r.facialTransformationMatrixes[0]?.data || []), direction, r.faceLandmarks.length,
        (x1-x0)*bitmap.width, bitmap.width, brightness, Math.max(blend('mouthSmileLeft'),blend('mouthSmileRight')), Math.max(blend('eyeBlinkLeft'),blend('eyeBlinkRight')), mode, guideFit(landmarks));
      self.postMessage({id, result}); // No pixels or raw coordinates leave the worker.
    } catch {self.postMessage({id, error: '얼굴 검사기를 시작할 수 없습니다. 페이지를 새로 열고 다시 시도해 주세요.'});}
    finally {bitmap.close();}
  });
};
