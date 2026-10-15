import {DIRECTIONS, type Conditions, type Direction} from './types';
// Configurable acquisition defaults; these are not validated clinical thresholds.
export const ACQUISITION = {holdMs: 800, goodFrames: 8, maxGapMs: 300, maxAgeMs: 400, frontYaw: 12, obliqueMin: 20, obliqueMax: 60, pitchCenter: -5, pitchTolerance: 16, roll: 12, minFacePixels: 150, minFaceRatio: .25, maxFaceRatio: .8, minBrightness: 45, maxBrightness: 235, maxSmile: .3, maxBlink: .5};
export interface Token {generation: number; direction: Direction | undefined}
export class CaptureState {
  private generation = 0;
  private index = 0;
  private since: number | null = null;
  private previous = -Infinity;
  private mediaTime = -Infinity;
  private frames = 0;
  private pending = false;
  private submitted = false;
  token(): Token { return {generation: this.generation, direction: DIRECTIONS[this.index]}; }
  reset() {this.generation++; this.index = 0; this.resetHold(); this.pending = false; this.submitted = false; this.mediaTime = -Infinity;}
  resetHold() {this.since = null; this.frames = 0; this.previous = -Infinity;}
  matches(token: Token) {return token.generation === this.generation && token.direction === DIRECTIONS[this.index];}
  update(c: Conditions, capturedAt: number, evaluatedAt: number, mediaTime: number, token: Token) {
    const ignored = {ignored: true, progress: 0, capture: false};
    if (!this.matches(token) || !token.direction || this.pending) return ignored;
    if (![capturedAt, evaluatedAt, mediaTime].every(Number.isFinite) || mediaTime <= this.mediaTime || capturedAt <= this.previous || evaluatedAt < capturedAt || evaluatedAt - capturedAt > ACQUISITION.maxAgeMs) {this.resetHold(); return ignored;}
    this.mediaTime = mediaTime;
    if (capturedAt - this.previous > ACQUISITION.maxGapMs) this.resetHold();
    this.previous = capturedAt;
    if (![c.face, c.direction, c.size, c.brightness, c.expression].every(Boolean)) {this.since = null; this.frames = 0; return {ignored: false, progress: 0, capture: false};}
    if (this.since === null) this.since = capturedAt;
    this.frames++;
    const progress = Math.min(1, (capturedAt - this.since) / ACQUISITION.holdMs, this.frames / ACQUISITION.goodFrames);
    this.pending = progress === 1;
    return {ignored: false, progress, capture: this.pending};
  }
  commit(token: Token) {if (!this.pending || !this.matches(token)) return false; this.index++; this.generation++; this.resetHold(); this.pending = false; return true;}
  submitOnce() {if (this.index !== 3 || this.submitted) return false; this.submitted = true; return true;}
}
export interface Pose {pitch: number; yaw: number; roll: number}
// Direction IDs refer to the unmirrored image sent to the server. A mirrored
// self-view reverses only the on-screen instruction, never the captured pixels.
export function captureDirectionLabel(direction: Direction, mode: 'camera' | 'upload' = 'camera'): string {
  if (direction === 'front') return '정면';
  const screenLeft = (direction === 'left_oblique') !== (mode === 'camera');
  return screenLeft ? '화면 왼쪽' : '화면 오른쪽';
}
export function captureDirectionHint(direction: Direction, mode: 'camera' | 'upload' = 'camera'): string {
  if (direction === 'front') return '고개를 기울이지 말고 정면을 봐 주세요.';
  const label = captureDirectionLabel(direction, mode);
  if (mode === 'upload') return `${label}을 향한 사진을 선택해 주세요.`;
  return label === '화면 왼쪽'
    ? '← 화면 속 코가 왼쪽을 향하도록 고개를 돌려 주세요.'
    : '화면 속 코가 오른쪽을 향하도록 고개를 돌려 주세요. →';
}
// MatrixData's packed data are column-major. Equivalent to scipy as_euler('xyz')
// for a proper rotation matrix; source pixels and detection are never mirrored.
export function eulerXYZ(m: number[]): Pose {
  if (m.length !== 16 || m.some(v => !Number.isFinite(v))) return {pitch: NaN, yaw: NaN, roll: NaN};
  const norm = (a: number, b: number, c: number) => Math.hypot(a, b, c);
  const x = norm(m[0], m[1], m[2]), y = norm(m[4], m[5], m[6]), z = norm(m[8], m[9], m[10]);
  const yaw = Math.asin(Math.max(-1, Math.min(1, -m[2] / x)));
  const pitch = Math.abs(Math.cos(yaw)) > 1e-6 ? Math.atan2(m[6] / y, m[10] / z) : Math.atan2(-m[9] / z, m[5] / y);
  const roll = Math.abs(Math.cos(yaw)) > 1e-6 ? Math.atan2(m[1] / x, m[0] / x) : 0;
  return {pitch: pitch * 180 / Math.PI, yaw: yaw * 180 / Math.PI, roll: roll * 180 / Math.PI};
}
export function quality(p: Pose, direction: Direction, faceCount: number, faceWidth: number, frameWidth: number, brightness: number, smile: number, blink: number, mode: 'camera' | 'upload' = 'camera'): Conditions {
  const a = ACQUISITION;
  const yawOk = direction === 'front' ? Math.abs(p.yaw) <= a.frontYaw : Math.abs(p.yaw) >= a.obliqueMin && Math.abs(p.yaw) <= a.obliqueMax && (direction === 'left_oblique' ? p.yaw < 0 : p.yaw > 0);
  const c: Conditions = {face: faceCount === 1, direction: yawOk && Math.abs(p.pitch - a.pitchCenter) <= a.pitchTolerance && Math.abs(p.roll) <= a.roll, size: faceWidth >= a.minFacePixels && (mode === 'upload' || (faceWidth / frameWidth >= a.minFaceRatio && faceWidth / frameWidth <= a.maxFaceRatio)), brightness: brightness >= a.minBrightness && brightness <= a.maxBrightness, expression: smile <= a.maxSmile && blink <= a.maxBlink, yaw: p.yaw, message: ''};
  c.message = !c.face ? (faceCount > 1 ? '한 사람만 화면에 들어오게 해 주세요.' : '얼굴을 안내선 안에 맞춰 주세요.') : !c.direction ? captureDirectionHint(direction, mode) : !c.size ? (mode === 'upload' ? '얼굴이 더 크게 나온 원본 사진을 선택해 주세요.' : '얼굴이 안내선에 맞도록 거리를 조절해 주세요.') : !c.brightness ? '얼굴에 고르게 빛이 들어오도록 조명을 조절해 주세요.' : !c.expression ? '눈을 뜨고 편안한 무표정을 유지해 주세요.' : '좋습니다. 잠시 그대로 유지해 주세요.';
  return c;
}
