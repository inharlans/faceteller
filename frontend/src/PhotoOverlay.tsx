import {useState, type CSSProperties} from 'react';
import type {FaceOverlay, Item, OverlayFeature} from './types';

// Bright hues with a dark halo stay readable on any skin tone, background or theme.
export const OVERLAY_COLORS = ['#ff5f6d', '#ffc23d', '#38e0a0', '#4cc9f0', '#c39bff', '#ff8fd0', '#9be15d', '#ffa36c'];
const RATIO_TOLERANCE = .01;

export const overlayColor = (index: number) => OVERLAY_COLORS[index % OVERLAY_COLORS.length];
export const pathPoints = (path: [number, number][]) => path.map(([x, y]) => `${x},${y}`).join(' ');

/** Classification text only for confirmed items; a held item never gets a label. */
export function featureVerdict(feature: OverlayFeature, items: Item[]) {
  const item = items.find(i => i.item_id === feature.item_id);
  return !item ? '' : item.status === 'CONFIRMED' ? item.label_name || '' : '판정 보류';
}
const DEGREE_UNITS = new Set(['deg', 'degree', 'degrees', '°']);
export function measureText(feature: OverlayFeature) {
  if (typeof feature.value !== 'number') return '';
  const value = String(Number(feature.value.toPrecision(3)));
  if (feature.unit === 'ratio') return `비율 ${value}`;
  if (feature.unit === 'log_ratio') return `차이 지수 ${value}`;
  if (feature.unit && DEGREE_UNITS.has(feature.unit)) return `${value}°`;
  return `${value}${feature.unit ? ` ${feature.unit}` : ''}`;
}
/** A readable starting view; every other measured line stays one toggle away. */
export const DEFAULT_OVERLAY_FEATURES = ['brow_length', 'eye_tilt', 'nose_width', 'mouth_width'];
export type OverlayPreset = 'default' | 'all' | 'none';
export function presetHidden(overlay: FaceOverlay | null | undefined, preset: OverlayPreset = 'default') {
  const ids = (overlay?.features ?? []).map(f => f.item_id);
  if (preset === 'all') return [];
  if (preset === 'none') return ids;
  const shown = ids.filter(id => DEFAULT_OVERLAY_FEATURES.includes(id));
  const keep = shown.length ? shown : ids.slice(0, DEFAULT_OVERLAY_FEATURES.length);
  return ids.filter(id => !keep.includes(id));
}
/** True when the decoded photo has the overlay's aspect ratio, so viewBox scaling lands on the same pixels. */
export const ratioMatches = (overlay: FaceOverlay, width: number, height: number) =>
  width > 0 && height > 0 && Math.abs((width / height) / (overlay.width / overlay.height) - 1) <= RATIO_TOLERANCE;

export function PhotoOverlay({url, overlay, items, hidden, onToggle, onPreset, onLoadSettled, print = false}: {
  url: string; overlay: FaceOverlay | null | undefined; items: Item[]; hidden: string[];
  onToggle?: (id: string) => void; onPreset?: (preset: OverlayPreset) => void; onLoadSettled?: () => void; print?: boolean;
}) {
  const [fit, setFit] = useState<'unknown' | 'match' | 'mismatch'>('unknown');
  const [focus, setFocus] = useState<string | null>(null);
  const features = overlay?.features ?? [];
  const defaultCount = features.length - presetHidden(overlay, 'default').length;
  const showLines = !!overlay && features.length > 0 && fit !== 'mismatch';
  // Numbers and colours follow the visible lines only, so a few lines never share a colour.
  const visible = features.filter(f => !hidden.includes(f.item_id)), others = features.filter(f => hidden.includes(f.item_id));
  const font = overlay ? Math.max(overlay.width, overlay.height) * .04 : 0;
  const frame: CSSProperties | undefined = overlay && fit !== 'mismatch' ? {aspectRatio: `${overlay.width} / ${overlay.height}`} : undefined;
  const row = (f: OverlayFeature, k: number | null) => {
    const verdict = featureVerdict(f, items), measure = measureText(f);
    const body = <>{k !== null && <span className="swatch" style={{background: overlayColor(k)}} aria-hidden="true">{k + 1}</span>}<span className="legend-name">{f.label}</span>{verdict && <b className={verdict === '판정 보류' ? 'held' : ''}>{verdict}</b>}{measure && <span className="legend-measure">{measure}</span>}</>;
    return <li key={f.item_id} data-item={f.item_id} onMouseEnter={() => setFocus(f.item_id)} onMouseLeave={() => setFocus(null)}>
      {print || !onToggle ? body : <label><input type="checkbox" checked={k !== null} onChange={() => onToggle(f.item_id)} onFocus={() => setFocus(f.item_id)} onBlur={() => setFocus(null)}/>{body}</label>}
    </li>;
  };
  return <figure className={`photo-overlay ${print ? 'print' : ''}`}>
    <div className="photo-frame" style={frame}>
      <img src={url} alt="분석에 사용한 정면 사진" onLoad={e => {
        if (overlay) setFit(ratioMatches(overlay, e.currentTarget.naturalWidth, e.currentTarget.naturalHeight) ? 'match' : 'mismatch');
        requestAnimationFrame(() => onLoadSettled?.());
      }}/>
      {showLines && <svg viewBox={`0 0 ${overlay.width} ${overlay.height}`} preserveAspectRatio="none" role="img" aria-label={`정면 사진 위 측정선 ${visible.length}개`}>
        {visible.map((f, k) => <g key={f.item_id} data-item={f.item_id} className={`overlay-feature ${focus && focus !== f.item_id ? 'dim' : ''}`} style={{'--stroke': overlayColor(k)} as CSSProperties}>
          {f.paths.map((path, j) => <g key={j}><polyline className="halo" points={pathPoints(path)}/><polyline className="line" points={pathPoints(path)}/></g>)}
          <text className="tag" x={f.paths[0][0][0]} y={Math.max(font, f.paths[0][0][1] - font * .5)} fontSize={font} strokeWidth={font * .2}>{k + 1}</text>
        </g>)}
      </svg>}
    </div>
    <figcaption>
      {!overlay || features.length === 0 ? <p className="note">이 사진에는 표시할 측정선이 없습니다.</p>
        : fit === 'mismatch' ? <p className="note">측정선 좌표가 사진 크기와 맞지 않아 선을 표시하지 않았습니다.</p>
        : <>
          {!print && onPreset && <div className="legend-actions" role="group" aria-label="측정선 묶음 선택"><button type="button" className="text-button" onClick={() => onPreset('default')}>기본 {defaultCount}개</button><button type="button" className="text-button" onClick={() => onPreset('all')}>전체 {features.length}개</button><button type="button" className="text-button" onClick={() => onPreset('none')}>모두 숨기기</button></div>}
          {visible.length > 0 ? <ul className={`overlay-legend ${visible.length > 6 ? 'many' : ''}`}>{visible.map((f, k) => row(f, k))}</ul> : <p className="note">선택한 측정선이 없습니다.</p>}
          {!print && others.length > 0 && <details className="legend-more"><summary>다른 측정선 {others.length}개</summary><ul className="overlay-legend">{others.map(f => row(f, null))}</ul></details>}
          <p className="note">{print ? '' : '선은 얼굴 위치점과 부위 경계에서 잰 길이·각도입니다. '}모델이 주목한 위치를 뜻하지 않습니다.</p>
        </>}
    </figcaption>
  </figure>;
}
