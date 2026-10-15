export const DIRECTIONS = ['front', 'left_oblique', 'right_oblique'] as const;
export type Direction = typeof DIRECTIONS[number];
export const DIRECTION_LABELS: Record<Direction, string> = {front: '정면', left_oblique: '좌측 사선', right_oblique: '우측 사선'};
export const STAGES = ['얼굴 정렬', '부위 추출', '형태 분류', '깊이 분석', '문헌 선정'];
export const NOTICE = '전통 문헌 조회 결과이며 개인의 성격·지능·건강·운명을 판정하거나 예측하지 않습니다.';
export type InputMode = 'camera' | 'upload';
export type InputSet = Record<Direction, Blob>;
export interface Item {item_id: string; region: string; status: 'CONFIRMED' | 'HELD'; kind: 'classification' | 'depth'; label_id: string | number | null; label_name: string | null; confidence: number | null; measurement_score?: number | null; measurement: {indicator?: string; value?: number; ratio?: number; unit?: string} | null; basis: string; hold_reasons: string[]}
export interface ComponentScore {item_id: string; label_name: string; kind: string; score: number; score_kind: string}
export type Valence = 'favorable' | 'unfavorable' | 'mixed' | 'neutral';
export const VALENCES: readonly Valence[] = ['favorable', 'unfavorable', 'mixed', 'neutral'];
export const VALENCE_LABELS: Record<Valence, string> = {favorable: '길하게 본 기록', unfavorable: '흉하게 본 기록', mixed: '길흉이 섞인 기록', neutral: '길흉 판단 없음'};
/** Editorial reading supplied by the server; the UI never invents it. */
/** meaning_basis 'representative': the meaning restates the one shown source, never a merged group summary. */
export interface Reading {title: string; meaning: string; valence: Valence; valence_label?: string; topics?: string[]; meaning_basis?: 'representative' | 'group'}
/** Traditional named morphology type matched by the server; may be empty. */
export interface NamedType {id: string; name: string; name_hanja?: string | null; description?: string; meaning?: string; valence?: Valence; valence_label?: string; evidence_item_ids: string[]; source_url?: string; original_text?: string}
export interface Passage {reading?: Reading | null; passage_id: string | number; original_text: string; translation: string; commentary: string; source: {literature_title?: string; title?: string; chapter: string; passage_no?: string; locator?: string; review_status?: string; annotation_status?: string; source_url?: string; url?: string; [key: string]: unknown}; evidence_item_ids: string[]; G?: number | null; rank?: number; score_components?: ComponentScore[]; score_feature_count?: number}
/** Measured landmark/segmentation lines in EXIF-oriented frontal-photo pixels. Never model attention. */
export interface OverlayFeature {item_id: string; label: string; source: 'landmark' | 'segmentation'; paths: [number, number][][]; value?: number | null; unit?: string | null}
export interface FaceOverlay {width: number; height: number; features: OverlayFeature[]}
export interface AnalysisResult {status: 'COMPLETE' | 'PARTIAL' | 'REINPUT_REQUIRED'; items: Item[]; passages: Passage[]; named_types?: NamedType[]; face_overlay?: FaceOverlay | null; notice?: string; literature_status?: string}
export interface Conditions {face: boolean; direction: boolean; size: boolean; brightness: boolean; expression: boolean; message: string; yaw?: number}
export const REGIONS: Record<string, string> = {eye: '눈', brow: '눈썹', nose: '코', mouth: '입', face: '얼굴형', forehead: '이마', glabella: '미간', under_eye: '애교살', cheekbone: '광대', ear: '귀', philtrum: '인중', mole: '점'};
export const FEATURES: Record<string, string> = {eyelid: '눈꺼풀', sclera_show: '흰자 노출', eye_depth: '눈 깊이', brow_shape: '눈썹 모양', brow_thickness: '눈썹 굵기', brow_density: '눈썹 밀도', brow_tail_clarity: '눈썹 꼬리 선명도', nose_tip: '코끝 모양', nostril_show: '콧구멍 노출', alar_fullness: '콧방울 풍만함', lip_bow: '윗입술 활 모양', face_shape: '얼굴 윤곽', chin_shape: '턱끝 모양', cheekbone: '광대 돌출', cheek_fullness: '볼살', nose_projection: '코끝 돌출', nose_bridge: '코뿌리 높이', lip_projection: '입 돌출', chin_projection: '턱 돌출'};
Object.assign(FEATURES, {eye_size:'눈 가로 길이',eye_roundness:'눈 개방 형태',eye_tilt:'눈꼬리 기울기',eye_asymmetry:'좌우 눈 차이',brow_length:'눈썹 길이',brow_eye_gap:'눈썹과 눈 사이 간격',brow_tail:'눈썹 꼬리 방향',nose_width:'콧방울 너비',nose_length:'코 길이',mouth_width:'입 너비',lip_thickness:'입술 두께',lip_balance:'위아래 입술 비율',mouth_corner:'입꼬리 높이',face_length:'얼굴 세로·가로 비율',jaw_width:'아래턱 너비',brow_gap:'미간 너비',eye_spacing:'두 눈 사이 간격',philtrum_length:'인중 길이',lower_face_ratio:'아래 얼굴 비율',under_eye:'애교살 윤곽',philtrum_definition:'인중 골 선명도',forehead_height:'이마 높이',forehead_width:'이마 너비',ear_length:'귀 길이',ear_width:'귀 너비',pigmented_marks:'이마·윗볼 피부 표시'});
/** 0..1 → one decimal with a unit. A value below 1 never rounds up to a perfect 100. */
const scaled = (n: number | null | undefined, unit: string) => {
  if (typeof n !== 'number' || !Number.isFinite(n) || n < 0 || n > 1) return '산출 없음';
  const text = (n * 100).toFixed(1);
  return text === '100.0' && n < 1 ? `99.9${unit} 이상` : `${text}${unit}`;
};
export const points = (n: number | null | undefined) => scaled(n, '점');
export const percent = (n: number | null | undefined) => scaled(n, '%');
export const valenceLabel = (v: {valence?: Valence; valence_label?: string}) => v.valence_label?.trim() || (v.valence ? VALENCE_LABELS[v.valence] : '');
export const safeUrl = (value: unknown) => typeof value === 'string' && /^https?:\/\//i.test(value) ? value : undefined;
export const sourceText = (p: Passage) => {
  const s=p.source,scan=typeof s.pdf_page==='number' ? s.pdf_page : typeof s.scan_page==='string' ? /[/:](\d+)$/.exec(s.scan_page)?.[1] : undefined;
  return [s.representative === true ? '대표 출처' : undefined,s.title || s.literature_title,s.chapter,s.passage_no || s.locator,scan ? `원문 스캔 ${scan}쪽` : undefined].filter(Boolean).join(' · ');
};
export const sourceUrl = (p: Passage) => safeUrl(p.source.url || p.source.source_url);
export const featureName = (i: {item_id: string}) => FEATURES[i.item_id] || i.item_id;
