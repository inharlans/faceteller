import type {AnalysisResult, FaceOverlay} from '../src/types';
// Authored synthetic text only. No K-Face inputs or real personal results.
export function fixture(passages=5,repeat=1):AnalysisResult {
  const base={region:'nose',status:'CONFIRMED' as const,kind:'classification' as const,label_id:'nose_tip:round',label_name:'둥근 편',confidence:.92,measurement_score:null,measurement:null,basis:'합성 검증용 외형 분류 근거',hold_reasons:[]};
  return {status:'PARTIAL',items:[{...base,item_id:'nose_tip'}, {...base,item_id:'nose_bridge',kind:'depth',label_name:'높은 편',label_id:'nose_bridge:high',confidence:null,measurement_score:.864,measurement:{indicator:'nose_bridge',value:.24,unit:'ratio'},basis:'합성 검증용 상대 깊이 비율'}, {...base,item_id:'region_ear',region:'ear',status:'HELD',label_name:null,label_id:null,confidence:null,hold_reasons:['MODEL_NOT_TRAINED']}],passages:Array.from({length:passages},(_,i)=>({passage_id:'fixture-'+i,rank:i+1,G:.9-i*.03,score_feature_count:2,score_components:[{item_id:'nose_tip',label_name:'둥근 편',kind:'classification',score:.92,score_kind:'calibrated_classification_confidence'},{item_id:'nose_bridge',label_name:'높은 편',kind:'depth',score:.864,score_kind:'geometry_membership_not_calibrated_accuracy'}],original_text:`【合成檢證 ${i+1}】`+'面部形態與文獻記錄。此文僅爲畫面測試。'.repeat(repeat),translation:`[합성 검증 ${i+1}] `+'얼굴의 관찰 가능한 형태를 문헌 기록과 대조합니다. 이 문장은 화면 검증을 위해 작성한 예시입니다. '.repeat(repeat),commentary:'인쇄에 포함되면 안 되는 상세해설 FIXTURE_COMMENTARY_ONLY',source:{literature_title:'합성 문헌 UI 검증',chapter:'제1편',passage_no:String(i+1),review_status:'synthetic_fixture_not_reviewed'},evidence_item_ids:['nose_tip','nose_bridge'],reading:i%2?null:{title:`합성 유형 ${i+1}`,meaning:'합성 화면 검증용 풀이 문장입니다.',valence:(['favorable','mixed','unfavorable'] as const)[i%3],valence_label:undefined}})),named_types:passages?[{id:'fixture-type',name:'합성 유형',name_hanja:'合成',description:'화면 검증용 설명',meaning:'화면 검증용 풀이',valence:'neutral' as const,evidence_item_ids:['nose_tip']}]:[],notice:'합성 결과 · 개인의 성격·지능·건강·운명을 판정하지 않습니다.'};
}

type Pt = [number, number];
/** Synthetic 19-feature overlay mirroring facework/release_overlay.py keys, labels, units and path shapes.
 *  Geometry is invented on a 640x480 canvas; no real landmarks or photos. */
export function overlay19(): FaceOverlay {
  const face_width: Pt[] = [[200, 250], [440, 250]], face_height: Pt[] = [[320, 70], [320, 420]];
  const eyes: Pt[][] = [[[255, 200], [295, 198]], [[345, 198], [385, 200]]], eyeH: Pt[][] = [[[275, 190], [275, 208]], [[365, 190], [365, 208]]];
  const brows: Pt[][] = [[[245, 175], [265, 168], [290, 170], [300, 175]], [[340, 175], [350, 170], [375, 168], [395, 175]]];
  const mouth: Pt[] = [[285, 340], [355, 340]], upper: Pt[] = [[320, 325], [320, 336]], lower: Pt[] = [[320, 344], [320, 356]];
  const f = (item_id: string, label: string, paths: Pt[][], value?: number, unit = 'ratio') =>
    ({item_id, label, source: 'landmark' as const, paths, ...(value === undefined ? {} : {value, unit})});
  return {width: 640, height: 480, features: [
    f('eye_size', '눈 가로 길이', [...eyes, face_width], .1742),
    f('eye_roundness', '눈의 가로·세로', [...eyes, ...eyeH], .3311),
    f('eye_tilt', '눈꼬리 기울기', eyes, 3.418, 'degrees'),
    f('eye_asymmetry', '좌우 눈의 가로·세로', [...eyes, ...eyeH], .0412, 'log_ratio'),
    f('brow_length', '눈썹 기준선', [...brows, ...eyes], 1.204),
    f('brow_tail', '눈썹 앞머리와 꼬리', brows.map(b => [b[0], b[b.length - 1]]), -2.07, 'degrees'),
    f('brow_eye_gap', '눈과 눈썹 사이', [...eyes, [[275, 190], [275, 172]], [[365, 190], [365, 172]]], .412),
    f('brow_gap', '눈썹 사이 간격', [[[300, 175], [340, 175]], face_width], .1667),
    f('nose_width', '코 너비', [[[300, 290], [340, 290]], face_width], .1667),
    f('nose_length', '코 길이', [[[320, 195], [320, 300]], face_height], .3),
    f('mouth_width', '입 너비', [mouth, face_width], .2917),
    f('lip_thickness', '윗입술·아랫입술 두께', [upper, lower, mouth], .3286),
    f('lip_balance', '윗입술과 아랫입술 비율', [upper, lower], .9167),
    f('mouth_corner', '입꼬리 위치', [mouth, [[285, 338], [285, 340]], [[355, 337], [355, 340]]], -.0286),
    f('face_length', '얼굴 세로·가로', [face_height, face_width], 1.458),
    f('jaw_width', '턱 너비', [[[230, 370], [410, 370]], face_width], .75),
    f('eye_spacing', '눈 사이 간격', [[[295, 198], [345, 198]], ...eyes], 1.25),
    f('philtrum_length', '인중 길이', [[[320, 305], [320, 325]], face_height], .0571),
    f('lower_face_ratio', '얼굴 가운데와 아래 길이', [[[320, 195], [320, 305]], [[320, 305], [320, 420]]], 1.045),
  ]};
}
