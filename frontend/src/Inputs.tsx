import {useEffect, useRef, useState} from 'react';
import {CaptureState, captureDirectionHint, captureDirectionLabel} from './capture';
import {createVision, decodeFile, type VisionPort} from './vision';
import {DIRECTIONS, type Conditions, type Direction, type InputSet} from './types';

// Upload photos are shown as taken; the camera self-view is mirrored. Labels follow what the user sees.
const uploadLabel = (d: Direction) => d === 'front' ? '정면 사진' : `${captureDirectionLabel(d, 'upload')}을 향한 사진`;

type Slot = {status: 'empty' | 'checking' | 'valid' | 'invalid'; file?: File; url?: string; message?: string};
const emptySlots = (): Record<Direction,Slot> => ({front:{status:'empty'},left_oblique:{status:'empty'},right_oblique:{status:'empty'}});
export function Upload({onReady, onBack, onCamera}: {onReady: (set: InputSet) => void; onBack: () => void; onCamera: () => void}) {
  const [slots,setSlots] = useState(emptySlots);
  const current = useRef(slots), versions = useRef({front:0,left_oblique:0,right_oblique:0}), alive = useRef(true), submitted = useRef(false), vision = useRef<VisionPort | null>(null);
  const ready = useRef(onReady); ready.current = onReady;
  useEffect(() => {alive.current = true; return () => {alive.current = false; vision.current?.close(); vision.current = null; for (const d of DIRECTIONS) {versions.current[d]++; if (current.current[d].url) URL.revokeObjectURL(current.current[d].url!);} current.current = emptySlots();};}, []);
  function update(direction: Direction, next: Slot) {
    const old = current.current[direction]; if (old.url && old.url !== next.url) URL.revokeObjectURL(old.url);
    const state = {...current.current,[direction]:next}; current.current = state; setSlots(state);
    if (!submitted.current && DIRECTIONS.every(d => state[d].status === 'valid')) {
      submitted.current = true;
      ready.current({front:state.front.file!,left_oblique:state.left_oblique.file!,right_oblique:state.right_oblique.file!});
    }
  }
  async function select(direction: Direction, file?: File) {
    const version = ++versions.current[direction];
    update(direction, {status:file ? 'checking' : 'empty'});
    if (!file) return;
    let bitmap: ImageBitmap | undefined;
    try {
      bitmap = await decodeFile(file);
      if (!alive.current || versions.current[direction] !== version) {bitmap.close(); return;}
      vision.current ||= createVision();
      const inspector = vision.current;
      const check = await inspector.evaluate(bitmap, direction, 'upload').catch(error => {
        if (vision.current === inspector) {inspector.close(); vision.current = null;}
        throw error;
      }); bitmap = undefined;
      if (!alive.current || versions.current[direction] !== version) return;
      if (![check.face,check.direction,check.size,check.brightness,check.expression].every(Boolean)) throw new Error(check.message);
      update(direction, {status:'valid',file,url:URL.createObjectURL(file),message:'얼굴·방향 검사 완료'});
    } catch (e) {
      bitmap?.close();
      if (alive.current && versions.current[direction] === version) update(direction,{status:'invalid',message:e instanceof Error ? e.message : '사진을 다시 선택해 주세요.'});
    }
  }
  return <section aria-labelledby="upload-title"><div className="section-heading center"><p className="eyebrow">사진 업로드</p><h1 id="upload-title">같은 사람의 사진 3장을 준비해 주세요</h1><p className="lead">정면 한 장, 고개를 왼쪽·오른쪽으로 돌린 사진 각 한 장<br/>세 장이 모두 확인되면 바로 분석을 시작합니다.</p></div>
    <div className="upload-grid">{DIRECTIONS.map((d,i) => <div key={d} className={`upload-slot ${slots[d].status}`} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault(); if (e.dataTransfer.files.length === 1) void select(d,e.dataTransfer.files[0]); else {versions.current[d]++; update(d,{status:'invalid',message:'방향마다 사진 한 장을 선택해 주세요.'});}}}>
      <h2><span className="step-number">0{i+1}</span>{uploadLabel(d)}</h2>
      <label className="drop-area" htmlFor={`file-${d}`}>{slots[d].url ? <img src={slots[d].url} alt={`${uploadLabel(d)} 임시 미리보기`}/> : <><span className="direction-icon" aria-hidden="true">{i===0 ? '◎' : i===1 ? '↶' : '↷'}</span><span>사진을 놓거나 파일 선택</span></>}<input id={`file-${d}`} type="file" accept="image/jpeg,image/png,image/webp" onChange={e => {const file = e.target.files?.[0]; e.target.value = ''; void select(d,file);}}/></label>
      <p className="slot-status" role="status">{slots[d].status==='empty' ? '미선택' : slots[d].status==='checking' ? '사진 검사 중…' : slots[d].status==='valid' ? '✓ '+slots[d].message : '재선택 필요 · '+slots[d].message}</p>
      {slots[d].status!=='empty' && <button className="text-button" onClick={() => void select(d)} aria-label={`${uploadLabel(d)} 지우기`}>사진 지우기</button>}
    </div>)}</div>
    <p className="note center">JPG · PNG · WebP / 장당 최대 10 MiB · 가로·세로 160px 이상<br/>옆 사진은 사진 속 코가 왼쪽 또는 오른쪽을 향한 모습입니다.</p>
    <div className="actions center"><button className="button ghost" onClick={onBack}>처음으로</button><button className="button ghost" onClick={onCamera}>카메라로 촬영</button></div>
  </section>;
}

function snapshotBlob(bitmap: ImageBitmap): Promise<Blob> {
  const canvas = document.createElement('canvas'); canvas.width = bitmap.width; canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d'); if (!ctx) throw new Error('사진을 촬영할 수 없습니다.');
  ctx.drawImage(bitmap,0,0);
  return new Promise((resolve,reject) => canvas.toBlob(blob => {canvas.width=0; canvas.height=0; if(blob) resolve(blob); else reject(new Error('촬영한 사진을 읽을 수 없습니다.'));},'image/jpeg',.92));
}
export function Camera({onReady,onError,onBack}: {onReady: (inputs: InputSet) => void; onError: (message: string) => void; onBack: () => void}) {
  const video = useRef<HTMLVideoElement>(null);
  const callbacks = useRef({onReady,onError}); callbacks.current = {onReady,onError};
  const [step,setStep] = useState(0), [gauge,setGauge] = useState(0), [message,setMessage] = useState('카메라 권한을 요청합니다…'), [conditions,setConditions] = useState<Conditions | null>(null);
  useEffect(() => {
    let active = true, stream: MediaStream | undefined, vision: VisionPort | undefined, timer: ReturnType<typeof setTimeout> | undefined, lastVideoTime = -1;
    const machine = new CaptureState(), photos: Partial<InputSet> = {};
    const stop = () => {active=false; machine.reset(); if(timer) clearTimeout(timer); stream?.getTracks().forEach(t => t.stop()); vision?.close(); if(video.current) video.current.srcObject=null; for(const d of DIRECTIONS) delete photos[d];};
    const fail = (error: unknown) => {if(!active) return; stop(); callbacks.current.onError(error instanceof DOMException && error.name==='NotAllowedError' ? '카메라 권한이 거부되었습니다. 사진 3장을 업로드하거나 브라우저 권한을 허용한 뒤 다시 촬영해 주세요.' : error instanceof Error ? error.message : '카메라를 사용할 수 없습니다. 사진 3장을 업로드해 주세요.');};
    const hidden = () => {if(document.hidden) fail(new Error('화면이 숨겨져 촬영을 중단하고 임시 사진을 정리했습니다.'));};
    document.addEventListener('visibilitychange',hidden);
    const pagehide = () => stop(); window.addEventListener('pagehide',pagehide);
    const tick = async () => {
      if(!active || !video.current) return;
      const v=video.current;
      if(v.readyState<2 || v.currentTime===lastVideoTime) {timer=setTimeout(tick,80);return;}
      const mediaTime=v.currentTime; lastVideoTime=mediaTime;
      const token=machine.token(), capturedAt=performance.now();
      let frame: ImageBitmap | undefined;
      try {
        frame=await createImageBitmap(v);
        if(!active) return;
        const copy=await createImageBitmap(frame);
        const check=await vision!.evaluate(copy,token.direction!);
        if(!active || !machine.matches(token)) return;
        const next=machine.update(check,capturedAt,performance.now(),mediaTime,token);
        setGauge(next.progress); setConditions(check); setMessage(next.ignored ? '영상 상태를 확인하고 있습니다. 잠시 유지해 주세요.' : check.message);
        if(next.capture) {
          const blob=await snapshotBlob(frame); // Same unmirrored frame that passed inspection.
          if(!active || !machine.commit(token)) return;
          photos[token.direction!]=blob; setStep(s=>s+1); setGauge(0); setConditions(null);
          if(machine.submitOnce()) {const complete={...photos} as InputSet; stop(); callbacks.current.onReady(complete); return;}
        }
      } catch(error) {fail(error);}
      finally {frame?.close();}
      if(active) timer=setTimeout(tick,80);
    };
    void (async () => {
      try {
        if(!navigator.mediaDevices?.getUserMedia) throw new Error('이 브라우저에서 카메라를 사용할 수 없습니다. HTTPS 또는 localhost에서 열어 주세요.');
        const opened=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:960},height:{ideal:720}},audio:false});
        if(!active) {opened.getTracks().forEach(t=>t.stop());return;}
        stream=opened; stream.getTracks().forEach(t=>t.addEventListener('ended',()=>fail(new Error('카메라 연결이 끊어졌습니다. 다시 촬영해 주세요.'))));
        const v=video.current!; v.srcObject=stream; await v.play(); if(!active) return;
        vision=createVision(); setMessage('얼굴 검사기를 준비하고 있습니다…'); timer=setTimeout(tick,0);
      } catch(error) {fail(error);}
    })();
    return () => {document.removeEventListener('visibilitychange',hidden);window.removeEventListener('pagehide',pagehide);stop();};
  },[]);
  const direction=DIRECTIONS[Math.min(step,2)], label=captureDirectionLabel(direction,'camera');
  return <section aria-labelledby="camera-title" data-capture-direction={direction}><div className="section-heading center"><p className="eyebrow">자동 촬영 {Math.min(step,2)+1}/3</p><h1 id="camera-title">{direction==='front'?'정면 촬영':`${label}으로 고개 돌리기`}</h1><p className="lead direction-hint">{captureDirectionHint(direction,'camera')}</p></div>
    <ol className="capture-steps">{DIRECTIONS.map((d,i)=><li className={step>i?'done':step===i?'active':''} key={d}>{step>i?'✓':i+1} {captureDirectionLabel(d,'camera')}</li>)}</ol>
    <div className="camera-viewport"><video className="mirrored" ref={video} autoPlay playsInline muted aria-label="거울처럼 보이는 카메라 미리보기"/><span className="camera-caption">거울처럼 보이는 미리보기</span><svg className="camera-guide" viewBox="0 0 200 260" aria-hidden="true"><ellipse cx="100" cy="130" rx="78" ry="108" className="ring-background"/><ellipse cx="100" cy="130" rx="78" ry="108" pathLength="100" className="ring-progress" strokeDasharray={`${gauge*100} 100`}/></svg>{direction!=='front' && <span className={`turn-arrow ${label==='화면 왼쪽'?'left':'right'}`} aria-hidden="true">{label==='화면 왼쪽'?'←':'→'}</span>}</div>
    <p className="capture-guidance" role="status">{message}</p><div className="gauge" role="progressbar" aria-label="촬영 조건 유지" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(gauge*100)}>{Math.round(gauge*100)}%</div>
    <ul className="condition-list">{(['face','direction','size','brightness','expression'] as const).map((k,i)=><li key={k} className={conditions?.[k]?'good':''}>{conditions?.[k]?'✓':'○'} {['얼굴 1명','방향','거리','밝기','무표정·눈 뜨기'][i]}</li>)}</ul>
    <p className="note center">조건에서 벗어나면 게이지가 처음부터 다시 찹니다. 세 방향을 다 찍으면 바로 분석합니다.</p><div className="actions center"><button className="button ghost" onClick={onBack}>촬영 취소</button><button className="button ghost" onClick={()=>onError('촬영을 종료했습니다. 사진 3장을 업로드해 주세요.')}>업로드로 전환</button></div>
  </section>;
}
