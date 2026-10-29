import {useCallback, useEffect, useRef, useState} from 'react';
import {AnalysisError, submitAnalysis} from './api';
import {captureDirectionLabel} from './capture';
import {Camera, Upload} from './Inputs';
import {presetHidden} from './PhotoOverlay';
import {Printable, Results, type PhotoView} from './Results';
import {DIRECTIONS, NOTICE, STAGES, type AnalysisResult, type InputMode, type InputSet} from './types';
type Page = 'home'|'camera'|'camera-retry'|'upload'|'review'|'progress'|'results'|'preview';
// Server analysis is paused for now: captured photos go to an on-screen check instead.
const ANALYSIS_ENABLED = false;
export default function App() {
  const [page,setPage]=useState<Page>('home'),[notice,setNotice]=useState(''),[stage,setStage]=useState('사진을 전달하고 있습니다…'),[result,setResult]=useState<AnalysisResult|null>(null),[dark,setDark]=useState(false);
  const generation=useRef(0),request=useRef<AbortController|null>(null),mode=useRef<InputMode>('upload'),heading=useRef<HTMLDivElement>(null);
  // Only the frontal Blob outlives the request, in memory, for the opt-in photo view. Obliques are dropped.
  const [front,setFront]=useState<Blob|null>(null),[showPhoto,setShowPhoto]=useState(false),[hidden,setHidden]=useState<string[]>([]),[photoUrl,setPhotoUrl]=useState<string|null>(null);
  const [captured,setCaptured]=useState<{inputs:InputSet;mode:InputMode}|null>(null),[capturedUrls,setCapturedUrls]=useState<Partial<Record<typeof DIRECTIONS[number],string>>>({});
  useEffect(()=>{if(!captured) return; const urls=Object.fromEntries(DIRECTIONS.map(d=>[d,URL.createObjectURL(captured.inputs[d])]));setCapturedUrls(urls);return()=>{Object.values(urls).forEach(URL.revokeObjectURL);setCapturedUrls({});};},[captured]);
  const clearPhoto=useCallback(()=>{setFront(null);setShowPhoto(false);setHidden([]);},[]);
  useEffect(()=>{if(!front||!showPhoto) return; const url=URL.createObjectURL(front);setPhotoUrl(url);return()=>{URL.revokeObjectURL(url);setPhotoUrl(null);};},[front,showPhoto]);
  const reset=useCallback(()=>{generation.current++; request.current?.abort();request.current=null;setResult(null);setCaptured(null);clearPhoto();setNotice('');setPage('home');},[clearPhoto]);
  useEffect(()=>{const hide=()=>reset(); window.addEventListener('pagehide',hide);return()=>{generation.current++;request.current?.abort();window.removeEventListener('pagehide',hide);};},[reset]);
  useEffect(()=>{heading.current?.focus();},[page]);
  const start=useCallback(async(inputs:InputSet,inputMode:InputMode)=>{
    if(request.current) return;
    const epoch=++generation.current, controller=new AbortController();request.current=controller;mode.current=inputMode;
    setResult(null);clearPhoto();setNotice('');setStage('사진을 전달하고 있습니다…');setPage('progress');
    try {
      const answer=await submitAnalysis(inputs,inputMode,controller.signal,s=>{if(epoch===generation.current)setStage(s);});
      if(epoch!==generation.current)return;
      if(answer.status==='REINPUT_REQUIRED') {setNotice('확정 가능한 항목이 없습니다. 임시 사진을 정리했습니다. 촬영 또는 업로드를 다시 진행해 주세요.');setPage(inputMode==='camera'?'camera-retry':'upload');}
      else {setResult(answer);setFront(inputs.front);setHidden(presetHidden(answer.face_overlay));setPage('results');}
    } catch(error) {
      if(epoch!==generation.current)return;
      const prefix=error instanceof AnalysisError && error.direction && DIRECTIONS.includes(error.direction) ? (error.direction==='front' ? '정면 사진: ' : `${captureDirectionLabel(error.direction,inputMode)}을 향한 사진: `) : '';
      setNotice(prefix+(error instanceof Error ? error.message : '분석 요청에 실패했습니다.')+' 임시 사진을 정리했습니다. 다시 입력해 주세요.');
      setPage(inputMode==='camera'?'camera-retry':'upload');
    } finally {if(epoch===generation.current)request.current=null;}
  },[clearPhoto]);
  const startCamera=useCallback(()=>{setCaptured(null);setNotice('');setPage('camera');mode.current='camera';},[]);
  const cameraError=useCallback((message:string)=>{setCaptured(null);setNotice(message);setPage('upload');mode.current='upload';},[]);
  const ready=useCallback((inputs:InputSet,inputMode:InputMode)=>{if(ANALYSIS_ENABLED){void start(inputs,inputMode);return;}mode.current=inputMode;setCaptured({inputs,mode:inputMode});setPage('review');},[start]);
  const cameraReady=useCallback((inputs:InputSet)=>ready(inputs,'camera'),[ready]);
  const uploadReady=useCallback((inputs:InputSet)=>ready(inputs,'upload'),[ready]);
  const cancel=()=>{generation.current++;request.current?.abort();request.current=null;clearPhoto();setNotice('요청을 취소하고 임시 사진을 정리했습니다.');setPage(mode.current==='camera'?'camera-retry':'upload');};
  const photo:PhotoView={available:!!front,show:showPhoto,url:photoUrl,hidden,setShow:setShowPhoto,
    toggle:id=>setHidden(h=>h.includes(id)?h.filter(v=>v!==id):[...h,id]),preset:p=>setHidden(presetHidden(result?.face_overlay,p))};
  const step=page==='home'?0:page==='camera'||page==='camera-retry'||page==='upload'||page==='review'?1:page==='progress'?2:page==='results'?3:4;
  const fixture=import.meta.env.MODE==='test' && window.__FACETELLER_TEST__?.fixture;
  return <div className={`app ${dark?'dark':''} ${page==='preview'?'preview-page':''}`}>
    <div className="wrap screen-only"><header className="doc-head"><span className="wordmark">페이스텔러</span><span className="meta">얼굴 형태 분류와 전통 문헌 조회</span></header>
      <ol className="steps" aria-label="이용 단계">{['시작','사진 입력','분석','결과','결과지'].map((s,i)=><li key={s} aria-current={step===i?'step':undefined}><span>{i+1}</span>{s}</li>)}</ol>
      <div className="device"><div className="appbar"><button className="brand" onClick={reset} aria-label="페이스텔러 처음으로">페이스텔러 <span>FaceTeller</span></button><button className="theme-toggle" onClick={()=>setDark(v=>!v)} aria-label={dark?'밝은 테마로 전환':'어두운 테마로 전환'}>{dark?'☀':'◐'}</button></div>
        <main className="stage" ref={heading} tabIndex={-1}>{fixture && <p className="fixture-banner">합성 UI 검증 · 실제 얼굴 분석 결과 아님</p>}{notice && <div className="alert" role="alert">{notice}</div>}
          {page==='home' && <section className="home center"><p className="eyebrow">FACE · FORM · LITERATURE</p><h1>얼굴 형태를 분류하고<br/>전통 문헌의 기록을 함께 봅니다</h1><p className="lead">눈썹·눈·코·입 등의 관찰 가능한 형태와<br/>그 형태를 기술한 문헌 조항을 근거와 함께 확인하세요.</p><div className="home-motif" aria-hidden="true"><span>形</span><i/><span>文</span></div><div className="stack"><button className="button" onClick={startCamera}>웹캠으로 촬영하기 <span aria-hidden="true">→</span></button><button className="button ghost" onClick={()=>{mode.current='upload';setNotice('');setPage('upload');}}>이미지 파일 업로드</button></div><p className="privacy">정면과 양옆으로 고개를 돌린 사진, 모두 3장이 필요합니다.<br/>사진은 서버에 저장하지 않습니다. 정면 사진 한 장만 결과를 보는 동안 이 브라우저 메모리에 남고,<br/>처음으로 돌아가거나 창을 닫으면 지워집니다.</p><p className="note">{NOTICE}</p></section>}
          {page==='camera' && <Camera onReady={cameraReady} onError={cameraError} onBack={reset}/>}
          {page==='camera-retry' && <section className="center"><h1>사진을 다시 촬영해 주세요</h1><p className="lead">이전 사진은 정리했습니다. 정면부터 세 방향을 다시 촬영합니다.</p><div className="actions center"><button className="button" onClick={startCamera}>다시 촬영</button><button className="button ghost" onClick={()=>cameraError('사진 3장을 업로드해 주세요.')}>업로드로 전환</button></div></section>}
          {page==='review' && captured && <section className="center" aria-labelledby="review-title"><p className="eyebrow">촬영 확인</p><h1 id="review-title">사진이 제대로 나왔는지 확인해 주세요</h1><p className="lead">실제로 분석에 쓰일 사진 그대로입니다. 지금은 서버 분석을 하지 않습니다.</p><div className="review-grid">{DIRECTIONS.map(d=><figure key={d} className="review-shot">{capturedUrls[d] && <img src={capturedUrls[d]} alt={`${d==='front'?'정면':captureDirectionLabel(d,'upload')+'을 향한'} 사진`}/>}<figcaption>{d==='front'?'정면':`${captureDirectionLabel(d,'upload')}을 향한 사진`}</figcaption></figure>)}</div><div className="actions center"><button className="button" onClick={captured.mode==='camera'?startCamera:()=>{setCaptured(null);setPage('upload');}}>{captured.mode==='camera'?'다시 촬영':'다시 업로드'}</button><button className="button ghost" onClick={reset}>처음으로</button></div></section>}
          {page==='upload' && <Upload onReady={uploadReady} onBack={reset} onCamera={startCamera}/>}
          {page==='progress' && <section className="progress-screen center" aria-labelledby="progress-title"><div className="spinner" aria-hidden="true"/><h1 id="progress-title">얼굴 형태와 문헌을 확인하고 있습니다</h1><p role="status" aria-live="polite">{stage}</p><ol className="progress-list">{STAGES.map((s,i)=><li key={s} className={STAGES.indexOf(stage)>i?'done':stage===s?'active':''}><span>{STAGES.indexOf(stage)>i?'✓':i+1}</span>{s}</li>)}</ol><button className="button ghost" onClick={cancel}>분석 취소</button></section>}
          {result && page==='results' && <Results result={result} photo={photo} onPreview={()=>setPage('preview')} onReset={reset}/>}
          {result && page==='preview' && <div className="preview-heading"><h1>문헌 기록을 한 장에</h1><p className="lead">원문·풀이·출처와 확인한 생김새를 담습니다.</p></div>}
        </main></div><footer className="app-footer">FaceTeller <span>관찰 가능한 형태와 역사적 문헌의 연결</span></footer></div>
    {result && <Printable result={result} photo={photo} preview={page==='preview'} onBack={()=>setPage('results')}/>}
  </div>;
}
