import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AnalysisError, submitAnalysis} from './api';
import {captureDirectionLabel} from './capture';
import {Camera, Upload} from './Inputs';
import {presetHidden} from './PhotoOverlay';
import {Printable, Results, type PhotoView} from './Results';
import {FINISH_MS, PORTRAIT_TIMEOUT_MS, PROGRESS_POLL_MS, PROGRESS_WAIT_MS, STAGE_STEP_MS, newRequestId, portraitAnchors, portraitProgress, requestPortrait, stageStepMs, type DrawProgress, type Portrait} from './portrait';
import {DIRECTIONS, NOTICE, STAGES, type AnalysisResult, type InputMode, type InputSet} from './types';
import {ProgressScreen} from './ProgressScreen';
import {LOG_DEFAULTS, estimateFraction, logPlan, runFacts, stageView} from './processSteps';
type Page = 'home'|'camera'|'camera-retry'|'upload'|'review'|'progress'|'results'|'preview';
// Server analysis is ON for the lab trial (2026-11-16): photos go to POST /api/v1/analysis. Set to false to show the
// captured photos for an on-screen check instead (the 'review' page), without sending them.
const ANALYSIS_ENABLED: boolean = true;
export default function App() {
  const [page,setPage]=useState<Page>('home'),[notice,setNotice]=useState(''),[stage,setStage]=useState('사진을 전달하고 있습니다…'),[result,setResult]=useState<AnalysisResult|null>(null),[dark,setDark]=useState(false);
  const generation=useRef(0),request=useRef<AbortController|null>(null),mode=useRef<InputMode>('upload'),heading=useRef<HTMLDivElement>(null);
  // Only the frontal Blob outlives the request, in memory, for the opt-in photo view. Obliques are dropped.
  const [front,setFront]=useState<Blob|null>(null),[showPhoto,setShowPhoto]=useState(false),[hidden,setHidden]=useState<string[]>([]),[photoUrl,setPhotoUrl]=useState<string|null>(null);
  const [captured,setCaptured]=useState<{inputs:InputSet;mode:InputMode}|null>(null),[capturedUrls,setCapturedUrls]=useState<Partial<Record<typeof DIRECTIONS[number],string>>>({});
  useEffect(()=>{if(!captured) return; const urls=Object.fromEntries(DIRECTIONS.map(d=>[d,URL.createObjectURL(captured.inputs[d])]));setCapturedUrls(urls);return()=>{Object.values(urls).forEach(URL.revokeObjectURL);setCapturedUrls({});};},[captured]);
  // Character drawing for the report sheet: asked for together with the analysis, shown when it arrives, skipped when it fails.
  const [portrait,setPortrait]=useState<Portrait|null>(null),[portraitPending,setPortraitPending]=useState(false),portraitReq=useRef<AbortController|null>(null);
  useEffect(()=>{if(!portrait) return; return()=>URL.revokeObjectURL(portrait.url);},[portrait]);
  // Pacing of the progress screen while the drawing is being made (portrait.ts).
  const [analysisDone,setAnalysisDone]=useState(false),[now,setNow]=useState(0),[stepMs,setStepMs]=useState(STAGE_STEP_MS),startedAt=useRef(0),finishAt=useRef<number|null>(null),finishBase=useRef<number[]>([]);
  // Real progress of the drawing (image server /progress/<id>): the last answer, when it came, and how long one step takes.
  const [prog,setProg]=useState<{p:DrawProgress;at:number}|null>(null),portraitId=useRef(''),stepTimes=useRef<{step:number;at:number}[]>([]),drawMax=useRef(0);
  const stopPortrait=useCallback(()=>{portraitReq.current?.abort();portraitReq.current=null;setPortrait(null);setPortraitPending(false);},[]);
  const clearPhoto=useCallback(()=>{setFront(null);setShowPhoto(false);setHidden([]);},[]);
  useEffect(()=>{if(!front||!showPhoto) return; const url=URL.createObjectURL(front);setPhotoUrl(url);return()=>{URL.revokeObjectURL(url);setPhotoUrl(null);};},[front,showPhoto]);
  const reset=useCallback(()=>{generation.current++; request.current?.abort();request.current=null;stopPortrait();setResult(null);setCaptured(null);clearPhoto();setNotice('');setPage('home');},[clearPhoto,stopPortrait]);
  useEffect(()=>{const hide=()=>reset(); window.addEventListener('pagehide',hide);return()=>{generation.current++;request.current?.abort();window.removeEventListener('pagehide',hide);};},[reset]);
  useEffect(()=>{heading.current?.focus();},[page]);
  const start=useCallback(async(inputs:InputSet,inputMode:InputMode)=>{
    if(request.current) return;
    const epoch=++generation.current, controller=new AbortController();request.current=controller;mode.current=inputMode;
    startedAt.current=Date.now();finishAt.current=null;setProg(null);stepTimes.current=[];drawMax.current=0;portraitId.current=newRequestId();setNow(startedAt.current);setAnalysisDone(false);setStepMs(STAGE_STEP_MS);void stageStepMs(STAGES.length).then(ms=>{if(epoch===generation.current)setStepMs(ms);});
    setResult(null);clearPhoto();stopPortrait();setNotice('');setStage('사진을 전달하고 있습니다…');setPage('progress');
    const drawing=new AbortController(), timer=setTimeout(()=>drawing.abort(),PORTRAIT_TIMEOUT_MS);portraitReq.current=drawing;setPortraitPending(true);
    void requestPortrait(inputs.front,drawing.signal,portraitId.current).then(async png=>{const anchors=await portraitAnchors(png); if(epoch!==generation.current||drawing.signal.aborted) return; setPortrait({url:URL.createObjectURL(png),anchors});})
      .catch(()=>{}).finally(()=>{clearTimeout(timer); if(epoch===generation.current){setPortraitPending(false); if(portraitReq.current===drawing) portraitReq.current=null;}});
    try {
      const answer=await submitAnalysis(inputs,inputMode,controller.signal,s=>{if(epoch===generation.current)setStage(s);});
      if(epoch!==generation.current)return;
      if(answer.status==='REINPUT_REQUIRED') {stopPortrait();setNotice('확정 가능한 항목이 없습니다. 임시 사진을 정리했습니다. 촬영 또는 업로드를 다시 진행해 주세요.');setPage(inputMode==='camera'?'camera-retry':'upload');}
      else {setResult(answer);setFront(inputs.front);setHidden(presetHidden(answer.face_overlay));setAnalysisDone(true);}   // the effect below moves on to the results
    } catch(error) {
      if(epoch!==generation.current)return;
      stopPortrait();
      const prefix=error instanceof AnalysisError && error.direction && DIRECTIONS.includes(error.direction) ? (error.direction==='front' ? '정면 사진: ' : `${captureDirectionLabel(error.direction,inputMode)}을 향한 사진: `) : '';
      setNotice(prefix+(error instanceof Error ? error.message : '분석 요청에 실패했습니다.')+' 임시 사진을 정리했습니다. 다시 입력해 주세요.');
      setPage(inputMode==='camera'?'camera-retry':'upload');
    } finally {if(epoch===generation.current)request.current=null;}
  },[clearPhoto,stopPortrait]);
  const startCamera=useCallback(()=>{setCaptured(null);setNotice('');setPage('camera');mode.current='camera';},[]);
  const cameraError=useCallback((message:string)=>{setCaptured(null);setNotice(message);setPage('upload');mode.current='upload';},[]);
  const ready=useCallback((inputs:InputSet,inputMode:InputMode)=>{if(ANALYSIS_ENABLED){void start(inputs,inputMode);return;}mode.current=inputMode;setCaptured({inputs,mode:inputMode});setPage('review');},[start]);
  const cameraReady=useCallback((inputs:InputSet)=>ready(inputs,'camera'),[ready]);
  const uploadReady=useCallback((inputs:InputSet)=>ready(inputs,'upload'),[ready]);
  const cancel=()=>{generation.current++;request.current?.abort();request.current=null;stopPortrait();setResult(null);setAnalysisDone(false);clearPhoto();setNotice('요청을 취소하고 임시 사진을 정리했습니다.');setPage(mode.current==='camera'?'camera-retry':'upload');};
  // While on the progress screen: tick the clock; show the results once the analysis is in and the drawing is in,
  // has failed, or has taken longer than PROGRESS_WAIT_MS.
  useEffect(()=>{if(page!=='progress') return; const t=setInterval(()=>setNow(Date.now()),80); return()=>clearInterval(t);},[page]);
  // Ask the image server how far our drawing is, every PROGRESS_POLL_MS while it is being made.
  useEffect(()=>{if(page!=='progress'||!portraitPending||!portraitId.current) return; const id=portraitId.current, ctl=new AbortController();
    const t=setInterval(()=>{void portraitProgress(id,ctl.signal).then(p=>{if(!p||id!==portraitId.current) return; const at=Date.now();
      if(p.state==='drawing'){const st=stepTimes.current; if(!st.length||st[st.length-1].step!==p.step) st.push({step:p.step,at});}
      setProg(prev=>prev&&prev.p.state===p.state&&prev.p.step===p.step?prev:{p,at});});},PROGRESS_POLL_MS);
    return()=>{clearInterval(t);ctl.abort();};},[page,portraitPending]);
  // How much of the drawing is done (0..1), moving smoothly between the answers at the measured step rate; null when the
  // server gives no progress (then the stages follow the time estimate as before).
  const stepDur=(()=>{const st=stepTimes.current; return st.length>=2?(st[st.length-1].at-st[0].at)/Math.max(1,st[st.length-1].step-st[0].step):0;})();
  const drawF=!prog||prog.p.state==='unknown'?null:prog.p.state==='done'?1:prog.p.state==='queued'||!prog.p.steps?0
    :Math.min(.99,(prog.p.step+(stepDur?Math.min(.95,(now-prog.at)/stepDur):0))/prog.p.steps);
  // never let the shown fraction go back (a late or repeated answer must not pull the gauges down)
  const drawShown=drawF===null?null:Math.max(drawMax.current,drawF); if(drawShown!==null) drawMax.current=drawShown;
  const real=drawShown!==null&&portraitPending, queued=real&&prog?.p.state==='queued';
  const serverStage=analysisDone?STAGES.length:STAGES.indexOf(stage);
  // The written account (processSteps.ts) and the gauges share one clock: the part of the wait that has passed, from the
  // image server's own progress or, without it, the time estimate. Every line gets its share of the wait, so each stage
  // takes as long as it has to tell. Never ahead of the stage the server is at, nor of the answer the figures come from.
  const facts=useMemo(()=>runFacts(result?.research_report),[result]), plan=useMemo(()=>logPlan(STAGES,facts,LOG_DEFAULTS),[facts]);
  const cap=analysisDone?1:serverStage<0?-1:Math.min(plan.bounds[serverStage+1],plan.factsFrom)-1e-6;
  const fraction=Math.min(cap,!portraitPending?1:real?drawShown!:estimateFraction(now-startedAt.current,stepMs*STAGES.length));
  const view=stageView(plan.bounds,fraction), paced=!portraitPending&&analysisDone?STAGES.length:view.paced;
  const shownStage=queued?'그림 순서를 기다리는 중…':paced>=0&&paced<STAGES.length?STAGES[paced]:stage;
  const live=paced>=STAGES.length?STAGES.map(()=>100):view.percents;
  // When everything is in, the gauges first run up to 100 over FINISH_MS, then the results are shown.
  useEffect(()=>{if(page!=='progress' || !analysisDone || !result || (portraitPending && now-startedAt.current<PROGRESS_WAIT_MS)) return;
    if(finishAt.current===null){finishAt.current=now;finishBase.current=live;return;}
    if(now-finishAt.current>=FINISH_MS){finishAt.current=null;setAnalysisDone(false);setPage('results');}},[page,analysisDone,result,portraitPending,now]);   // eslint-disable-line react-hooks/exhaustive-deps
  const percent=(i:number)=>{const f=finishAt.current; if(f===null) return live[i]; const base=finishBase.current[i]??live[i]; return Math.round(base+(100-base)*Math.min(1,(now-f)/FINISH_MS));};
  const photo:PhotoView={available:!!front,show:showPhoto,url:photoUrl,hidden,setShow:setShowPhoto,
    toggle:id=>setHidden(h=>h.includes(id)?h.filter(v=>v!==id):[...h,id]),preset:p=>setHidden(presetHidden(result?.face_overlay,p))};
  const step=page==='home'?0:page==='camera'||page==='camera-retry'||page==='upload'||page==='review'?1:page==='progress'?2:page==='results'?3:4;
  const fixture=import.meta.env.MODE==='test' && window.__FACETELLER_TEST__?.fixture;
  return <div className={`app ${dark?'dark':''} ${page==='preview'?'preview-page':''}`}>
    <div className="wrap screen-only"><header className="doc-head"><span className="wordmark">페이스텔러</span><span className="meta">얼굴 형태 분류와 전통 문헌 조회</span></header>
      <div className="device"><div className="appbar"><button className="brand" onClick={reset} aria-label="페이스텔러 처음으로">페이스텔러 <span>FaceTeller</span></button><button className="theme-toggle" onClick={()=>setDark(v=>!v)} aria-label={dark?'밝은 테마로 전환':'어두운 테마로 전환'}>{dark?'☀':'◐'}</button></div>
        <ol className="steps" aria-label="이용 단계">{['시작','사진 입력','분석','결과','결과지'].map((s,i)=><li key={s} aria-current={step===i?'step':undefined}><span>{i+1}</span>{s}</li>)}</ol>
        <main className="stage" ref={heading} tabIndex={-1}>{fixture && <p className="fixture-banner">합성 UI 검증 · 실제 얼굴 분석 결과 아님</p>}{notice && <div className="alert" role="alert">{notice}</div>}
          {page==='home' && <section className="home center"><p className="eyebrow">FACE · FORM · LITERATURE</p><h1>얼굴 형태를 분류하고<br/>전통 문헌의 기록을 함께 봅니다</h1><p className="lead">눈썹·눈·코·입 등의 관찰 가능한 형태와<br/>그 형태를 기술한 문헌 조항을 근거와 함께 확인하세요.</p><div className="home-motif" aria-hidden="true"><span>形</span><i/><span>文</span></div><div className="stack"><button className="button" onClick={startCamera}>웹캠으로 촬영하기 <span aria-hidden="true">→</span></button><button className="button ghost" onClick={()=>{mode.current='upload';setNotice('');setPage('upload');}}>이미지 파일 업로드</button></div><p className="note">{NOTICE}</p></section>}
          {page==='camera' && <Camera onReady={cameraReady} onError={cameraError} onBack={reset}/>}
          {page==='camera-retry' && <section className="center"><h1>사진을 다시 촬영해 주세요</h1><p className="lead">이전 사진은 정리했습니다. 정면부터 세 방향을 다시 촬영합니다.</p><div className="actions center"><button className="button" onClick={startCamera}>다시 촬영</button><button className="button ghost" onClick={()=>cameraError('사진 3장을 업로드해 주세요.')}>업로드로 전환</button></div></section>}
          {page==='review' && captured && <section className="center" aria-labelledby="review-title"><p className="eyebrow">촬영 확인</p><h1 id="review-title">사진이 제대로 나왔는지 확인해 주세요</h1><p className="lead">실제로 분석에 쓰일 사진 그대로입니다. 지금은 서버 분석을 하지 않습니다.</p><div className="review-grid">{DIRECTIONS.map(d=><figure key={d} className="review-shot">{capturedUrls[d] && <img src={capturedUrls[d]} alt={`${d==='front'?'정면':captureDirectionLabel(d,'upload')+'을 향한'} 사진`}/>}<figcaption>{d==='front'?'정면':`${captureDirectionLabel(d,'upload')}을 향한 사진`}</figcaption></figure>)}</div><div className="actions center"><button className="button" onClick={captured.mode==='camera'?startCamera:()=>{setCaptured(null);setPage('upload');}}>{captured.mode==='camera'?'다시 촬영':'다시 업로드'}</button><button className="button ghost" onClick={reset}>처음으로</button></div></section>}
          {page==='upload' && <Upload onReady={uploadReady} onBack={reset} onCamera={startCamera}/>}
          {page==='progress' && <ProgressScreen stages={STAGES} paced={paced} percents={STAGES.map((_,i)=>percent(i))} status={shownStage} plan={plan} fraction={finishAt.current===null?fraction:1} elapsed={Math.max(0,now-startedAt.current)} onCancel={cancel}/>}
          {result && page==='results' && <Results result={result} photo={photo} portrait={{url:portrait?.url??null,anchors:portrait?.anchors??null,pending:portraitPending}} onPreview={()=>setPage('preview')} onReset={reset}/>}
          {result && page==='preview' && <div className="preview-heading"><h1>문헌 기록을 한 장에</h1><p className="lead">원문·풀이·출처와 확인한 생김새를 담습니다.</p></div>}
        </main></div><footer className="app-footer">FaceTeller <span>관찰 가능한 형태와 역사적 문헌의 연결</span></footer></div>
    {result && <Printable result={result} photo={photo} preview={page==='preview'} onBack={()=>setPage('results')}/>}
  </div>;
}
