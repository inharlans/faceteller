import {describe,it,expect} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {FaceDiagram,itemSummary} from '../src/Results';
import {REGIONS,sourceText,sourceUrl,type AnalysisResult,type Item,type Passage} from '../src/types';
import {validateResult} from '../src/api';

const passage:Passage={passage_id:'book:12',original_text:'原文',translation:'직역',commentary:'해설',source:{title:'신상전편',chapter:'눈썹',locator:'제12조',url:'https://zh.wikisource.org/wiki/example',annotation_status:'automated_unreviewed'},evidence_item_ids:[]};
describe('release database and twelve-region UI integration',()=>{
  it('accepts actual PostgreSQL source vocabulary and retains source location',()=>{
    expect(validateResult({status:'PARTIAL',items:[],passages:[passage]}).passages[0].source).toEqual(passage.source);
    expect(sourceText(passage)).toBe('신상전편 · 눈썹 · 제12조');
    expect(sourceUrl(passage)).toBe(passage.source.url);
    expect(sourceUrl({...passage,source:{...passage.source,url:'javascript:alert(1)'}})).toBeUndefined();
  });
  it('shows a scanned-book page number without the technical character offset',()=>{
    expect(sourceText({...passage,source:{...passage.source,title:'柳莊相法',chapter:'上冊',passage_no:'論眉',pdf_page:28}})).toBe('柳莊相法 · 上冊 · 論眉 · 원문 스캔 28쪽');
  });
  it('renders every declared region, including ear and the newly added regions',()=>{
    const result:AnalysisResult={status:'PARTIAL',items:[],passages:[]};
    const html=renderToStaticMarkup(<FaceDiagram result={result} active={['ear']}/>);
    for(const region of Object.keys(REGIONS))expect(html).toContain(`data-region="${region}"`);
    expect((html.match(/data-region=/g)||[]).length).toBe(12);
    expect(html).toMatch(/data-region="ear" class="pin held active"/);
  });
  it('explains production hold codes in plain Korean',()=>{
    const item:Item={item_id:'forehead_height',region:'forehead',status:'HELD',kind:'classification',label_id:null,label_name:null,confidence:.3,measurement:null,basis:'자동 주석 기준',hold_reasons:['LOW_OR_INVALID_CONFIDENCE','HAIRLINE_NOT_SUFFICIENTLY_VISIBLE']};
    const text=itemSummary(item);
    expect(text).toContain('이마 높이');expect(text).toContain('머리선이 충분히 보이지 않음');expect(text).not.toContain('LOW_OR_INVALID_CONFIDENCE');
  });
});
