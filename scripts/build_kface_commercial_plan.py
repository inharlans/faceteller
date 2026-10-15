from pathlib import Path
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/kface-application'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold')
styles={
 'title':ParagraphStyle('Title',fontName='MalgunBold',fontSize=18,leading=25,spaceAfter=15,textColor=colors.black,wordWrap='CJK'),
 'head':ParagraphStyle('Heading',fontName='MalgunBold',fontSize=12,leading=18,spaceBefore=12,spaceAfter=6,keepWithNext=True,wordWrap='CJK'),
 'body':ParagraphStyle('Body',fontName='Malgun',fontSize=10.5,leading=16.3,spaceAfter=7,wordWrap='CJK'),
 'cell':ParagraphStyle('Cell',fontName='Malgun',fontSize=9.7,leading=14.5,wordWrap='CJK'),
 'label':ParagraphStyle('Label',fontName='MalgunBold',fontSize=9.7,leading=14.5,wordWrap='CJK'),
}

title='한국인 얼굴 사진 데이터베이스 활용 계획서'
metadata=[
 ('연구 과제','AI 기반 얼굴 형태 분류 및 전통 관상 문헌 연계 서비스 개발'),
 ('신청자와 책임자','성명, 소속 및 자료 이용 책임자는 함께 제출하는 자료 요청서의 기재 내용과 동일함'),
 ('이용 목적과 기간','형태 분류 연구 및 상업화 목적 시제품 개발·평가 / 자료 요청서에 기재한 이용 예정 기간 내'),
 ('요청 자료','제공 가능한 승인 대상 참여자의 고해상도 얼굴 사진 및 해당 촬영 조건·참여자 구분 메타데이터'),
]
pages=[[
 ('1 연구 목적과 활용 범위',[
 '한국인 얼굴 사진에서 눈, 눈썹, 코, 입 및 얼굴의 외형을 관찰하고, 일관된 형태 레이블을 구축하여 부위별 이미지 분류 모델을 개발하고자 합니다. 얼굴 특징점 추출과 영상 전처리를 이용해 촬영 조건에 따른 영향을 줄이고, 한국인 자료에서 형태 분류의 재현성과 성능을 평가하겠습니다.',
 '전통 관상 문헌에 나타나는 형태 표현을 정리하여 모델이 분류한 외형과 관련 문헌을 함께 확인하는 상업화를 고려한 서비스 시제품을 구현할 계획입니다. 문헌의 역사적 해석과 사진에서 관찰되는 외형을 구분하며, 사진으로 개인의 성격·지능·도덕성·건강·운명을 사실처럼 판정하는 정답 레이블은 생성하지 않겠습니다.',
 '본 신청은 데이터셋 구축, CNN 학습·평가, 상업화 목적의 시제품 개발과 향후 학습 완료 모델을 이용한 상업 서비스 운영 가능성을 포함합니다. 서비스에서는 이용자가 제공한 사진을 모델로 분석하고 관련 문헌 정보를 제공합니다. K-FACE 원본 사진을 고객에게 제공하거나 판매하지 않으며, 모델을 통한 상업적 이용의 허용 범위와 조건을 확인·승인받고자 합니다.'
 ]),
 ('2 요청 자료와 선정 기준',[
 '눈꺼풀, 눈썹, 코끝 및 입술의 세부 윤곽을 확인할 수 있도록 고해상도 자료를 요청합니다. 승인 범위 내에서 정면·무표정·가림이 적은 사진을 우선 선정하며, 동일인의 다른 각도·조명·표정 사진은 촬영 조건 변화에 대한 비교 평가에 사용하겠습니다. 저·중해상도 자료가 추가로 필요해지면 별도 협의하겠습니다.',
 '실제 이름 등 직접 식별정보는 필요하지 않습니다. 동일인을 학습·검증·시험 세트에 중복 배치하지 않기 위한 데이터셋 내 참여자 구분값과 촬영 조건 정보만 활용하고, 제공 가능한 자료의 범위와 수량에 맞춰 최종 분석 대상을 확정하겠습니다.'
 ]),
 ('3 처리와 레이블링 방법',[
 '자료 수신 후 파일 구성과 촬영 조건을 확인하고 화질·가림·얼굴 방향을 점검하겠습니다. MediaPipe 등 얼굴 특징점 추출 모델을 이용하여 얼굴을 정렬하고, 눈·눈썹·코·입·얼굴 부위의 분석 입력을 생성하겠습니다. 필요한 비율·각도와 관찰 메모를 기록하되, 자동 측정값과 사람이 검수한 형태 레이블을 구분하겠습니다.',
 '전통 유형명은 문헌의 형태 조건을 검토한 뒤 관찰 가능한 항목으로 분해하겠습니다. 판정이 모호하거나 사진으로 확인할 수 없는 항목은 보류하고, 표본의 중복 검수와 의견 조정을 통해 레이블 기준을 보완하겠습니다. 자동 생성한 초벌 주석을 독립 검수 없이 확정 정답으로 사용하지 않겠습니다.'
 ]),
], [
 ('4 모델 개발과 평가',[
 '눈, 눈썹, 코, 입, 얼굴형을 대상으로 부위별 합성곱 신경망(CNN) 분류기를 구성하고 사전학습 모델을 미세조정하겠습니다. 부위 입력의 종횡비를 보존하며, 같은 참여자의 모든 사진과 좌우 부위가 하나의 분할에만 포함되도록 참여자 단위로 학습·검증·시험 자료를 분리하겠습니다.',
 '모델 및 학습 설정은 검증 자료로 선택하고, 시험 자료에서 정확도, 클래스별 정밀도·재현율, macro F1, 혼동행렬 및 판정 보류 비율을 확인하겠습니다. 표정·조명·각도별 결과와 레이블 불일치 사례도 분석하여 적용 가능한 촬영 조건과 한계를 정리하겠습니다.'
 ]),
 ('5 산출물과 추진 순서',[
 '주요 산출물은 형태 레이블 정의서, 문헌 출처 사전, 승인 범위 내의 이미지별 주석·측정 자료, 부위별 분류 모델, 성능 평가 보고서 및 상업화 목적 시제품입니다. 학습 완료 모델을 서비스에 탑재하여 추론에 활용할 계획이며, 외부 서비스 공개와 유료 운영은 제공기관이 승인한 범위와 조건에 따라 진행하겠습니다.',
 '추진 순서는 자료 수신 및 승인 범위 확인, 분석 대상 선정과 전처리, 형태 레이블 작성과 검수, 모델 학습과 평가, 결과 정리와 자료 폐기입니다. 각 단계는 요청서에 기재한 이용 예정 기간 안에 수행하며, 기간 연장이 필요하면 승인번호·연장 목적·연장 기간을 제출하여 재승인 후 이용하겠습니다.'
 ]),
 ('6 접근 관리와 자료 보호',[
 '승인된 연구자와 자료 이용 책임자에게만 접근을 허용하겠습니다. 팀원이 참여하는 경우에도 승인받은 이용자 범위에 포함되는지 먼저 확인하고, 승인되지 않은 사람에게 자료를 열람·제공하지 않겠습니다. 외부 자료와 대조하여 신원을 찾거나 개인을 재식별하는 작업은 수행하지 않겠습니다.',
 '원본과 허용된 파생 자료는 접근을 통제한 연구용 저장공간에서 관리하겠습니다. 공개 저장소, 공유 링크, 공개 웹페이지에 올리지 않으며, 별도 승인 없이 외부 생성형 AI API나 클라우드 서비스에 얼굴 사진을 전송하지 않겠습니다. 연구용 화면에서도 승인된 이용자에게만 사진이 보이도록 제한하겠습니다.',
 '연구 종료 후 제공 자료와 승인된 처리 과정에서 생성된 얼굴 이미지·개인별 주석 등 관련 자료를 폐기하겠습니다. 자료 이용 기간 종료 후에도 학습된 모델 가중치를 보관하고 상업 서비스의 추론에 사용하는 것이 가능한지 함께 승인 요청드립니다. 가중치와 비식별 집계 결과는 승인된 기간·목적·조건에 따라 보관·이용하겠습니다.'
 ]),
 ('7 처리·상업적 이용 승인 요청과 결과 활용',[
 '서약서 제2항의 자료 수정·복사 금지 조건과 관련하여, 본 계획에 기재한 얼굴 정렬, 부위 추출, 학습 입력 변환, 주석·측정값 생성·저장 및 모델 학습에 필요한 처리의 허용 범위를 함께 확인·승인하여 주시기 바랍니다. 학습 완료 모델의 상업 서비스 탑재·추론, 자료 폐기 후 가중치 보관·이용의 허용 여부와 기간도 함께 확인·승인하여 주시기 바랍니다. 추가 계약이나 제한이 필요한 경우 해당 조건을 충족한 뒤 이용하겠습니다.',
 '연구 결과물을 활용할 때에는 예정 시점 14일 전에 지정 기관에 알리고, 한국과학기술연구원과 한국지능정보사회진흥원을 출처로 명시하며 결과 내역 사본을 지정 메일로 제출하겠습니다. 보고서·논문·발표자료 등에 홈페이지의 공개 예시 사진 이외의 제공 사진이 표출되지 않도록 하겠습니다.'
 ]),
]]

def para(text,style='body'):return Paragraph(escape(text),styles[style])
story=[para(title,'title')]
table=Table([[para(k,'label'),para(v,'cell')] for k,v in metadata],colWidths=[91,416])
table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f0f1f3')),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#c8cdd3'))]))
story.append(table)
for i,sections in enumerate(pages):
    if i:story.append(PageBreak())
    for heading,paras in sections:
        story.append(para(heading,'head'))
        story.extend(para(p) for p in paras)

def footer(canvas,doc):
    canvas.setFont('Malgun',8);canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawRightString(A4[0]-44,24,str(doc.page))

pdf=OUT/'K-FACE_데이터_활용계획서_상업화고려.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=40,title=title,author='',subject='K-FACE 연구 및 상업화 목적 자료 이용 신청 첨부 계획서')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
lines=[title,'']+[f'{k}: {v}' for k,v in metadata]
for sections in pages:
    for heading,paras in sections:lines.extend(['',heading,'',*sum(([p,''] for p in paras),[])])
(OUT/'K-FACE_데이터_활용계획서_상업화고려_수정용.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
print(pdf)
