"""Write the reviewed run's measured results, keeping automated agreement explicit."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def report(candidate, service_report=None, download=None):
    protocol=load(candidate/'protocol.json');targets=load(candidate/'targets.json')
    manifest=load(candidate/'deployment_candidate.json');test=load(candidate/'test_metrics.json')
    registry=load(ROOT/'data/release/training_registry.json')
    models={p:load(candidate/f'models/resnet18/{p}.json') for p in registry['parts']}
    output=ROOT/'outputs/release/reviewed-report-20261015';output.mkdir(parents=True,exist_ok=True)
    total_seconds=sum(e['train_seconds'] for m in models.values() for e in m['history'])
    environment=load(candidate/'training_environment.json') if (candidate/'training_environment.json').exists() else {}
    fields=['항목','항목_ID','학습용_사람수','학습용_부위사진수','학습_반복횟수','선택된_반복차수',
            '학습대상_정답있는사람수','보정용_정답있는사람수','검사용_정답있는사람수',
            '자동레이블_일치율','macro_F1','보정가능','보정불가이유','확정허용형태','보류기준점수',
            '검사중_확정수','검사중_잘못확정수','검사중_보류수']
    rows=[]
    for key, result in test['features'].items():
        spec=registry['features'][key];model=models[spec['part']];cal=manifest['features'][key]
        metric=result['calibrated'] or result['raw'];policy=result.get('policy') or {}
        reason=({'unsupported_head':'자동 레이블의 신뢰성 부족',
                 'insufficient_calibration_support':'서로 다른 형태의 검증 사례 부족',
                 'temperature_fit_failed':'점수 보정 실패'}.get(cal['status'],cal['status']) if not cal['calibrated'] else '')
        allowed=','.join(spec['classes_ko'][spec['classes'].index(c)] for c in cal['allowed_confirmation_classes'])
        rows.append(dict(zip(fields,[spec['ko'],key,model['train_people'],model['train_crops'],len(model['history']),
            model['selected_epoch'],sum(targets['support']['train'][key]['hard']),cal['n_val'],metric['valid_count'],
            metric['accuracy'],metric['macro_f1'],cal['calibrated'],reason,allowed,cal['threshold'],
            policy.get('accepted',0),policy.get('wrong_confirmed',0),policy.get('withheld',metric['valid_count'])])))
    metrics_file=output/'FaceTeller_재학습_항목별결과_20261015.csv'
    with metrics_file.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['FaceTeller 수정·재학습 결과 (2026-10-15)','',
        f"후보: {protocol['version']}",
        '실행 GPU: '+environment.get('gpu','기록 없음'),
        '대상 400명: 학습270 / 모델선택34 / 점수보정34 / 최종검사62. 같은 사람이 집합 사이에 겹치지 않습니다.',
        'ResNet18만 사용했습니다. 부위별 모델12개, 분류항목41개입니다. VGGT 측정4개는 기존 설정을 유지했습니다.',
        f'실제 학습·반복별 선택검사 소요 합계: {total_seconds/60:.1f}분. 자료 준비·최종 보정·HTTP 검사는 이 시간에 포함하지 않습니다.',
        f"별도 자료로 점수를 보정할 수 있었던 분류항목: {sum(c['calibrated'] for c in manifest['features'].values())}/41.",
        '', '수정한 내용',
        '- 눈썹 기준선 대신 분할된 눈썹 털을 측정해 학습 레이블을 만들었습니다.',
        '- 떨어진 조각, 잘린 영역, 반복 촬영에서 흔들리는 값은 제외했습니다.',
        '- 얇은 꼬리를 길이에서 누락하는 문제와 양쪽 평균에 불안정한 한쪽 값이 남는 문제를 수정했습니다.',
        '- 얼굴 전체와 비교하는 항목은 부위 사진과 얼굴 전체 사진을 같은 ResNet18에 함께 입력합니다.',
        '- 경계 부근을 참/거짓으로 버리지 않고 반복 측정 변동을 반영한 부드러운 레이블을 사용했습니다.',
        '- 검증 사례가 부족한 형태는 높은 점수가 나와도 확정하지 않습니다.',
        '- 의미가 맞지 않는 문헌 연결은 순위를 계산하기 전에 제외했습니다.',
        '- 정면 사진과 측정선을 결과 화면과 인쇄에 선택해서 표시할 수 있습니다.',
        '', '수치 해석',
        'CSV의 일치율은 자동 측정·AI 레이블과 모델 분류가 얼마나 일치하는지입니다. 사람이 정한 독립 정답과의 정확도가 아닙니다.',
        '눈썹 길이의 짧음·김은 눈 가로 대비 비율을 비교 집단과 비교한 값입니다. 짧은 편이라고 눈보다 짧다는 뜻은 아닙니다.',
        '레이블·입력 방식이 달라진 항목은 이전 일치율과 단순 비교할 수 없습니다. 최종62명도 과거 실험에 쓰인 사람들입니다.',
        'VGGT의 밀리미터 오차나 관상 풀이의 사실성을 검증한 결과는 아닙니다.',
        '촬영 사진과 측정선 좌표는 이 보고서에 포함하지 않았습니다.', '']
    brow=test['features']['brow_length'];bm=brow['calibrated'] or brow['raw'];bp=brow.get('policy') or {}
    lines += ['아직 부족한 부분',
        f"눈썹 길이: 정답을 만들 수 있었던 검사 {bm['valid_count']}명에서 자동 레이블 일치율 {bm['accuracy']*100:.1f}%. "
        f"확정 {bp.get('accepted',0)}명, 그중 자동 레이블 불일치 {bp.get('wrong_confirmed',0)}명입니다.",
        '확정한 소수 사례가 맞았다는 것과 전체 분류가 충분히 정확하다는 것은 다릅니다.',
        '사진의 측정선으로 위치 오류를 확인할 수 있게 됐지만, 눈썹 길이 분류와 보류 과다 문제까지 모두 해결됐다고 볼 수는 없습니다.', '']
    if service_report:
        service=load(service_report);people=service.get('rows',service.get('participants',[]))
        # The client keeps per-person scalar summaries; never copy full responses.
        if not isinstance(people,list):people=[]
        if people:
            holds=Counter(reason for p in people for i in p.get('items',[]) for reason in i.get('hold_reasons',[]))
            lines += ['실제 HTTP 분석 검사',f'검사한 사람: {len(people)}명.',
                      f"오류 응답: {sum(p.get('status')=='error' for p in people)}건.",
                      f"평균 확정 항목: {sum(p.get('confirmed',0) for p in people)/len(people):.2f}/45.",
                      f"평균 보류 항목: {sum(p.get('held',0) for p in people)/len(people):.2f}/45.",
                      '보류 사유 집계: '+json.dumps(dict(holds),ensure_ascii=False),'']
            seconds=sorted(p['seconds'] for p in people)
            lines += [f'로컬 요청부터 결과까지: 평균 {statistics.mean(seconds):.2f}초, 최댓값 {max(seconds):.2f}초.',
                      '원본 사진 3장으로 검사했습니다. 사용자 촬영 대기와 외부 인터넷 업로드 시간은 포함하지 않습니다.',
                      f"측정선 구조·사진 범위 검사 통과: {sum(p.get('overlay_verified',False) for p in people)}명.",
                      '선이 해부학적으로 정확한지는 사람이 정한 정답과 비교한 검사가 아닙니다.', '']
            baseline_path=ROOT/'outputs/release/service-verification.json'
            if baseline_path.exists():
                old=load(baseline_path).get('participants',[])
                if len(old)==len(people)==400:
                    lines += [f"기존 평균 보류 {statistics.mean(p['held'] for p in old):.2f}개 → 현재 {statistics.mean(p['held'] for p in people):.2f}개.",
                              '이번 수정은 근거가 부족한 확정을 막은 항목도 있어 보류가 늘 수 있습니다. 보류 감소가 달성됐다는 결과는 아닙니다.', '']
    summary=output/'FaceTeller_수정_재학습_결과_20261015.txt'
    summary.write_text('\n'.join(lines),encoding='utf-8-sig')
    if download:
        download.mkdir(parents=True,exist_ok=True)
        for file in (metrics_file,summary):shutil.copy2(file,download/file.name)
    return summary,metrics_file


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--service-report',type=Path)
    parser.add_argument('--download',type=Path)
    args=parser.parse_args()
    print('\n'.join(str(p) for p in report(ROOT/args.candidate,args.service_report,args.download)))
