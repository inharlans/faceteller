"""Non-GPU pilot contracts for design.txt (2026-10-14), independent of server.py.

Integration: create_app(lambda: AnalysisPipeline(preprocessor, classifier, depth,
repository)). Inject request-scoped services; model instances may be shared by
the adapter. No models, camera devices, databases, or sockets open on import.
Adapters must not retain InputSet/context after a request or log face data.
Async adapters must offload blocking inference themselves.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator, Callable, Iterable, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image, UnidentifiedImageError
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

DIRECTIONS = ("front", "left_oblique", "right_oblique")
NOTICE = "전통 문헌 조회 결과이며 개인의 성격·지능·건강·운명을 판정하거나 예측하지 않습니다."
STAGES = ("얼굴 정렬", "부위 추출", "형태 분류", "깊이 분석", "문헌 선정")
SERVICE_SCORE_MODE = "include_measurement_scores"
SCORING_VERSION = "cnn_depth_geomean_v1"


@dataclass(frozen=True)
class Observation:
    item_id: str
    region: str
    kind: str  # classification | depth
    label_id: str | None
    label_name: str | None
    confidence: float | None = None
    class_count: int = 3
    calibrated: bool = False
    measurement: dict | None = None
    measurement_score: float | None = None  # CDF membership score; separate from confidence, included in G
    quality_passed: bool = False
    boundaries_validated: bool = False
    basis: str = ""
    hold_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    item_id: str
    region: str
    status: str
    label_id: str | None
    label_name: str | None
    confidence: float | None
    measurement: dict | None
    measurement_score: float | None
    basis: str
    hold_reasons: tuple[str, ...]
    kind: str
    measurement_score_kind: str | None = None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class DecisionPolicy:
    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = dict(thresholds or {})

    def apply(self, observations: Iterable[Observation]) -> list[Decision]:
        out, ids = [], set()
        for o in observations:
            if o.item_id in ids:
                raise ValueError("Duplicate item_id")
            ids.add(o.item_id)
            reasons = list(o.hold_reasons)
            confidence, measurement = None, None
            if o.kind == "classification":
                if o.class_count < 2:
                    raise ValueError("Classification needs at least two classes")
                threshold = self.thresholds.get(o.item_id, .60)
                floor = .60 if o.class_count == 2 else .50
                if not finite(threshold) or not floor <= threshold <= 1:
                    raise ValueError("Threshold below design policy floor or above 1")
                if finite(o.confidence) and 0 <= o.confidence <= 1:
                    confidence = o.confidence
                if confidence is None or confidence < threshold:
                    reasons.append("LOW_OR_INVALID_CONFIDENCE")
                if not o.calibrated:
                    reasons.append("CONFIDENCE_NOT_CALIBRATED")
            elif o.kind == "depth":
                m = o.measurement or {}
                if m.get("indicator") and m.get("unit") == "ratio" and finite(m.get("value")):
                    measurement = {"indicator": m["indicator"], "value": m["value"], "unit": "ratio"}
                if measurement is None:
                    reasons.append("MEASUREMENT_UNAVAILABLE")
                if not o.quality_passed:
                    reasons.append("DEPTH_QUALITY_FAILED")
                if not o.boundaries_validated:
                    reasons.append("DEPTH_BOUNDARIES_UNVALIDATED")
                if not finite(o.measurement_score) or not 0 <= o.measurement_score <= 1:
                    reasons.append("DEPTH_SCORE_UNAVAILABLE")
            else:
                raise ValueError("Unknown observation kind")
            if not o.label_id or not o.label_name:
                reasons.append("LABEL_UNAVAILABLE")
            if not o.basis:
                reasons.append("BASIS_UNAVAILABLE")
            out.append(Decision(o.item_id, o.region, "HELD" if reasons else "CONFIRMED",
                                None if reasons else o.label_id, None if reasons else o.label_name,
                                confidence, measurement,
                                o.measurement_score if o.kind == "depth" and finite(o.measurement_score) and 0 <= o.measurement_score <= 1 else None,
                                o.basis, tuple(dict.fromkeys(reasons)), o.kind,
                                "geometry_membership_not_calibrated_accuracy" if o.kind == "depth" and finite(o.measurement_score) and 0 <= o.measurement_score <= 1 else None))
        return out


@dataclass(frozen=True)
class Passage:
    passage_id: str
    required_labels: frozenset[str]
    original_text: str
    translation: str
    commentary: str
    source: dict
    literature_priority: int = 0
    passage_order: int = 0


class ReferenceRepository(Protocol):
    def load_candidates(self, label_ids: set[str]) -> list[Passage]: ...


class MemoryRepository:
    def __init__(self, passages: Iterable[Passage]):
        self.passages = tuple(passages)

    def load_candidates(self, label_ids: set[str]) -> list[Passage]:
        # Return ALL conditions on any overlapping passage, never just matched labels.
        return [p for p in self.passages if p.required_labels & label_ids]


class PassageSelector:
    def __init__(self, score_mode: str = SERVICE_SCORE_MODE):
        if score_mode not in ("classification_only", "include_measurement_scores"):
            raise ValueError("Unknown score_mode")
        self.score_mode = score_mode

    def select(self, items: list[Decision], candidates: Iterable[Passage]) -> list[dict]:
        confirmed = {}
        for i in items:
            if i.status == "CONFIRMED":
                if i.label_id in confirmed:
                    raise ValueError("Ambiguous evidence: duplicate confirmed label_id")
                confirmed[i.label_id] = i
        eligible, seen = [], {}
        for p in candidates:
            if p.passage_id in seen:
                if seen[p.passage_id] != p:
                    raise ValueError("Conflicting duplicate passage data")
                continue
            seen[p.passage_id] = p
            if not p.required_labels or not p.required_labels <= confirmed.keys():
                continue
            evidence = [confirmed[k] for k in sorted(p.required_labels)]
            components = [{"item_id": i.item_id, "label_id": i.label_id,
                           "label_name": i.label_name, "kind": i.kind,
                           "score": i.confidence if i.kind == "classification" else i.measurement_score,
                           "score_kind": "calibrated_classification_confidence" if i.kind == "classification"
                                         else "geometry_membership_not_calibrated_accuracy"}
                          for i in evidence if i.kind == "classification" or self.score_mode == SERVICE_SCORE_MODE]
            scores = [c["score"] for c in components]
            if any(not finite(c) or not 0 <= c <= 1 for c in scores):
                raise ValueError("Invalid confirmed feature score")
            g = (0.0 if 0 in scores else math.exp(math.fsum(math.log(c) for c in scores)
                                                / len(scores))) if scores else None
            eligible.append((p, evidence, g, components))
        def identifier_key(identifier):
            # Production IDs are BIGINT, source-pilot IDs are strings.
            return (0, int(identifier)) if str(identifier).isdecimal() else (1, str(identifier))

        eligible.sort(key=lambda x: (x[2] is None, -(x[2] or 0), x[0].literature_priority,
                                     x[0].passage_order, identifier_key(x[0].passage_id)))
        selected, counts = [], Counter()
        for p, evidence, g, components in eligible:
            regions = {i.region for i in evidence}
            if any(counts[r] >= 3 for r in regions):
                continue
            selected.append({"passage_id": p.passage_id, "original_text": p.original_text,
                             "translation": p.translation, "commentary": p.commentary,
                             "source": dict(p.source), "evidence_item_ids": [i.item_id for i in evidence],
                             "G": g, "score_mode": self.score_mode,
                             "rank": len(selected) + 1,
                             "score_kind": "geometric_feature_match",
                             "score_feature_count": len(components),
                             "score_components": components})
            counts.update(regions)
            if len(selected) == 5:
                break
        return selected


def build_result(items: list[Decision], repository: ReferenceRepository, *, score_mode: str = SERVICE_SCORE_MODE) -> dict:
    labels = {i.label_id for i in items if i.status == "CONFIRMED"}
    selector = PassageSelector(score_mode)
    passages = selector.select(items, repository.load_candidates(labels)) if labels else []
    status = "REINPUT_REQUIRED" if not labels else "PARTIAL" if any(i.status == "HELD" for i in items) else "COMPLETE"
    return {"status": status, "items": [asdict(i) for i in items], "passages": passages,
            "notice": NOTICE, "literature_status": "NOT_QUERIED" if not labels else "FOUND" if passages else "NONE",
            "ranking_policy": {"version": SCORING_VERSION, "score_mode": score_mode,
                               "formula": "exp(sum(log(s_i))/n)", "weighting": "equal_per_required_feature",
                               "required_conditions": "all_confirmed", "max_passages": 5, "max_per_region": 3,
                               "tie_break": ["literature_priority", "passage_order", "passage_id"],
                               "score_kind": "geometric_feature_match", "is_correctness_probability": False}}


@dataclass
class InputSet:
    images: dict[str, bytearray]
    input_mode: str
    media_types: dict[str, str] = field(default_factory=dict)

    def clear(self):
        for data in self.images.values():
            data.clear()
        self.images.clear()
        self.media_types.clear()


class Preprocessor(Protocol):
    async def validate(self, inputs: InputSet) -> None:
        """Check faces in all three images; raise ContractError with direction."""
        ...

    async def align(self, inputs: InputSet, context: dict) -> None: ...
    async def extract(self, inputs: InputSet, context: dict) -> None:
        """Populate context['detections'] (three-view landmarks), regions, etc."""
        ...


class RegionClassifier(Protocol):
    async def classify(self, context: dict) -> list[Observation]: ...


class DepthEstimator(Protocol):
    async def estimate(self, inputs: InputSet, context: dict) -> list[Observation]:
        """Three images PLUS preprocessing detections/landmarks are dependencies.

        Main adapter: VGGTEngine.infer(PILimages) -> geometry (network independent
        of landmarks); measure_geometry(geometry, context['detections']) -> values
        after 3D mapping. Preserve CDF output as measurement_score, not confidence.
        This pilot neither imports nor implements the GPU depth engine.
        """
        ...


class AnalysisService(Protocol):
    """Request-scoped adapter. validate may allocate decoded images/detections.

    API owns InputSet buffers. Adapters retaining additional resources may expose
    optional synchronous, idempotent close(). The API invokes it on validation
    failure/cancellation and response finalization, including response-start
    failure before analyze begins. Adapter finally blocks may also call close().
    """
    async def validate(self, inputs: InputSet) -> None: ...
    def analyze(self, inputs: InputSet) -> AsyncIterator[dict]: ...


class AnalysisPipeline:
    def __init__(self, preprocessor: Preprocessor, classifier: RegionClassifier,
                 depth: DepthEstimator, repository: ReferenceRepository, policy: DecisionPolicy | None = None):
        self.preprocessor, self.classifier, self.depth = preprocessor, classifier, depth
        self.repository, self.policy = repository, policy or DecisionPolicy()

    async def validate(self, inputs):
        await self.preprocessor.validate(inputs)

    async def analyze(self, inputs):
        context, observations = {}, []
        try:
            yield {"type": "progress", "stage": STAGES[0]}
            await self.preprocessor.align(inputs, context)
            yield {"type": "progress", "stage": STAGES[1]}
            await self.preprocessor.extract(inputs, context)
            yield {"type": "progress", "stage": STAGES[2]}
            observations.extend(await self.classifier.classify(context))
            yield {"type": "progress", "stage": STAGES[3]}
            observations.extend(await self.depth.estimate(inputs, context))
            items = self.policy.apply(observations)
            yield {"type": "progress", "stage": STAGES[4]}
            yield {"type": "result", "data": build_result(items, self.repository)}
        finally:
            context.clear()
            observations.clear()


class ContractError(Exception):
    def __init__(self, code: str, message: str, status: int, direction: str | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.direction = code, message, status, direction

    def event(self):
        return {"type": "error", "code": self.code, "message": self.message, "direction": self.direction}


@dataclass(frozen=True)
class InputLimits:
    file_bytes: int = 8 * 1024 * 1024
    request_bytes: int = 24 * 1024 * 1024 + 64 * 1024
    header_bytes: int = 8192
    pixels: int = 12_000_000


class MemoryMultipart:
    """Low-level callbacks only: never UploadFile, request.form(), or tempfile."""
    def __init__(self, content_type: str, limits: InputLimits):
        media, options = parse_options_header(content_type)
        boundary = options.get(b"boundary")
        if media != b"multipart/form-data" or not boundary or len(boundary) > 200:
            raise ContractError("INVALID_FORMAT", "multipart boundary가 필요합니다.", 415)
        self.limits, self.parts, self.types = limits, {}, {}
        self.complete, self.total, self.count = False, 0, 0
        self.current, self.name = bytearray(), None
        self.headers, self.header_name, self.header_value = {}, bytearray(), bytearray()
        self.header_size = 0
        self.parser = MultipartParser(boundary, {
            "on_part_begin": self.begin, "on_header_field": self.header_field,
            "on_header_value": self.header_val, "on_header_end": self.header_end,
            "on_headers_finished": self.headers_done, "on_part_data": self.data,
            "on_part_end": self.end_part, "on_end": self.end,
        })

    def begin(self):
        self.count += 1
        if self.count > 4:
            raise ContractError("INVALID_FORMAT", "필드 중복 또는 추가 필드입니다.", 415)
        self.headers, self.current, self.name = {}, bytearray(), None
        self.header_size = 0

    def header_piece(self, target, data, start, end):
        self.header_size += end - start
        if self.header_size > self.limits.header_bytes:
            raise ContractError("INPUT_TOO_LARGE", "헤더가 너무 큽니다.", 413)
        target.extend(data[start:end])

    def header_field(self, data, start, end):
        self.header_piece(self.header_name, data, start, end)

    def header_val(self, data, start, end):
        self.header_piece(self.header_value, data, start, end)

    def header_end(self):
        key = bytes(self.header_name).lower()
        if key in self.headers:
            raise ContractError("INVALID_FORMAT", "중복 헤더입니다.", 415)
        self.headers[key] = bytes(self.header_value)
        self.header_name.clear()
        self.header_value.clear()

    def headers_done(self):
        disposition, options = parse_options_header(self.headers.get(b"content-disposition"))
        self.name = options.get(b"name", b"").decode("ascii", errors="replace")
        if disposition != b"form-data" or self.name not in (*DIRECTIONS, "input_mode") or self.name in self.parts:
            raise ContractError("INVALID_FORMAT", "잘못된 필드 또는 중복 필드입니다.", 415)
        media = self.headers.get(b"content-type", b"").decode("ascii", errors="replace").lower()
        if self.name in DIRECTIONS and (b"filename" not in options or media not in ("image/jpeg", "image/png", "image/webp")):
            raise ContractError("INVALID_FORMAT", "JPEG/PNG/WebP 이미지가 필요합니다.", 415, self.name)
        if self.name == "input_mode" and b"filename" in options:
            raise ContractError("INVALID_FORMAT", "input_mode는 문자열 필드입니다.", 415)
        self.types[self.name] = media

    def data(self, data, start, end):
        limit = self.limits.file_bytes if self.name in DIRECTIONS else 16
        if len(self.current) + end - start > limit:
            raise ContractError("INPUT_TOO_LARGE", "입력 용량 제한을 초과했습니다.", 413, self.name if self.name in DIRECTIONS else None)
        self.current.extend(data[start:end])

    def end_part(self):
        self.parts[self.name] = self.current

    def end(self):
        self.complete = True

    def feed(self, chunk):
        self.total += len(chunk)
        if self.total > self.limits.request_bytes:
            raise ContractError("INPUT_TOO_LARGE", "요청 용량 제한을 초과했습니다.", 413)
        self.parser.write(chunk)

    def finish(self):
        self.parser.finalize()
        # python-multipart.finalize() does not itself reject truncated bodies.
        if not self.complete:
            raise ContractError("INVALID_FORMAT", "완료되지 않은 multipart 요청입니다.", 415)
        for direction in DIRECTIONS:
            if not self.parts.get(direction):
                raise ContractError("INPUT_MISSING", "방향별 사진 3장이 필요합니다.", 400, direction)
        mode = bytes(self.parts.get("input_mode", b"")).decode("ascii", errors="replace")
        if mode not in ("camera", "upload"):
            raise ContractError("INPUT_MISSING", "input_mode는 camera 또는 upload입니다.", 400)
        inputs = InputSet({d: self.parts.pop(d) for d in DIRECTIONS}, mode,
                          {d: self.types[d] for d in DIRECTIONS})
        self.current = bytearray()  # ownership transferred to InputSet
        return inputs

    def clear(self):
        self.current.clear()
        for data in self.parts.values():
            data.clear()
        self.parts.clear()
        self.headers.clear()
        self.header_name.clear()
        self.header_value.clear()


async def receive_inputs(request: Request, limits: InputLimits) -> InputSet:
    length = request.headers.get("content-length")
    if length is not None:
        if not length.isdecimal():
            raise ContractError("INVALID_FORMAT", "잘못된 Content-Length입니다.", 415)
        if int(length) > limits.request_bytes:
            raise ContractError("INPUT_TOO_LARGE", "요청 용량 제한을 초과했습니다.", 413)
    parser = MemoryMultipart(request.headers.get("content-type", ""), limits)
    try:
        async for chunk in request.stream():
            parser.feed(chunk)
        return parser.finish()
    except MultipartParseError:
        raise ContractError("INVALID_FORMAT", "잘못된 multipart 요청입니다.", 415) from None
    finally:
        parser.clear()


def validate_images(inputs: InputSet, limits: InputLimits):
    formats = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    for direction, raw in inputs.images.items():
        try:
            with Image.open(BytesIO(raw)) as im:
                if im.format != formats[inputs.media_types[direction]]:
                    raise ContractError("INVALID_FORMAT", "이미지 형식과 MIME이 다릅니다.", 415, direction)
                if im.width * im.height > limits.pixels:
                    raise ContractError("INPUT_TOO_LARGE", "이미지 해상도 제한을 초과했습니다.", 413, direction)
                im.verify()
            with Image.open(BytesIO(raw)) as im:
                im.load()
        except Image.DecompressionBombError:
            raise ContractError("INPUT_TOO_LARGE", "이미지 해상도 제한을 초과했습니다.", 413, direction) from None
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise ContractError("INVALID_FORMAT", "이미지를 디코딩할 수 없습니다.", 415, direction) from None


def ndjson(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


class _MemoryResponse(StreamingResponse):
    def __init__(self, *args, inputs: InputSet, cleanup: Callable[[], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        # Own cleanup even if send(response.start) fails before the async
        # generator begins, or send(body) fails while it is suspended at yield.
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                if hasattr(self.body_iterator, "aclose"):
                    await self.body_iterator.aclose()
            finally:
                try:
                    if self.cleanup is not None:
                        self.cleanup()
                finally:
                    self.inputs.clear()


def create_app(service_factory: Callable[[], AnalysisService], limits: InputLimits | None = None) -> FastAPI:
    """Mandatory injection: no demo/real inference silently selected by the API."""
    app, limits = FastAPI(), limits or InputLimits()

    @app.post("/api/v1/analysis")
    async def stream_analysis(request: Request):
        inputs = service = None
        try:
            inputs = await receive_inputs(request, limits)
            received_at = perf_counter()
            validate_images(inputs, limits)
            service = service_factory()
            await service.validate(inputs)  # face checks before HTTP response starts
        except BaseException as exc:
            try:
                cleanup = getattr(service, "close", None)
                if cleanup is not None:
                    cleanup()
            finally:
                if inputs is not None:
                    inputs.clear()
            if not isinstance(exc, Exception):
                raise  # cancellation is not a server error
            err = exc if isinstance(exc, ContractError) else ContractError("SERVER_ERROR", "분석 준비 중 오류가 발생했습니다.", 500)
            return Response(ndjson(err.event()), status_code=err.status, media_type="application/x-ndjson")

        async def events():
            stream = None
            try:
                stream = service.analyze(inputs)
                async for event in stream:
                    if event.get("type") == "progress" and event.get("stage") in STAGES:
                        yield ndjson(event)
                    elif event.get("type") in ("result", "error"):
                        if event.get("type") == "result":
                            runtime = event.get("data", {}).get("runtime")
                            if isinstance(runtime, dict):
                                # Continuous server interval: receive complete to
                                # result ready, BEFORE terminal serialization/send.
                                runtime.setdefault("timings", {})["received_to_result_seconds"] = perf_counter() - received_at
                        yield ndjson(event)
                        return
                    else:
                        raise ValueError("Invalid service event")
                raise RuntimeError("Service stream ended without a terminal event")
            except Exception as exc:
                err = exc if isinstance(exc, ContractError) else ContractError("SERVER_ERROR", "분석 중 오류가 발생했습니다.", 500)
                yield ndjson(err.event())
            finally:
                try:
                    if stream is not None and hasattr(stream, "aclose"):
                        await stream.aclose()
                finally:
                    inputs.clear()

        return _MemoryResponse(events(), inputs=inputs, cleanup=getattr(service, "close", None),
                               media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    return app


@dataclass
class CameraState:
    """Pure state pilot. Tokens reject late async frame checks after direction changes."""
    hold_seconds: float = 1.0  # pilot setting, not a design-mandated duration
    state: str = "permission"
    index: int = 0
    generation: int = 0
    since: float | None = None
    last_time: float | None = None
    gauge: float = 0.0
    captured: list[str] = field(default_factory=list)
    guidance: tuple[str, ...] = ()
    submitted: bool = False
    stream_open: bool = False

    @property
    def token(self):
        return (self.generation, DIRECTIONS[self.index]) if self.index < 3 else None

    def permission(self, granted: bool):
        if self.state != "permission":
            return
        self.state, self.stream_open = ("capturing", True) if granted else ("upload", False)

    def update(self, token, conditions: dict[str, bool], timestamp: float) -> str | None:
        if self.state != "capturing" or token != self.token:
            return None
        if not finite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            return None
        if not finite(self.hold_seconds) or self.hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive")
        self.last_time = timestamp
        self.guidance = tuple(k for k in ("face", "direction", "size", "brightness") if conditions.get(k) is not True)
        if self.guidance:
            self.since, self.gauge = None, 0.0
            return None
        if self.since is None:
            self.since = timestamp
        self.gauge = min(1.0, (timestamp - self.since) / self.hold_seconds)
        if self.gauge < 1:
            return None
        captured = DIRECTIONS[self.index]
        self.captured.append(captured)
        self.index += 1
        self.generation += 1
        self.since, self.gauge = None, 0.0
        if self.index == 3:
            self.state, self.stream_open = "ready", False
        return captured

    def submit_once(self) -> bool:
        if self.state != "ready" or self.submitted:
            return False
        self.submitted, self.state = True, "analyzing"
        return True

    def close(self):
        self.stream_open, self.state = False, "closed"
        self.since, self.gauge = None, 0.0
        self.generation += 1
        self.captured.clear()


def result_destination(input_mode: str, result: dict) -> str:
    if input_mode not in ("camera", "upload"):
        raise ValueError("Invalid input mode")
    return input_mode if result["status"] == "REINPUT_REQUIRED" else "result"


def load_real_repository(root: Path, *, allow_draft: bool = False):
    """Read existing text, never invent translations or flatten partial/OR rules.

    Pilot IDs remain source strings. Production DB migration must assign BIGINT
    IDs. Legacy section headings are preserved as passage_no (not invented numbers).
    allow_draft explicitly enables experimental use of unreviewed texts, including
    the user-authorized real-inference pilot. Preserve source.review_status and
    report independently_reviewed=0; this is not reviewed production knowledge.
    Loading references does not execute inference: callers report whether their
    observations came from fixtures or actual models separately.
    """
    def load(name):
        return json.loads((root / "data" / (name + ".json")).read_text(encoding="utf-8"))
    catalog = load("interpretation_catalog_v1")["records"]
    services = {r["record_id"]: r for r in load("interpretation_service_v1")["records"]}
    rules = {r["record_id"]: r for r in load("type_rules_v3")["records"]}
    if len({r["record_id"] for r in catalog}) != len(catalog) or set(services) != set(rules) or set(rules) != {r["record_id"] for r in catalog}:
        raise ValueError("Real reference IDs are not unique/aligned")
    passages, excluded, errors = [], Counter(), []
    for order, row in enumerate(catalog):
        rid = row["record_id"]
        service, rule = services[rid], rules[rid]
        required = (row.get("original_section"), row.get("source_url"), row.get("name_zh"),
                    service.get("translation_ko"), service.get("easy_explanation"))
        quotes = service.get("evidence_quotes", []) + [c["evidence"] for c in rule["conditions"] + rule["unobservable_conditions"]]
        if not all(required) or any(not q or q not in row["original_section"] for q in quotes):
            errors.append(rid)
            continue
        if not allow_draft and not row.get("independent_review"):
            excluded["unreviewed"] += 1
        elif rule.get("photo_suggestion_hold"):
            excluded["photo_suggestion_hold"] += 1
        elif rule["rule_coverage"] != "full" or rule["unobservable_conditions"] or not rule["conditions"]:
            excluded["incomplete_conditions"] += 1
        elif any(len(c["values"]) != 1 for c in rule["conditions"]):
            excluded["multi_value_condition_needs_review"] += 1
        else:
            volume = rid[4:7]
            passages.append(Passage(rid, frozenset(c["feature"] + ":" + c["values"][0] for c in rule["conditions"]),
                                    row["original_section"], service["translation_ko"], service["easy_explanation"],
                                    {"literature_title": "欽定古今圖書集成", "chapter": "博物彙編/藝術典/第" + volume + "卷",
                                     "passage_no": row["name_zh"], "source_url": row["source_url"],
                                     "review_status": "reviewed" if row["independent_review"] else "draft",
                                     "passage_no_kind": "original_section_heading"},
                                    int(volume), order))
    return MemoryRepository(passages), {"total_records": len(catalog), "joined_records": len(services),
            "coverage": dict(Counter(r["rule_coverage"] for r in rules.values())),
            "independently_reviewed": sum(bool(r.get("independent_review")) for r in catalog),
            "eligible_records": len(passages), "excluded": dict(excluded), "integrity_errors": errors,
            "allow_draft": allow_draft, "real_model_predictions_used": None,
            "inference_execution": "not_performed_by_reference_loader",
            "usage_scope": "experimental_unreviewed" if allow_draft else "reviewed_only"}


POSTGRES_SCHEMA = """
CREATE TABLE literature (literature_id BIGINT PRIMARY KEY, title TEXT NOT NULL,
                         priority INTEGER NOT NULL DEFAULT 0);
CREATE TABLE passage (passage_id BIGINT PRIMARY KEY, literature_id BIGINT NOT NULL REFERENCES literature,
 chapter TEXT NOT NULL, passage_no TEXT NOT NULL, original_text TEXT NOT NULL,
 translation TEXT NOT NULL, commentary TEXT NOT NULL, passage_order INTEGER NOT NULL DEFAULT 0);
CREATE TABLE label (label_id BIGINT PRIMARY KEY, region TEXT NOT NULL, label_name TEXT NOT NULL,
 criteria TEXT NOT NULL, measurement_indicator TEXT, measurement_method TEXT NOT NULL);
CREATE TABLE label_passage (label_id BIGINT REFERENCES label, passage_id BIGINT REFERENCES passage,
 PRIMARY KEY(label_id, passage_id));
"""

POSTGRES_CANDIDATES = """
SELECT p.passage_id, p.original_text, p.translation, p.commentary,
       l.title AS literature_title, p.chapter, p.passage_no, l.priority, p.passage_order,
       ARRAY_AGG(lp.label_id ORDER BY lp.label_id) AS required_labels
FROM passage p JOIN literature l USING (literature_id)
JOIN label_passage lp USING (passage_id)
WHERE EXISTS (SELECT 1 FROM label_passage hit
              WHERE hit.passage_id=p.passage_id AND hit.label_id = ANY(%s::bigint[]))
GROUP BY p.passage_id, l.literature_id
ORDER BY p.passage_id
"""


class PostgresRepository:
    """Inject a DB-API connection; its caller owns lifecycle/transaction/readonly role."""
    def __init__(self, connection):
        self.connection = connection

    def load_candidates(self, label_ids: set[str]) -> list[Passage]:
        if not label_ids:
            return []
        with self.connection.cursor() as cursor:
            cursor.execute(POSTGRES_CANDIDATES, ([int(i) for i in sorted(label_ids)],))
            rows = cursor.fetchall()
        return [Passage(str(r[0]), frozenset(str(i) for i in r[9]), r[1], r[2], r[3],
                        {"literature_title": r[4], "chapter": r[5], "passage_no": r[6]}, r[7], r[8]) for r in rows]
