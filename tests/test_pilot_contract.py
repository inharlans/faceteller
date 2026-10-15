"""CPU-only contract tests: generated PNGs/observations are fixtures, not face inference."""
import asyncio
import builtins
import json
import math
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from facework.pilot_contract import (
    AnalysisPipeline, CameraState, ContractError, DecisionPolicy, DIRECTIONS,
    InputLimits, InputSet, MemoryMultipart, MemoryRepository, Observation, Passage,
    PassageSelector, PostgresRepository, POSTGRES_CANDIDATES, STAGES, build_result,
    create_app, load_real_repository, receive_inputs, result_destination,
)

ROOT = Path(__file__).resolve().parents[1]


def classified(item="eye", region="eye", label="eye:wide", confidence=.9, **kw):
    return Observation(item, region, "classification", label, label, confidence,
                       calibrated=True, basis="fixture classifier evidence", **kw)


def measured(item="depth", region="nose", label="depth:high", score=.01, **kw):
    return Observation(item, region, "depth", label, label,
                       measurement={"indicator": "relative_projection", "value": .12, "unit": "ratio"},
                       measurement_score=score, quality_passed=True, boundaries_validated=True,
                       basis="fixture geometric measurement; validated boundary flag is simulated", **kw)


def passage(pid="1", labels=("eye:wide",), priority=0, order=0):
    return Passage(pid, frozenset(labels), "fixture original", "fixture translation", "fixture commentary",
                   {"literature_title": "fixture source", "chapter": "fixture chapter", "passage_no": pid},
                   priority, order)


def png(size=(8, 8)):
    out = BytesIO()
    Image.new("RGB", size, (15, 70, 125)).save(out, "PNG")
    return out.getvalue()


def multipart(mode="upload", images=None):
    images = images if images is not None else {d: png() for d in DIRECTIONS}
    return [(d, (d + ".png", raw, "image/png")) for d, raw in images.items()] + [("input_mode", (None, mode))]


class FixtureModules:
    def __init__(self, observations=None, depth=None, failure=None):
        self.observations = [classified()] if observations is None else observations
        self.depth_items = [] if depth is None else depth
        self.failure, self.calls, self.inputs, self.context = failure, [], None, None
        self.raw_buffers = []

    async def validate(self, inputs):
        self.inputs, self.raw_buffers = inputs, list(inputs.images.values())
        self.calls.append("validate")
        if self.failure == "face":
            raise ContractError("FACE_NOT_FOUND", "fixture: no face", 422, "left_oblique")
        if self.failure == "preprocess":
            raise ContractError("PREPROCESS_FAILED", "fixture: preprocess failure", 422)

    async def align(self, inputs, context):
        self.calls.append("align")
        self.context = context
        context["aligned"] = object()

    async def extract(self, inputs, context):
        self.calls.append("extract")
        context["detections"] = {d: "fixture landmarks" for d in DIRECTIONS}

    async def classify(self, context):
        self.calls.append("classify")
        if self.failure == "inference":
            raise RuntimeError("private image information MUST NOT LEAK")
        return self.observations

    async def estimate(self, inputs, context):
        self.calls.append("depth")
        assert set(inputs.images) == set(DIRECTIONS)
        assert set(context["detections"]) == set(DIRECTIONS)
        return self.depth_items


def fixture_app(modules=None, repository=None, limits=None):
    modules = modules or FixtureModules()
    repository = repository if repository is not None else MemoryRepository([passage()])
    return create_app(lambda: AnalysisPipeline(modules, modules, modules, repository), limits), modules


def events(response):
    return [json.loads(line) for line in response.content.splitlines()]


class RetainingService:
    """Simulates runtime allocations made before analyze's finally can run."""
    def __init__(self, validation_error=None, cleanup_error=False):
        self.validation_error, self.cleanup_error = validation_error, cleanup_error
        self.images, self.detections, self.close_calls, self.analyze_started = [], {}, 0, False

    async def validate(self, inputs):
        self.inputs, self.raw_buffers = inputs, list(inputs.images.values())
        self.image = Image.open(BytesIO(inputs.images["front"]))
        self.image.load()
        self.images.append(self.image)
        self.detections["front"] = object()
        if self.validation_error is not None:
            raise self.validation_error

    async def analyze(self, inputs):
        self.analyze_started = True
        try:
            yield {"type": "result", "data": build_result([], MemoryRepository([]))}
        finally:
            self.close()

    def close(self):
        self.close_calls += 1
        for image in self.images:
            image.close()
        self.images.clear()
        self.detections.clear()
        if self.cleanup_error:
            raise RuntimeError("fixture cleanup failure")


class DecisionTests(unittest.TestCase):
    def test_default_threshold_boundary_and_hold_label_removed(self):
        items = DecisionPolicy().apply([classified(confidence=.60), classified("b", confidence=.5999)])
        self.assertEqual([i.status for i in items], ["CONFIRMED", "HELD"])
        self.assertIsNone(items[1].label_id)
        self.assertIsNone(items[1].label_name)

    def test_binary_and_multiclass_policy_floors(self):
        for classes, threshold in ((2, .599), (3, .499), (3, float("nan")), (3, 1.01)):
            with self.subTest(classes=classes, threshold=threshold), self.assertRaises(ValueError):
                DecisionPolicy({"eye": threshold}).apply([classified(class_count=classes)])
        self.assertEqual(DecisionPolicy({"eye": .5}).apply([classified(confidence=.5)])[0].status, "CONFIRMED")

    def test_uncalibrated_occluded_and_invalid_confidence_are_held(self):
        for o in [replace(classified(), calibrated=False), classified(hold_reasons=("OCCLUDED",)),
                  *[classified(confidence=c) for c in (None, float("nan"), float("inf"), -1, 1.01)]]:
            with self.subTest(observation=o):
                result = build_result(DecisionPolicy().apply([o]), MemoryRepository([]))
                self.assertEqual(result["status"], "REINPUT_REQUIRED")
                json.dumps(result, allow_nan=False)

    def test_depth_score_separate_and_quality_boundary_failures(self):
        item = DecisionPolicy().apply([replace(measured(score=.07), confidence=.99)])[0]
        self.assertIsNone(item.confidence)
        self.assertEqual(item.measurement_score, .07)
        self.assertEqual(item.measurement_score_kind, "geometry_membership_not_calibrated_accuracy")
        for o in [replace(measured(), quality_passed=False), replace(measured(), boundaries_validated=False),
                  replace(measured(), measurement={"value": float("nan"), "unit": "ratio", "indicator": "x"})]:
            self.assertEqual(DecisionPolicy().apply([o])[0].status, "HELD")
        for score in (-1, 2, float("nan")):
            d = DecisionPolicy().apply([measured(score=score)])[0]
            self.assertIsNone(d.measurement_score)
            self.assertEqual(d.status, 'HELD')
            self.assertIn('DEPTH_SCORE_UNAVAILABLE', d.hold_reasons)

    def test_no_confirmed_skips_repository_but_no_passages_is_normal(self):
        class MustNotQuery:
            def load_candidates(self, labels):
                raise AssertionError("should not query")
        empty = build_result(DecisionPolicy().apply([classified(confidence=.1)]), MustNotQuery())
        normal = build_result(DecisionPolicy().apply([classified()]), MemoryRepository([]))
        partial = build_result(DecisionPolicy().apply([classified(), replace(measured(), quality_passed=False)]), MemoryRepository([]))
        self.assertEqual((empty["status"], normal["status"], partial["status"]), ("REINPUT_REQUIRED", "COMPLETE", "PARTIAL"))
        for mode in ("camera", "upload"):
            self.assertEqual(result_destination(mode, empty), mode)
            self.assertEqual(result_destination(mode, normal), "result")
        self.assertEqual(normal["literature_status"], "NONE")


class SelectorTests(unittest.TestCase):
    def test_and_all_conditions_returned_from_repository(self):
        p = passage(labels=("eye:wide", "depth:high"))
        repo = MemoryRepository([p])
        candidates = repo.load_candidates({"eye:wide"})
        self.assertEqual(candidates[0].required_labels, p.required_labels)
        self.assertEqual(PassageSelector().select(DecisionPolicy().apply([classified()]), candidates), [])
        held = DecisionPolicy().apply([classified(), replace(measured(), quality_passed=False)])
        self.assertEqual(PassageSelector().select(held, candidates), [])

    def test_g_is_unweighted_classification_geometric_mean_only(self):
        items = DecisionPolicy().apply([classified(confidence=.64), classified("b", "brow", "b:arched", .81), measured()])
        p = passage(labels=("eye:wide", "b:arched", "depth:high"))
        result = PassageSelector('classification_only').select(items, [p, p])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["G"], .72)
        self.assertEqual(result[0]["score_mode"], "classification_only")
        self.assertEqual(set(result[0]["evidence_item_ids"]), {"eye", "b", "depth"})
        other = DecisionPolicy().apply([classified(confidence=.64), classified("b", "brow", "b:arched", .81), measured(score=.99)])
        self.assertEqual(PassageSelector('classification_only').select(other, [p])[0]["G"], result[0]["G"])

    def test_default_measurement_score_mode_and_missing_score_held(self):
        items = DecisionPolicy().apply([classified(confidence=.81), measured(score=.04)])
        p = passage(labels=("eye:wide", "depth:high"))
        self.assertAlmostEqual(PassageSelector("include_measurement_scores").select(items, [p])[0]["G"], .18)
        self.assertAlmostEqual(PassageSelector().select(items, [p])[0]['G'], .18)
        self.assertIsNone(items[1].confidence)
        missing = DecisionPolicy().apply([classified(), measured(score=None)])
        self.assertEqual(PassageSelector().select(missing, [p]), [])
        corrupted = [missing[0], replace(missing[1], status='CONFIRMED', label_id='depth:high')]
        with self.assertRaises(ValueError):
            PassageSelector().select(corrupted, [p])

    def test_depth_only_after_scored_and_ties_priority_order_id(self):
        items = DecisionPolicy().apply([classified(), measured()])
        ps = [passage("z", ("depth:high",), -100), passage("c", priority=1),
              passage("b", order=1), passage("a", order=1), passage("d", priority=5)]
        selected = PassageSelector('classification_only').select(items, reversed(ps))
        self.assertEqual([p["passage_id"] for p in selected], ["a", "b", "c", "z"])
        self.assertIsNone(selected[-1]["G"])

    def test_mixed_uses_all_features_in_denominator_and_returns_components(self):
        items=DecisionPolicy().apply([classified(confidence=.64), classified('b','brow','b:arched',.81), measured(score=.125)])
        p=passage(labels=('eye:wide','b:arched','depth:high'))
        result=build_result(items,MemoryRepository([p]))
        selected=result['passages'][0]
        self.assertAlmostEqual(selected['G'], (.64*.81*.125)**(1/3))
        self.assertEqual(selected['score_feature_count'],3)
        self.assertEqual(selected['rank'],1)
        self.assertEqual({c['kind'] for c in selected['score_components']},{'classification','depth'})
        self.assertEqual(result['ranking_policy']['score_mode'],'include_measurement_scores')
        self.assertFalse(result['ranking_policy']['is_correctness_probability'])

    def test_depth_only_competes_by_score_and_mixed_can_drop_below_cnn(self):
        items=DecisionPolicy().apply([classified(confidence=.81), measured(score=.95),
                                     measured('d2','face','d2:high',score=.1)])
        ps=[passage('1'), passage('2',('depth:high',)), passage('3',('eye:wide','d2:high'))]
        out=PassageSelector().select(items,ps)
        self.assertEqual([p['passage_id'] for p in out],['2','1','3'])
        self.assertAlmostEqual(out[0]['G'],.95)
        self.assertAlmostEqual(out[2]['G'],math.sqrt(.081))

    def test_mixed_top_five_region_limits_and_held_condition(self):
        items=DecisionPolicy().apply([classified(confidence=.81), measured(score=.9),
                                     measured('d2','face','d2:high',score=.8),
                                     replace(measured('held','mouth','held:high',score=.99),quality_passed=False)])
        ps=[passage(str(i),('depth:high',),order=i) for i in range(4)]
        ps += [passage('4',('eye:wide',)),passage('5',('d2:high',)),passage('6',('eye:wide',)),passage('7',('held:high',))]
        out=PassageSelector().select(items,ps)
        self.assertEqual([p['passage_id'] for p in out],['0','1','2','4','6'])
        self.assertEqual([p['rank'] for p in out],[1,2,3,4,5])

    def test_zero_is_valid_score_without_log_domain_error(self):
        items=DecisionPolicy().apply([classified(),measured(score=0)])
        out=PassageSelector().select(items,[passage(labels=('eye:wide','depth:high'))])
        self.assertEqual(out[0]['G'],0)

    def test_multi_region_counts_each_region_once_and_skips_full_region(self):
        items = DecisionPolicy().apply([classified("e", "eye", "e"), classified("e2", "eye", "e2"),
                                        classified("n", "nose", "n"), classified("m", "mouth", "m")])
        ps = [passage(str(i), ("e", "e2", "n"), order=i) for i in range(4)]
        ps += [passage("4", ("m",), order=4), passage("5", ("m",), order=5), passage("6", ("m",), order=6)]
        self.assertEqual([p["passage_id"] for p in PassageSelector().select(items, ps)], ["0", "1", "2", "4", "5"])

    def test_numeric_passage_identifier_tie_break(self):
        items = DecisionPolicy().apply([classified()])
        selected = PassageSelector().select(items, [passage("10"), passage("2")])
        self.assertEqual([p["passage_id"] for p in selected], ["2", "10"])

    def test_conflicting_duplicate_rejected_and_empty_conditions_not_match(self):
        items = DecisionPolicy().apply([classified()])
        self.assertEqual(PassageSelector().select(items, [passage(labels=())]), [])
        with self.assertRaises(ValueError):
            PassageSelector().select(items, [passage(), replace(passage(), original_text="different")])


class ApiTests(unittest.TestCase):
    def assert_service_released(self, service):
        self.assertEqual(service.images, [])
        self.assertEqual(service.detections, {})
        self.assertEqual(service.inputs.images, {})
        self.assertTrue(all(not b for b in service.raw_buffers))
        with self.assertRaises(ValueError):
            service.image.getpixel((0, 0))

    def test_optional_close_releases_validation_allocations_even_when_cleanup_raises(self):
        for cleanup_error in (False, True):
            with self.subTest(cleanup_error=cleanup_error):
                service = RetainingService(ContractError("FACE_NOT_FOUND", "fixture partial validation", 422), cleanup_error)
                with TestClient(create_app(lambda: service)) as client:
                    if cleanup_error:
                        with self.assertRaisesRegex(RuntimeError, "fixture cleanup failure"):
                            client.post("/api/v1/analysis", files=multipart())
                    else:
                        response = client.post("/api/v1/analysis", files=multipart())
                        self.assertEqual(response.status_code, 422)
                        self.assertEqual(events(response)[0]["code"], "FACE_NOT_FOUND")
                self.assert_service_released(service)
                self.assertEqual(service.close_calls, 1)
                self.assertFalse(service.analyze_started)

    def test_optional_close_on_response_start_failure_and_idempotent_normal_finish(self):
        from starlette.requests import ClientDisconnect
        async def fail_start(cleanup_error):
            service = RetainingService(cleanup_error=cleanup_error)
            app = create_app(lambda: service)
            body, content_type = encoded_body()
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            async def send(message):
                if message["type"] == "http.response.start":
                    raise OSError("fixture socket closed before generator starts")
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                     "method": "POST", "path": "/api/v1/analysis", "raw_path": b"/api/v1/analysis",
                     "query_string": b"", "scheme": "http", "server": ("test", 80), "client": ("test", 1),
                     "headers": [(b"content-type", content_type.encode())]}
            with self.assertRaises(RuntimeError if cleanup_error else ClientDisconnect):
                await app(scope, receive, send)
            self.assertFalse(service.analyze_started)
            self.assertEqual(service.close_calls, 1)
            self.assert_service_released(service)
        for cleanup_error in (False, True):
            with self.subTest(cleanup_error=cleanup_error):
                asyncio.run(fail_start(cleanup_error))
        service = RetainingService()
        with TestClient(create_app(lambda: service)) as client:
            response = client.post("/api/v1/analysis", files=multipart())
        self.assertEqual(events(response)[-1]["type"], "result")
        self.assertEqual(service.close_calls, 2)  # analyze.finally + response.finally
        self.assert_service_released(service)

    def test_real_testclient_ndjson_three_images_both_modes_and_cleanup(self):
        for mode in ("camera", "upload"):
            app, m = fixture_app()
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart(mode))
            lines = events(response)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "application/x-ndjson")
            self.assertEqual([e["stage"] for e in lines[:-1]], list(STAGES))
            self.assertEqual(lines[-1]["data"]["status"], "COMPLETE")
            self.assertEqual(m.calls, ["validate", "align", "extract", "classify", "depth"])
            self.assertEqual(m.inputs.images, {})
            self.assertTrue(all(not raw for raw in m.raw_buffers))
            self.assertEqual(m.context, {})
            self.assertNotIn("landmarks", response.text)

    def test_no_uploadfile_spooling_or_file_writes_even_over_1mb(self):
        # Valid uncompressed PNG > default Starlette spool threshold.
        buf = BytesIO()
        Image.new("RGB", (650, 650), (50, 70, 90)).save(buf, "PNG", compress_level=0)
        self.assertGreater(len(buf.getvalue()), 1024 * 1024)
        app, m = fixture_app()
        with TestClient(app) as client, ExitStack() as stack:
            for target in ("tempfile.SpooledTemporaryFile", "tempfile.TemporaryFile", "tempfile.NamedTemporaryFile",
                           "starlette.formparsers.SpooledTemporaryFile", "starlette.datastructures.UploadFile.__init__"):
                stack.enter_context(patch(target, side_effect=AssertionError("disk/spool path forbidden")))
            original_open = builtins.open
            def read_only(file, mode="r", *args, **kwargs):
                if any(flag in mode for flag in "wax+"):
                    raise AssertionError("file write forbidden")
                return original_open(file, mode, *args, **kwargs)
            stack.enter_context(patch("builtins.open", side_effect=read_only))
            response = client.post("/api/v1/analysis", files=multipart(images={d: buf.getvalue() for d in DIRECTIONS}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events(response)[-1]["type"], "result")

    def test_missing_direction_and_input_mode(self):
        app, m = fixture_app()
        with TestClient(app) as client:
            response = client.post("/api/v1/analysis", files=multipart(images={"front": png(), "left_oblique": png()}))
            self.assertEqual(response.status_code, 400)
            self.assertEqual(events(response)[0]["direction"], "right_oblique")
            for mode in ("", "bad"):
                response = client.post("/api/v1/analysis", files=multipart(mode))
                self.assertEqual(response.status_code, 400)
        self.assertEqual(m.calls, [])

    def test_duplicate_extra_mime_and_corrupt_inputs_rejected(self):
        app, m = fixture_app()
        cases = [multipart() + [("front", ("again.png", png(), "image/png"))],
                 multipart() + [("extra", (None, "bad"))],
                 [("front", ("face.png", png(), "image/jpeg")), *multipart()[1:]],
                 multipart(images={d: b"not image" if d == "left_oblique" else png() for d in DIRECTIONS})]
        with TestClient(app) as client:
            for files in cases:
                with self.subTest(files=len(files)):
                    response = client.post("/api/v1/analysis", files=files)
                    self.assertEqual(response.status_code, 415)
                    self.assertEqual(events(response)[-1]["code"], "INVALID_FORMAT")
        self.assertEqual(m.calls, [])

    def test_size_and_resolution_limits(self):
        for limits in (InputLimits(file_bytes=10), InputLimits(request_bytes=20), InputLimits(pixels=16)):
            app, m = fixture_app(limits=limits)
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            self.assertEqual(response.status_code, 413)
            self.assertEqual(m.calls, [])

    def test_face_preprocess_errors_pre_stream_and_cleanup(self):
        for failure, code in (("face", "FACE_NOT_FOUND"), ("preprocess", "PREPROCESS_FAILED")):
            app, m = fixture_app(FixtureModules(failure=failure))
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            self.assertEqual(response.status_code, 422)
            self.assertEqual(events(response)[0]["code"], code)
            self.assertEqual(m.inputs.images, {})

    def test_midstream_inference_and_repository_errors_are_not_empty_literature(self):
        class BrokenRepo:
            def load_candidates(self, ids):
                raise RuntimeError("private DB connection information")
        for modules, repo in ((FixtureModules(failure="inference"), None), (FixtureModules(), BrokenRepo())):
            app, m = fixture_app(modules, repo)
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(events(response)[-1]["code"], "SERVER_ERROR")
            self.assertFalse(any(e["type"] == "result" for e in events(response)))
            self.assertNotIn("private", response.text)
            self.assertEqual(m.inputs.images, {})
            self.assertEqual(m.context, {})

    def test_partial_no_literature_and_reinput_results(self):
        for observations, depth, expected in (([classified()], [replace(measured(), quality_passed=False)], "PARTIAL"),
                                               ([classified(confidence=.1)], [], "REINPUT_REQUIRED"),
                                               ([classified()], [], "COMPLETE")):
            app, _ = fixture_app(FixtureModules(observations, depth), MemoryRepository([]))
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            result = events(response)[-1]["data"]
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["passages"], [])

    def test_service_injection_required_and_premature_end_errors(self):
        with self.assertRaises(TypeError):
            create_app()
        class Incomplete:
            async def validate(self, inputs):
                self.inputs = inputs
            async def analyze(self, inputs):
                yield {"type": "progress", "stage": STAGES[0]}
        service = Incomplete()
        with TestClient(create_app(lambda: service)) as client:
            response = client.post("/api/v1/analysis", files=multipart())
        self.assertEqual(events(response)[-1]["code"], "SERVER_ERROR")
        self.assertEqual(service.inputs.images, {})

    def test_asgi_disconnect_cleans_input_and_pipeline_context(self):
        # TestClient buffers response bodies; direct ASGI disconnect is needed to
        # exercise actual streaming cancellation rather than merely closing a client.
        async def run():
            started, never = asyncio.Event(), asyncio.Event()
            class Slow(FixtureModules):
                async def classify(self, context):
                    started.set()
                    await never.wait()
            app, m = fixture_app(Slow())
            body, content_type = encoded_body()
            sent = False
            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await started.wait()
                return {"type": "http.disconnect"}
            messages = []
            async def send(message):
                messages.append(message)
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                     "method": "POST", "path": "/api/v1/analysis", "raw_path": b"/api/v1/analysis",
                     "query_string": b"", "scheme": "http", "server": ("test", 80), "client": ("test", 1),
                     "headers": [(b"content-type", content_type.encode())]}
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
            self.assertEqual(m.inputs.images, {})
            self.assertEqual(m.context, {})
            self.assertFalse(any(b'"type": "result"' in msg.get("body", b"") for msg in messages))
        asyncio.run(run())

    def test_send_failure_before_generator_starts_and_during_body(self):
        from starlette.requests import ClientDisconnect
        async def run(fail_at):
            app, m = fixture_app()
            body, content_type = encoded_body()
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            body_count = 0
            async def send(message):
                nonlocal body_count
                if message["type"] == "http.response.body":
                    body_count += 1
                if (fail_at == "start" and message["type"] == "http.response.start") or (fail_at == "body" and body_count == 2):
                    raise OSError("fixture disconnected socket")
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                     "method": "POST", "path": "/api/v1/analysis", "raw_path": b"/api/v1/analysis",
                     "query_string": b"", "scheme": "http", "server": ("test", 80), "client": ("test", 1),
                     "headers": [(b"content-type", content_type.encode())]}
            with self.assertRaises(ClientDisconnect):
                await app(scope, receive, send)
            self.assertEqual(m.inputs.images, {})
            self.assertTrue(all(not b for b in m.raw_buffers))
            if fail_at == "body":
                self.assertEqual(m.context, {})
        for fail_at in ("start", "body"):
            with self.subTest(fail_at=fail_at):
                asyncio.run(run(fail_at))


def encoded_body():
    import httpx
    request = httpx.Request("POST", "http://test/api/v1/analysis", files=multipart())
    return request.read(), request.headers["content-type"]


class MultipartTests(unittest.TestCase):
    def test_one_byte_chunks_and_truncation(self):
        body, content_type = encoded_body()
        parser = MemoryMultipart(content_type, InputLimits())
        for b in body:
            parser.feed(bytes([b]))
        inputs = parser.finish()
        parser.clear()
        self.assertEqual(set(inputs.images), set(DIRECTIONS))
        self.assertTrue(all(inputs.images.values()))
        inputs.clear()
        parser = MemoryMultipart(content_type, InputLimits())
        parser.feed(body[:-10])
        with self.assertRaises(ContractError):
            parser.finish()
        parser.clear()
        self.assertEqual(parser.parts, {})

    def test_actual_bytes_limit_even_without_content_length(self):
        async def run():
            body, content_type = encoded_body()
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request = Request({"type": "http", "headers": [(b"content-type", content_type.encode())]}, receive)
            with self.assertRaises(ContractError) as error:
                await receive_inputs(request, InputLimits(request_bytes=len(body) - 1))
            self.assertEqual(error.exception.status, 413)
        asyncio.run(run())

    def test_partial_receive_cancellation_clears_buffers(self):
        async def run():
            body, content_type = encoded_body()
            called = False
            async def receive():
                nonlocal called
                if not called:
                    called = True
                    return {"type": "http.request", "body": body[:300], "more_body": True}
                raise asyncio.CancelledError()
            parser = MemoryMultipart(content_type, InputLimits())
            request = Request({"type": "http", "headers": [(b"content-type", content_type.encode())]}, receive)
            with patch("facework.pilot_contract.MemoryMultipart", return_value=parser):
                with self.assertRaises(asyncio.CancelledError):
                    await receive_inputs(request, InputLimits())
            self.assertEqual(parser.parts, {})
            self.assertEqual(parser.current, bytearray())
        asyncio.run(run())


class CameraTests(unittest.TestCase):
    OK = {"face": True, "direction": True, "size": True, "brightness": True}

    def test_continuity_reset_stale_results_three_captures_single_submission(self):
        state = CameraState()
        state.permission(True)
        self.assertFalse(state.submit_once())
        token = state.token
        state.update(token, self.OK, 0)
        state.update(token, self.OK, .5)
        self.assertEqual(state.gauge, .5)
        state.update(token, {**self.OK, "brightness": False}, .7)
        self.assertEqual(state.gauge, 0)
        self.assertEqual(state.guidance, ("brightness",))
        state.update(token, self.OK, 1)
        self.assertEqual(state.update(token, self.OK, 2), "front")
        self.assertIsNone(state.update(token, self.OK, 100))
        for start, direction in ((3, "left_oblique"), (5, "right_oblique")):
            token = state.token
            state.update(token, self.OK, start)
            self.assertIsNone(state.update(token, self.OK, start - .5))
            self.assertEqual(state.update(token, self.OK, start + 1), direction)
        self.assertEqual(state.captured, list(DIRECTIONS))
        self.assertFalse(state.stream_open)
        self.assertTrue(state.submit_once())
        self.assertFalse(state.submit_once())

    def test_permission_denied_close_and_incomplete_conditions(self):
        state = CameraState()
        state.permission(False)
        self.assertEqual(state.state, "upload")
        self.assertFalse(state.stream_open)
        state = CameraState()
        state.permission(True)
        state.update(state.token, {"face": True}, 0)
        state.update(state.token, {"face": True}, 3)
        self.assertEqual(state.captured, [])
        token = state.token
        state.close()
        self.assertIsNone(state.update(token, self.OK, 10))
        self.assertFalse(state.stream_open)


class RealDataTests(unittest.TestCase):
    def test_103_join_text_integrity_review_limitations(self):
        repo, audit = load_real_repository(ROOT)
        self.assertEqual(audit["total_records"], 103)
        self.assertEqual(audit["integrity_errors"], [])
        self.assertEqual(audit["coverage"], {"full": 15, "partial": 65, "none": 23})
        self.assertEqual(audit["independently_reviewed"], 0)
        self.assertEqual(repo.passages, ())

    def test_real_text_synthetic_confirmed_labels_selection_and_api(self):
        repo, audit = load_real_repository(ROOT, allow_draft=True)
        self.assertGreater(len(repo.passages), 0)
        self.assertLessEqual(len(repo.passages), 15)
        schema = json.loads((ROOT / "data/label_schema_kface_v3.json").read_text(encoding="utf8"))["features"]
        catalog = {r["record_id"]: r for r in json.loads((ROOT / "data/interpretation_catalog_v1.json").read_text(encoding="utf8"))["records"]}
        # Test every eligible passage separately: a single person need not satisfy
        # contradictory labels used across all source clauses.
        for p in repo.passages:
            observations, depths = [], []
            for label in sorted(p.required_labels):
                feature = label.split(":")[0]
                region = schema[feature]["part"]
                if schema[feature]["source"] == "multiview":
                    depths.append(measured(feature, region, label))
                else:
                    observations.append(classified(feature, region, label))
            items = DecisionPolicy().apply(observations + depths)
            selected = PassageSelector().select(items, [p])
            self.assertEqual(selected[0]["original_text"], catalog[p.passage_id]["original_section"])
            for missing in p.required_labels:
                subset = [i for i in items if i.label_id != missing]
                self.assertEqual(PassageSelector().select(subset, [p]), [])
        p = repo.passages[0]
        observations = [classified(label.split(":")[0], "brow", label) for label in sorted(p.required_labels)]
        app, _ = fixture_app(FixtureModules(observations), repo)
        with TestClient(app) as client:
            response = client.post("/api/v1/analysis", files=multipart())
        self.assertEqual(events(response)[-1]["data"]["passages"][0]["source"]["review_status"], "draft")


class SqlContractTests(unittest.TestCase):
    def test_dbapi_query_is_parameterized_and_preserves_unmatched_conditions(self):
        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, query, args): self.query, self.args = query, args
            def fetchall(self): return [(1, "original", "translation", "commentary", "title", "chapter", "1", 0, 0, [11, 12])]
        cursor = Cursor()
        class Connection:
            def cursor(self): return cursor
        repo = PostgresRepository(Connection())
        rows = repo.load_candidates({"11"})
        self.assertEqual(cursor.query, POSTGRES_CANDIDATES)
        self.assertEqual(cursor.args, ([11],))
        self.assertEqual(rows[0].required_labels, frozenset({"11", "12"}))
        self.assertEqual(PassageSelector().select(DecisionPolicy().apply([classified(label="11")]), rows), [])

    def test_gpu_modules_not_imported(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("facework.analysis_v2", sys.modules)
        self.assertNotIn("facework.server", sys.modules)


if __name__ == "__main__":
    unittest.main()
