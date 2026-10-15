"""CPU-only checks for the historical-literature reference corpus. No faces, models or network."""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release import build_reference_corpus as brc  # noqa: E402

CORPUS = brc.CORPUS


def load_inputs():
    registry = brc.load_json(brc.REGISTRY)
    manifest = brc.load_json(brc.SNAPSHOT_MANIFEST)
    return registry, manifest, brc.load_label_vocabulary()


class SnapshotTests(unittest.TestCase):
    def test_snapshots_match_manifest_and_pinned_revisions(self):
        registry, manifest, _ = load_inputs()
        for page in [p for p in registry["pages"] if p["kind"] != "transcription"]:
            entry = manifest["pages"][page["page_id"]]
            raw = (brc.SNAPSHOTS / entry["snapshot_file"]).read_bytes()
            self.assertEqual(brc.sha256_bytes(raw), entry["snapshot_sha256"], page["page_id"])
            self.assertEqual(entry["revid"], page["revid"])
            self.assertIn(f"oldid={page['revid']}", entry["url"])
            self.assertTrue(entry["url"].startswith("https://zh.wikisource.org/"))

    def test_scan_pdfs_match_commons_hashes(self):
        import hashlib
        registry, _, _ = load_inputs()
        self.assertEqual({s["source_id"] for s in registry["scan_sources"]},
                         {"nlc511_63880", "nlc511_70286", "nlc416_42702"})
        for scan in registry["scan_sources"]:
            raw = (brc.BASE / scan["snapshot_file"]).read_bytes()
            self.assertEqual(hashlib.sha1(raw).hexdigest(), scan["commons_sha1"], scan["source_id"])
            self.assertEqual(brc.sha256_bytes(raw), scan["sha256"], scan["source_id"])
            self.assertTrue(scan["url"].startswith("https://upload.wikimedia.org/wikipedia/commons/"))

    def test_scan_transcriptions_complete_and_valid(self):
        registry, _, _ = load_inputs()
        for scan in registry["scan_sources"]:
            files = sorted((brc.TRANSCRIPTIONS / scan["source_id"]).glob("*.json"))
            self.assertTrue(files, scan["source_id"])
            for f in files:
                doc = brc.load_json(f)
                self.assertEqual(brc.check_transcription(doc, registry), [], f.name)
                self.assertIn("not human-proofread", doc["transcriber"])
            t = brc.load_transcription(registry, scan["source_id"])
            self.assertEqual(t["missing_pages"], [], scan["source_id"])
            self.assertEqual(len(t["pages"]), scan["pages"])

    def test_every_scan_page_marker_survives_extraction(self):
        registry, manifest, _ = load_inputs()
        for page in [p for p in registry["pages"] if p["kind"] != "transcription"]:
            raw = (brc.SNAPSHOTS / manifest["pages"][page["page_id"]]["snapshot_file"]).read_text(encoding="utf-8")
            blocks = brc.extract_blocks(raw, page["kind"])
            self.assertEqual(sum(len(b["markers"]) for b in blocks), raw.count("data-page-name="), page["page_id"])
            text = "\n".join(b["text"] for b in blocks)
            self.assertNotIn("CC BY-SA 4.0", text)  # licence banner is not classical text

    def test_every_literature_has_pages_and_unique_priority(self):
        registry, _, _ = load_inputs()
        lits = registry["literature"]
        self.assertEqual(len({l["literature_id"] for l in lits}), len(lits))
        self.assertEqual(len({l["literature_priority"] for l in lits}), len(lits))
        for l in lits:
            self.assertTrue(any(p["literature_id"] == l["literature_id"] for p in registry["pages"]))
        self.assertIn("liuzhuang_xiangfa", {l["literature_id"] for l in lits})
        self.assertTrue(any(p["literature_id"] == "liuzhuang_xiangfa" and p["kind"] == "transcription" for p in registry["pages"]))


class VocabularyTests(unittest.TestCase):
    def test_release_vocabulary(self):
        vocab = brc.load_label_vocabulary()
        for label in ("forehead_height:high", "forehead_width:wide", "ear_length:long", "ear_width:narrow",
                      "under_eye:visible", "philtrum_definition:faint", "brow_length:long", "nose_bridge:high"):
            self.assertIn(label, vocab["labels"])
        declared = brc.load_json(brc.PENDING_LABELS)["features"]
        self.assertEqual(set(vocab["pending"]), {f"{feature}:{cls}" for feature, classes in declared.items() for cls in classes})
        for label in ("pigmented_marks:absent", "pigmented_marks:present"):
            self.assertIn(label, vocab["labels"] | vocab["pending"])
        if "pigmented_marks" not in declared:
            ready = brc.load_json(brc.BASE.parent / "extended_ready.json")
            self.assertIs(ready["ready"], True)
            self.assertEqual(ready["registry_sha256"], brc.sha256_bytes(brc.TRAINING_REGISTRY.read_bytes()))
            registry = brc.load_json(brc.TRAINING_REGISTRY)
            self.assertTrue(all(n >= 2 for n in registry["class_support"]["train"]["pigmented_marks"]["counts"]))
        self.assertNotIn("brow_glabella:present", vocab["labels"])
        self.assertIn("nose_dorsum", vocab["withheld"])


class DefinitionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = brc.load_definition_audit()

    def exact(self, label, evidence):
        return brc.audit_condition({"region": "brow", "evidence": evidence, "observable_ko": "x",
                                    "status": "exact", "label": label}, self.audit)

    def test_literal_relations_not_guaranteed_by_quantile_classes(self):
        self.assertIn("brow_length", self.audit["quantile_features"])
        for label, ev in [("brow_length:short", "眉短不及目"), ("brow_length:long", "眉長過目"),
                          ("brow_eye_gap:close", "眉壓眼"), ("brow_eye_gap:wide", "眉高居額")]:
            out = self.exact(label, ev)
            self.assertEqual(out["status"], "approximate", ev)
            self.assertIn(label, out["candidate_labels"])
            self.assertNotIn("label", out)
        self.assertEqual(self.exact("brow_length:long", "眉長")["status"], "exact")

    def test_prevalence_does_not_change_literal_meaning_but_compounds_do(self):
        self.assertEqual(self.exact("brow_tail:descending", "尾垂")["status"], "exact")
        self.assertEqual(self.exact("eye_tilt:up", "目尾朝天")["status"], "exact")
        self.assertEqual(self.exact("nose_bridge:high", "山根連額起")["status"], "approximate")
        self.assertEqual(self.exact("chin_shape:rounded", "地閣豐圓")["status"], "approximate")
        self.assertEqual(self.exact("mouth_corner:down", "兩角下垂")["status"], "exact")
        self.assertEqual(self.exact("nose_bridge:high", "山根高")["status"], "exact")
        self.assertEqual(self.exact("pigmented_marks:absent", "面無痣")["status"], "approximate")


    def test_lip_direction_rim_and_individual_thickness_are_not_other_targets(self):
        for label, evidence in [("lip_projection:protruded", "脣掀"),
                                ("lip_bow:pronounced", "口有稜"),
                                ("lip_thickness:full", "上下俱厚")]:
            with self.subTest(label=label):
                self.assertEqual(self.exact(label, evidence)["status"], "approximate")
        self.assertEqual(self.exact("lip_thickness:full", "脣厚")["status"], "exact")


class TranscriptionAgreementTests(unittest.TestCase):
    def test_agreement_folds_punctuation_and_glyph_variants(self):
        section = {"heading": "(whole transcription)", "start_page": None}
        check = [("lz42702", section, brc.variant_key("眉長過目，兄弟五六。爲人"))]
        self.assertEqual(brc.transcription_agreement("眉長過目兄弟五六為人", check)["status"],
                         "agrees_with_check_witness")
        self.assertEqual(brc.transcription_agreement("眉短過目", check)["status"], "not_found_in_check_witness")
        self.assertEqual(brc.transcription_agreement("眉長", [])["status"], "no_check_witness_transcribed")


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.vocab = load_inputs()
        cls.corpus = brc.load_json(CORPUS)
        cls.passages = cls.corpus["passages"]

    def test_rebuild_is_reproducible(self):
        rebuilt, report = brc.build(self.registry, self.manifest, brc.load_annotations(), self.vocab)
        self.assertTrue(report["ok"], report.get("errors"))
        self.assertEqual(json.dumps(rebuilt, ensure_ascii=False, sort_keys=True),
                         json.dumps(self.corpus, ensure_ascii=False, sort_keys=True))

    def test_private_archive_preserves_every_nonempty_source_section(self):
        source = brc.load_snapshot_sections(self.registry, self.manifest)
        expected = {(pid, sec["heading"], sec["occurrence"]): sec["text"]
                    for pid, sections in source.items() for sec in sections if sec["text"].strip()}
        actual = self.corpus["source_sections"]
        self.assertEqual(len({a["section_id"] for a in actual}), len(actual))
        self.assertEqual({(a["page_id"], a["heading"], a["occurrence"]): a["original_text"] for a in actual}, expected)
        for row in actual:
            self.assertFalse(row["service_eligible"])
            self.assertEqual(row["annotation_status"], "source_archive_unreviewed")
            self.assertEqual(brc.sha256_bytes(row["original_text"].encode("utf-8")), row["text_sha256"])
            self.assertNotIn("translation", row)
        pages = {p["page_id"] for p in self.registry["pages"]}
        self.assertTrue({f"gj{n}" for n in range(631,645)} <= pages)
        self.assertTrue({f"xf{n:02}" for n in range(17)} <= pages)
        self.assertTrue({f"tq{n}" for n in range(1,7)} <= pages)

    def test_validator_passes(self):
        report = brc.validate_corpus_document(self.corpus, self.vocab)
        self.assertTrue(report["ok"], report["errors"])
        self.assertGreater(report["eligible"], 0)
        self.assertGreater(report["excluded"], report["eligible"])

    def test_schema_fields(self):
        self.assertEqual(self.corpus["schema_version"], brc.SCHEMA_VERSION)
        for p in self.passages:
            for field in ("passage_id", "literature_id", "original_text", "translation", "commentary",
                          "source", "required_labels", "literature_priority", "passage_order",
                          "annotation_status", "service_eligible"):
                self.assertIn(field, p)
            for field in ("title", "url", "chapter", "locator"):
                self.assertTrue(p["source"][field].strip(), (p["passage_id"], field))
            self.assertIsInstance(p["passage_id"], str)

    def test_referential_integrity_and_uniqueness(self):
        lit = {s["literature_id"] for s in self.corpus["sources"]}
        ids = [p["passage_id"] for p in self.passages]
        self.assertEqual(len(ids), len(set(ids)))
        for p in self.passages:
            self.assertIn(p["literature_id"], lit)
        keys = [(p["literature_id"], p["passage_order"]) for p in self.passages]
        self.assertEqual(len(keys), len(set(keys)))

    def test_original_text_is_verbatim_in_snapshot(self):
        sections = brc.load_snapshot_sections(self.registry, self.manifest)
        for p in self.passages:
            src = p["source"]
            page_id = src["page_id"]
            section = next(s for s in sections[page_id] if s["heading"] == src["section_heading"]
                           and s["occurrence"] == src["section_occurrence"])
            a, b = src["char_span"]
            self.assertEqual(brc.squash(section["text"][a:b]), brc.squash(p["original_text"]), p["passage_id"])
            for c in p["observable_conditions"]:
                self.assertIn(brc.squash(c["evidence"]), brc.squash(p["original_text"]))

    def test_eligibility_invariants(self):
        labels = self.vocab["labels"]
        for p in self.passages:
            if p["service_eligible"]:
                self.assertTrue(p["required_labels"])
                self.assertEqual(p["exclusion_reasons"], [])
                self.assertEqual(p["pending_labels"], [])
                self.assertTrue(all(c["status"] == "exact" for c in p["observable_conditions"]))
                self.assertFalse(set(p["claim_topics"]) & set(brc.SENSITIVE_TOPICS), p["passage_id"])
                feats = [l.split(":")[0] for l in p["required_labels"]]
                self.assertEqual(len(feats), len(set(feats)))
            else:
                self.assertTrue(p["exclusion_reasons"], p["passage_id"])
            for l in p["required_labels"]:
                self.assertIn(l, labels)
            for l in p["pending_labels"]:
                self.assertNotIn(l, labels)
                self.assertNotIn(l, p["required_labels"])

    def test_scan_transcribed_passages_are_cross_checked(self):
        scans = [p for p in self.passages if p["source"].get("scan_source_id")]
        for p in scans:
            src = p["source"]
            self.assertEqual(src["transcription_status"], "ai_visual_transcription_not_human_proofread")
            self.assertIn("#page=", src["url"])
            self.assertGreaterEqual(src["pdf_page_end"], src["pdf_page"])
            if p["service_eligible"]:
                self.assertEqual(src["transcription_check"]["status"], "agrees_with_check_witness")
                self.assertNotIn("〓", p["original_text"])
            elif src["transcription_check"]["status"] != "agrees_with_check_witness":
                self.assertIn("TRANSCRIPTION_UNCONFIRMED", p["exclusion_reasons"])
        summary = {s["source_id"]: s for s in self.corpus["scan_transcriptions"]}
        for s in summary.values():
            self.assertEqual(s["missing_pages"], [])
            self.assertEqual(s["transcription_status"], "ai_visual_transcription_not_human_proofread")

    def test_eligible_passages_survive_definition_audit(self):
        audit = brc.load_definition_audit()
        for p in self.passages:
            if not p["service_eligible"]:
                continue
            for c in p["observable_conditions"]:
                self.assertEqual(brc.audit_condition(dict(c), audit)["status"], "exact", p["passage_id"])
                if c["label"].split(":")[0] in audit["quantile_features"]:
                    self.assertIsNone(brc.LITERAL_RELATION.search(c["evidence"]), p["passage_id"])
            self.assertNotRegex(p["commentary"], r"pending|학습·검증 전|아직 학습|서비스 연결에서 제외")

    def test_no_human_review_claimed(self):
        self.assertIn("none", self.corpus["annotation_policy"]["human_review"])
        for p in self.passages:
            self.assertEqual(p["annotation_status"], "ai_annotated_unreviewed")
            self.assertNotIn("당신", p["translation"] + p["commentary"])

    def test_one_eligible_copy_per_text(self):
        seen = {}
        for p in self.passages:
            if p["service_eligible"]:
                key = brc.text_key(p["original_text"])
                self.assertNotIn(key, seen, (p["passage_id"], seen.get(key)))
                seen[key] = p["passage_id"]

    def test_complete_sections_are_fully_covered(self):
        sections = brc.load_snapshot_sections(self.registry, self.manifest)
        coverage = {(c["page_id"], c["heading"]): c["coverage"]
                    for c in brc.section_char_coverage(sections, self.passages)}
        for path in sorted(brc.ANNOTATIONS.glob("*.json")):
            doc = brc.load_json(path)
            for heading in doc.get("sections_complete", []):
                self.assertEqual(coverage.get((doc["page_id"], heading)), 1.0, (path.name, heading))

    def test_db_worker_contract(self):
        if importlib.util.find_spec("psycopg") is None:
            self.skipTest("psycopg not installed")
        from scripts.release.setup_database import validate_corpus
        sources, prepared = validate_corpus(copy.deepcopy(self.corpus))
        self.assertEqual(len(prepared), len(self.passages))

    def test_pilot_selector_accepts_eligible_passages(self):
        from facework.pilot_contract import Decision, Passage, PassageSelector
        eligible = [p for p in self.passages if p["service_eligible"]]
        target = eligible[0]
        decisions = [Decision(f"item-{i}", self.vocab["labels"][l]["part"] or "face", "CONFIRMED", l, l, .9,
                              None, None, "fixture", (), "classification")
                     for i, l in enumerate(target["required_labels"])]
        passages = [Passage(p["passage_id"], frozenset(p["required_labels"]), p["original_text"],
                            p["translation"], p["commentary"], p["source"], p["literature_priority"],
                            p["passage_order"]) for p in eligible]
        only = PassageSelector().select(decisions, [p for p in passages if p.passage_id == target["passage_id"]])
        self.assertEqual([s["passage_id"] for s in only], [target["passage_id"]])
        selected = PassageSelector().select(decisions, passages)
        self.assertLessEqual(len(selected), 5)
        for s in selected:
            labels = next(p for p in eligible if p["passage_id"] == s["passage_id"])["required_labels"]
            self.assertTrue(set(labels) <= set(target["required_labels"]))


class NegativeBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.vocab = load_inputs()
        cls.base = [dict(r, _file="xf01_eye.json", page_id="xf01")
                    for r in brc.load_json(brc.ANNOTATIONS / "xf01_eye.json")["passages"]]

    def rows(self):
        return copy.deepcopy(self.base)

    def by_id(self, corpus, pid):
        return next(p for p in corpus["passages"] if p["passage_id"] == pid)

    def test_baseline_exemplar(self):
        corpus, report = brc.build(self.registry, self.manifest, self.rows(), self.vocab)
        self.assertTrue(report["ok"], report.get("errors"))
        p = self.by_id(corpus, "XF01-eye-006")
        self.assertTrue(p["service_eligible"])
        self.assertEqual(p["required_labels"], ["eye_tilt:up"])
        self.assertEqual(p["literature_id"], "xufu_xiangfa")

    def test_invented_quotation_rejected(self):
        rows = self.rows()
        rows[5]["original_text"] = "目尾朝天，福祿綿綿長。"
        _, report = brc.build(self.registry, self.manifest, rows, self.vocab)
        self.assertFalse(report["ok"])

    def test_evidence_must_be_inside_quotation(self):
        rows = self.rows()
        rows[5]["conditions"][0]["evidence"] = "眼尾上"
        _, report = brc.build(self.registry, self.manifest, rows, self.vocab)
        self.assertFalse(report["ok"])

    def test_non_canonical_exact_label_rejected(self):
        rows = self.rows()
        rows[5]["conditions"][0]["label"] = "eye_tilt:skyward"
        _, report = brc.build(self.registry, self.manifest, rows, self.vocab)
        self.assertFalse(report["ok"])

    def test_duplicate_passage_id_rejected(self):
        rows = self.rows()
        rows[1]["passage_id"] = rows[0]["passage_id"]
        _, report = brc.build(self.registry, self.manifest, rows, self.vocab)
        self.assertFalse(report["ok"])

    def test_sensitive_claim_excluded(self):
        rows = self.rows()
        rows[5]["claim_topics"] = ["death_lifespan"]
        corpus, report = brc.build(self.registry, self.manifest, rows, self.vocab)
        p = self.by_id(corpus, "XF01-eye-006")
        self.assertFalse(p["service_eligible"])
        self.assertTrue(any(r.startswith("SENSITIVE_HISTORICAL_CLAIM") for r in p["exclusion_reasons"]))

    def test_non_exact_condition_excluded_but_preserved(self):
        rows = self.rows()
        rows[5]["conditions"][0].update(status="approximate", label=None, candidate_labels=["eye_tilt:up"],
                                        reason="test")
        rows[5]["conditions"][0].pop("label")
        corpus, _ = brc.build(self.registry, self.manifest, rows, self.vocab)
        p = self.by_id(corpus, "XF01-eye-006")
        self.assertFalse(p["service_eligible"])
        self.assertEqual(p["required_labels"], [])
        self.assertEqual(p["observable_conditions"][0]["candidate_labels"], ["eye_tilt:up"])

    def test_pending_label_kept_out_of_required_labels(self):
        vocab = copy.deepcopy(self.vocab)
        vocab["pending"]["under_eye:visible"] = vocab["labels"].pop("under_eye:visible")
        corpus, _ = brc.build(self.registry, self.manifest, self.rows(), vocab)
        p = self.by_id(corpus, "XF01-eye-015")
        self.assertEqual(p["pending_labels"], ["under_eye:visible"])
        self.assertEqual(p["required_labels"], [])
        self.assertFalse(p["service_eligible"])

    def test_contradictory_classes_excluded(self):
        rows = self.rows()
        rows[9]["conditions"][1].update(label="eye_size:small", evidence="細")
        corpus, _ = brc.build(self.registry, self.manifest, rows, self.vocab)
        p = self.by_id(corpus, "XF01-eye-010")
        self.assertIn("CONTRADICTORY_CLASSES_FOR_ONE_FEATURE", p["exclusion_reasons"])

    def test_duplicate_text_only_first_kept(self):
        rows = self.rows()
        dup = copy.deepcopy(rows[5])
        dup["passage_id"] = "XF01-eye-dup"
        corpus, report = brc.build(self.registry, self.manifest, rows + [dup], self.vocab)
        self.assertTrue(report["ok"], report.get("errors"))
        flags = sorted((p["passage_id"], p["service_eligible"]) for p in corpus["passages"]
                       if p["passage_id"] in ("XF01-eye-006", "XF01-eye-dup"))
        self.assertEqual(flags, [("XF01-eye-006", True), ("XF01-eye-dup", False)])


if __name__ == "__main__":
    unittest.main()
