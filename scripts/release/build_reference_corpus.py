"""Build the FaceTeller historical-literature reference corpus. CPU only, no face data.

  .venv/Scripts/python.exe scripts/release/build_reference_corpus.py fetch      # pinned Wikisource revisions -> snapshots/
  .venv/Scripts/python.exe scripts/release/build_reference_corpus.py inventory  # snapshot sections -> section_inventory.json
  .venv/Scripts/python.exe scripts/release/build_reference_corpus.py build      # snapshots + annotations -> corpus + validation
  .venv/Scripts/python.exe scripts/release/build_reference_corpus.py validate   # re-check an existing corpus file

`build` and `validate` are offline: they read only the stored snapshots (sha256-checked
against snapshot_manifest.json), the hand-written annotations, and the canonical label
vocabulary data/label_schema_kface_v3.json. Every original_text must occur verbatim
(whitespace-insensitive) in its snapshot section, so no quotation can be invented.
Translations/commentary are AI-authored drafts and are labelled as such; nothing here
claims human review. Passages are historical text for display only: eligibility is a
morphology-matching decision, never a statement about a person.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "release" / "literature"
REGISTRY = BASE / "source_registry.json"
SNAPSHOTS = BASE / "snapshots"
SNAPSHOT_MANIFEST = SNAPSHOTS / "snapshot_manifest.json"
ANNOTATIONS = BASE / "annotations"
INVENTORY = BASE / "section_inventory.json"
CORPUS = BASE / "reference_corpus_v1.json"
VALIDATION = BASE / "validation_report.json"
LABEL_SCHEMA = ROOT / "data" / "label_schema_kface_v3.json"
PENDING_LABELS = BASE / "pending_labels.json"
TRAINING_REGISTRY = ROOT / "data" / "release" / "training_registry.json"
EXTENDED_SCHEMA = ROOT / "data" / "release" / "extended_schema.json"
LANDMARK_LABELS = ROOT / "data" / "release" / "landmark_labels.json"

SCHEMA_VERSION = "faceteller-reference-corpus-1"
ANNOTATION_STATUS = "ai_annotated_unreviewed"
USER_AGENT = "FaceTellerLiterature/1.0 (academic reference corpus; public-domain texts only)"
WIKISOURCE = "https://zh.wikisource.org"
CONDITION_STATUSES = ("exact", "approximate", "ambiguous", "unmapped", "unobservable")
# Historical claims the service must not attach to a matched user, even as quotation.
# Any lifespan claim (long or short) counts as health/fate inference and is sensitive.
# Predictions about one's own children/heirs (existence, number, sex) are fertility inferences.
SENSITIVE_TOPICS = ("criminality", "sexual_conduct", "health_disease", "death_lifespan",
                    "intelligence", "violence", "kinship_harm", "malice_deceit", "fertility_offspring")
CLAIM_TOPICS = SENSITIVE_TOPICS + ("wealth", "rank_office", "character", "fortune_general",
                                   "family_children", "marriage", "none", "other")
LEGACY_CATALOG = ROOT / "data" / "interpretation_catalog_v1.json"
REGIONS = ("eye", "brow", "nose", "mouth", "face", "philtrum", "ear", "forehead", "glabella",
           "under_eye", "cheekbone", "chin", "mole", "teeth", "tongue", "subject", "other")
MARK_OPEN, MARK_CLOSE = "", ""
WS = re.compile(r"[\s　​]+")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def squash(text: str) -> str:
    return WS.sub("", text)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def page_url(title: str, revid: int | None = None) -> str:
    if revid:
        return f"{WIKISOURCE}/w/index.php?title={urllib.parse.quote(title.replace(' ', '_'))}&oldid={revid}"
    return f"{WIKISOURCE}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


# ---------------------------------------------------------------- fetch

def fetch(registry: dict, *, refresh: bool = False, delay: float = 2.5) -> dict:
    from bs4 import BeautifulSoup, Comment

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    manifest = load_json(SNAPSHOT_MANIFEST) if SNAPSHOT_MANIFEST.exists() else {"pages": {}}
    for page in registry["pages"]:
        if page["kind"] == "transcription":
            continue  # image-only PDFs are pinned/downloaded separately, not Wikisource HTML
        path = SNAPSHOTS / f"{page['page_id']}.html"
        if path.exists() and not refresh and page["page_id"] in manifest["pages"]:
            continue
        url = page_url(page["title"], page["revid"])
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            status = response.status
        soup = BeautifulSoup(body, "html.parser")
        main = soup.select_one(".mw-parser-output")
        if main is None:
            raise RuntimeError(f"{page['page_id']}: no .mw-parser-output in {url}")
        for comment in main.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()
        fragment = str(main).encode("utf-8")
        path.write_bytes(fragment)
        manifest["pages"][page["page_id"]] = {
            "title": page["title"], "revid": page["revid"], "url": url, "http_status": status,
            "retrieved_at": time.strftime("%Y-%m-%d"), "response_sha256": sha256_bytes(body),
            "snapshot_file": path.name, "snapshot_sha256": sha256_bytes(fragment)}
        print(f"fetched {page['page_id']} {len(fragment)} bytes", flush=True)
        dump_json(SNAPSHOT_MANIFEST, manifest)
        time.sleep(delay)
    dump_json(SNAPSHOT_MANIFEST, manifest)
    return manifest


# ---------------------------------------------------------------- text extraction

def extract_blocks(html: str, kind: str) -> list[dict]:
    """Deterministic blocks: headings and paragraphs with scan-page markers.

    gj pages: one <p> per printed column group; internal newlines are scan line breaks
    and are removed. poem pages: <br> separates verse lines and is kept as newline.
    """
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".mw-parser-output") or soup
    # Wikisource chrome: edit links, navigation tables, licence banners (not classical text).
    for bad in main.select("style, script, .mw-editsection, table, .ws-noexport:not(.ws-pagenum), "
                           ".licenseContainer, .licensetpl"):
        bad.decompose()
    pages: list[str] = []

    def text_of(node) -> str:
        out = []
        for child in node.children:
            if isinstance(child, NavigableString):
                out.append(str(child) if kind == "poem" else str(child).replace("\n", ""))
            elif isinstance(child, Tag):
                classes = child.get("class") or []
                if "pagenum" in classes and child.get("data-page-name"):
                    pages.append(child["data-page-name"])
                    out.append(f"{MARK_OPEN}{len(pages) - 1}{MARK_CLOSE}")
                elif child.name == "br":
                    out.append("\n")
                elif child.name == "img":
                    out.append(child.get("alt", ""))
                else:
                    out.append(text_of(child))
        return "".join(out)

    blocks = []

    def add_para(raw: str):
        for line in raw.split("\n") if kind == "poem" else [raw]:
            heading = re.fullmatch(r"\s*(={2,4})\s*([^=]+?)\s*\1\s*(.*)", line, flags=re.S)
            if heading:  # unparsed wikitext heading left inline by the transcription
                blocks.append({"type": "heading", "text": heading.group(2), "markers": []})
                line = heading.group(3)
            markers, clean = [], []
            for part in re.split(f"({MARK_OPEN}\\d+{MARK_CLOSE})", line):
                if part.startswith(MARK_OPEN):
                    markers.append((len("".join(clean)), pages[int(part[1:-1])]))
                else:
                    clean.append(part)
            text = "".join(clean).strip("\n")
            if re.fullmatch(r"\s*<[^<>]*>\s*", text):  # 四庫 category tag like <子部,術數類,...>
                text = ""
            if text.strip() or markers:
                blocks.append({"type": "para", "text": text, "markers": markers})

    pending_marks: list[str] = []  # page markers standing between paragraphs

    def walk(node):
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if "pagenum" in (child.get("class") or []) and child.get("data-page-name"):
                pages.append(child["data-page-name"])
                pending_marks.append(f"{MARK_OPEN}{len(pages) - 1}{MARK_CLOSE}")
            elif child.name in ("h1", "h2", "h3", "h4", "h5"):
                blocks.append({"type": "heading", "text": text_of(child).strip(), "markers": []})
            elif child.name == "p":
                add_para("".join(pending_marks) + text_of(child))
                pending_marks.clear()
            elif child.name in ("div", "section", "dl", "dd", "blockquote", "center", "span"):
                walk(child)
    walk(main)
    return blocks


def sections_from_blocks(blocks: list[dict], chapter: str) -> list[dict]:
    """Group paragraphs under the nearest heading; carry the current scan page forward."""
    sections, current, page = [], None, None
    seen = Counter()
    for block in blocks:
        if block["type"] == "heading":
            for _, name in block["markers"]:  # a page can begin with a heading
                page = name
            seen[block["text"]] += 1
            current = {"heading": block["text"], "occurrence": seen[block["text"]],
                       "paragraphs": [], "scan_pages": [], "start_page": page}
            sections.append(current)
            continue
        if current is None:
            current = {"heading": chapter, "occurrence": 1, "paragraphs": [], "scan_pages": [],
                       "start_page": page}
            sections.append(current)
        offset = sum(len(p) + 1 for p in current["paragraphs"])
        if not current["paragraphs"] and current["start_page"] is None and block["markers"] and block["markers"][0][0] == 0:
            current["start_page"] = block["markers"][0][1]
        for pos, name in block["markers"]:
            current["scan_pages"].append([offset + pos, name])
            page = name
        current["paragraphs"].append(block["text"])
    for s in sections:
        s["text"] = "\n".join(s.pop("paragraphs"))
    return sections


TRANSCRIPTIONS = BASE / "transcriptions"
PAGE_TYPES = ("cover", "blank", "toc", "text", "figure", "colophon", "other")
BLOCK_TYPES = ("heading", "text", "figure_label")


def load_transcription(registry: dict, source_id: str) -> dict:
    """Merge the AI visual transcription files of one image-only scan, in PDF page order."""
    scan = next(s for s in registry["scan_sources"] if s["source_id"] == source_id)
    raw = (BASE / scan["snapshot_file"]).read_bytes()
    if sha256_bytes(raw) != scan["sha256"]:
        raise ValueError(f"{source_id}: PDF sha256 mismatch")
    files = sorted((TRANSCRIPTIONS / source_id).glob("*.json"))
    pages, digest = {}, hashlib.sha256()
    for f in files:
        doc = load_json(f)
        errors = check_transcription(doc, registry)
        if errors or doc["source_id"] != source_id:
            raise ValueError(f"{f.name}: {errors or 'source_id mismatch'}")
        for page in doc["pages"]:
            if page["pdf_page"] in pages:
                raise ValueError(f"{source_id}: page {page['pdf_page']} transcribed twice")
            pages[page["pdf_page"]] = dict(page, _file=f"transcriptions/{source_id}/{f.name}")
        digest.update(f.name.encode("utf-8") + b"\0" + f.read_bytes())
    return {"scan": scan, "pages": dict(sorted(pages.items())), "files": [f"transcriptions/{source_id}/{f.name}" for f in files],
            "sha256": digest.hexdigest(), "missing_pages": [n for n in range(1, scan["pages"] + 1) if n not in pages]}


def transcription_blocks(transcription: dict) -> list[dict]:
    blocks, carry = [], []
    for n, page in transcription["pages"].items():
        carry.append(f"pdf_page:{n}")
        for b in page.get("blocks", []):
            blocks.append({"type": "heading" if b["type"] == "heading" else "para", "text": b["text"],
                           "markers": [(0, m) for m in carry]})
            carry = []
    return blocks


def load_snapshot_sections(registry: dict, manifest: dict) -> dict[str, list[dict]]:
    out = {}
    for page in registry["pages"]:
        if page["kind"] == "transcription":
            t = load_transcription(registry, page["source_id"])
            out[page["page_id"]] = sections_from_blocks(transcription_blocks(t), page["chapter"])
            continue
        entry = manifest["pages"].get(page["page_id"])
        if entry is None:
            raise ValueError(f"{page['page_id']}: snapshot missing; run fetch")
        raw = (SNAPSHOTS / entry["snapshot_file"]).read_bytes()
        if sha256_bytes(raw) != entry["snapshot_sha256"]:
            raise ValueError(f"{page['page_id']}: snapshot sha256 mismatch")
        if entry["revid"] != page["revid"]:
            raise ValueError(f"{page['page_id']}: snapshot revid differs from registry")
        blocks = extract_blocks(raw.decode("utf-8"), page["kind"])
        out[page["page_id"]] = sections_from_blocks(blocks, page["chapter"])
    return out


def inventory(registry: dict, manifest: dict) -> dict:
    sections = load_snapshot_sections(registry, manifest)
    pages = []
    for page in registry["pages"]:
        pages.append({"page_id": page["page_id"], "literature_id": page["literature_id"],
                      "sections": [{"heading": s["heading"], "occurrence": s["occurrence"],
                                    "chars": len(squash(s["text"])), "start_page": s["start_page"],
                                    "text": s["text"]} for s in sections[page["page_id"]]]})
    return {"generated_from": "snapshots/snapshot_manifest.json", "pages": pages}


# ---------------------------------------------------------------- labels

def load_label_vocabulary(path: Path = LABEL_SCHEMA, *, training_registry: Path = TRAINING_REGISTRY,
                          extended_schema: Path = EXTENDED_SCHEMA, pending_path: Path = PENDING_LABELS) -> dict:
    """Release vocabulary: v3 schema + training-registry additions + confirmed extended features.

    Features listed in pending_labels.json stay pending (recorded, never service-eligible).
    """
    pending_features = load_json(pending_path)["features"] if pending_path.exists() else {}
    labels, pending, origins = {}, {}, {}

    def add(feature, spec, origin):
        if feature in origins:
            return
        origins[feature] = origin
        target = pending if feature in pending_features else labels
        for i, cls in enumerate(spec["classes"]):
            target[f"{feature}:{cls}"] = {
                "feature": feature, "class": cls, "part": spec.get("part"), "origin": origin,
                "measurement_source": spec.get("source"), "feature_ko": spec.get("ko"),
                "class_ko": (spec.get("classes_ko") or [None] * len(spec["classes"]))[i],
                "label_status": spec.get("label_status")}

    schema = load_json(path)
    for feature, spec in schema["features"].items():
        add(feature, spec, path.name)
    sources = [path.name]
    for extra in (training_registry, extended_schema):
        if extra.exists():
            sources.append(extra.name)
            for feature, spec in load_json(extra).get("features", {}).items():
                add(feature, spec, extra.name)
    for feature, classes in pending_features.items():
        if feature not in origins:
            add(feature, {"classes": classes}, pending_path.name)
    return {"version": schema.get("version"), "labels": labels, "pending": pending,
            "sources": sources, "feature_origins": dict(sorted(origins.items())),
            "withheld": sorted(f for f in schema.get("withheld_features", {}) if f not in origins)}


# ---------------------------------------------------------------- build

def locate(section: dict, original: str) -> tuple[int, int] | None:
    """Return (start, end) of original in section text, comparing without whitespace."""
    text = section["text"]
    target = squash(original)
    index_map = [i for i, ch in enumerate(text) if not WS.fullmatch(ch)]
    hay = "".join(text[i] for i in index_map)
    pos = hay.find(target)
    if pos < 0 or not target:
        return None
    if hay.find(target, pos + 1) >= 0:
        raise ValueError(f"original_text occurs more than once in section {section['heading']!r}: {original[:20]}")
    return index_map[pos], index_map[pos + len(target) - 1] + 1


def scan_page_at(section: dict, offset: int) -> str | None:
    page = section["start_page"]
    for pos, name in section["scan_pages"]:
        if pos <= offset:
            page = name
    return page


def load_annotations(directory: Path = ANNOTATIONS) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        doc = load_json(path)
        for row in doc["passages"]:
            rows.append(dict(row, _file=path.name, page_id=row.get("page_id", doc.get("page_id"))))
    return rows


# ---------------------------------------------------------------- definition audit

QUANTILE_METHOD = "training_participant_quantiles"
# Literal relations to another part / absolute amounts: never guaranteed by a cohort quantile class.
LITERAL_RELATION = re.compile(r"過目|過眼|於目|于目|於眼|于眼|不及|不覆|覆目|覆眼|壓眼|壓目|居額|居頭|一寸|連額|過眼角")
# Compound qualifiers a label does not measure: (feature prefix or label, regex on evidence).
UNMEASURED_QUALIFIERS = (("nose_bridge", re.compile(r"連")), ("chin_shape", re.compile(r"豐")),
                         ("nose_tip:tapered", re.compile(r"薄")), ("cheekbone", re.compile(r"尖")),
                         ("lip_thickness:thin", re.compile(r"弱")),
                         ("lip_projection", re.compile(r"掀|翻")),
                         ("lip_bow:pronounced", re.compile(r"稜|棱")),
                         ("lip_thickness:full", re.compile(r"(上下|兩脣|兩唇).{0,3}厚")),
                         ("pigmented_marks:absent", re.compile(r"面無痣|滿面|全臉|全顔")))
HIGH_PREVALENCE_SHARE = 0.90


def load_definition_audit(path: Path = LANDMARK_LABELS) -> dict:
    """Per-feature target definitions of the release landmark labels (train-270 fit)."""
    if not path.exists():
        return {"quantile_features": set(), "majority": {}, "source": None}
    doc = load_json(path)
    quantile = {f for f, d in doc["features"].items() if d.get("method") == QUANTILE_METHOD}
    majority = {}
    for f, sup in doc.get("class_support", {}).get("train", {}).items():
        counts = sup["counts"]
        if sum(counts):
            i = max(range(len(counts)), key=counts.__getitem__)
            share = counts[i] / sum(counts)
            if share > HIGH_PREVALENCE_SHARE:
                majority[f"{f}:{doc['features'][f]['classes'][i]}"] = round(share, 4)
    return {"quantile_features": quantile, "majority": majority, "source": path.name,
            "version": doc.get("version")}


def audit_condition(c: dict, audit: dict) -> dict:
    """Downgrade an annotator 'exact' condition the release target definition cannot guarantee."""
    if c.get("status") != "exact":
        return c
    label, ev = c["label"], c["evidence"]
    feature = label.split(":", 1)[0]
    reason = None
    if feature in audit["quantile_features"] and LITERAL_RELATION.search(ev):
        reason = ("literal relation in the text is not guaranteed by a cohort-relative train-quantile class "
                  f"({feature}); see data/release/landmark_labels.json")
    else:
        for key, pattern in UNMEASURED_QUALIFIERS:
            if (label == key or feature == key) and pattern.search(ev):
                reason = f"compound condition: '{pattern.pattern}' in {ev!r} is not measured by {label}"
                break
    if reason is None:
        return c
    out = {k: v for k, v in c.items() if k != "label"}
    out.update(status="approximate", candidate_labels=sorted(set(c.get("candidate_labels", [])) | {label}),
               reason=reason, annotator_status="exact", audit="release_definition_audit")
    return out


def check_annotation(row: dict, vocab: dict) -> list[str]:
    errors = []
    pid = row.get("passage_id", "?")
    for field in ("passage_id", "page_id", "section", "original_text", "translation", "commentary"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{pid}: {field} must be a nonempty string")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", str(row.get("passage_id", ""))):
        errors.append(f"{pid}: passage_id must be ASCII [A-Za-z0-9_.:-]")
    conditions = row.get("conditions")
    if not isinstance(conditions, list):
        errors.append(f"{pid}: conditions must be a list")
        conditions = []
    for c in conditions:
        if c.get("status") not in CONDITION_STATUSES:
            errors.append(f"{pid}: bad condition status {c.get('status')!r}")
        if c.get("region") not in REGIONS:
            errors.append(f"{pid}: bad region {c.get('region')!r}")
        if not isinstance(c.get("evidence"), str) or not c["evidence"].strip():
            errors.append(f"{pid}: condition evidence missing")
        elif squash(c["evidence"]) not in squash(row.get("original_text", "")):
            errors.append(f"{pid}: condition evidence {c['evidence']!r} not in original_text")
        if not isinstance(c.get("observable_ko"), str) or not c["observable_ko"].strip():
            errors.append(f"{pid}: condition observable_ko missing")
        if c.get("status") == "exact":
            if c.get("label") not in vocab["labels"] and c.get("label") not in vocab["pending"]:
                errors.append(f"{pid}: exact condition label {c.get('label')!r} is neither canonical v3 nor pending")
        elif c.get("label") is not None:
            errors.append(f"{pid}: only exact conditions may set label (use candidate_labels)")
        for cand in c.get("candidate_labels", []):
            if cand not in vocab["labels"] and cand not in vocab["pending"]:
                errors.append(f"{pid}: candidate label {cand!r} is not canonical")
        if c.get("status") != "exact" and not str(c.get("reason", "")).strip():
            errors.append(f"{pid}: non-exact condition needs a reason")
    for key in ("claim_topics",):
        bad = [t for t in row.get(key, []) if t not in CLAIM_TOPICS]
        if bad:
            errors.append(f"{pid}: unknown {key} {bad}")
    return errors


def build(registry: dict, manifest: dict, annotations: list[dict], vocab: dict) -> tuple[dict, dict]:
    sections = load_snapshot_sections(registry, manifest)
    pages = {p["page_id"]: p for p in registry["pages"]}
    literature = {l["literature_id"]: l for l in registry["literature"]}
    errors, ids = [], Counter(r.get("passage_id") for r in annotations)
    errors += [f"duplicate passage_id {k}" for k, n in ids.items() if n > 1]
    placed = []
    for row in annotations:
        errors += check_annotation(row, vocab)
        page = pages.get(row.get("page_id"))
        if page is None:
            errors.append(f"{row.get('passage_id')}: unknown page_id {row.get('page_id')!r}")
            continue
        wanted = [s for s in sections[page["page_id"]] if s["heading"] == row.get("section")
                  and s["occurrence"] == row.get("section_occurrence", 1)]
        if len(wanted) != 1:
            errors.append(f"{row.get('passage_id')}: section {row.get('section')!r} not found once in {page['page_id']}")
            continue
        section = wanted[0]
        try:
            span = locate(section, row.get("original_text", ""))
        except ValueError as e:
            errors.append(f"{row.get('passage_id')}: {e}")
            continue
        if span is None:
            errors.append(f"{row.get('passage_id')}: original_text not found verbatim in {page['page_id']} / {section['heading']}")
            continue
        if row.get("section_title"):  # inline titles in unheaded 四庫 text
            if squash(row["section_title"]) not in squash(section["text"][:span[0]]):
                errors.append(f"{row.get('passage_id')}: section_title {row['section_title']!r} not before passage")
        order_key = (registry["pages"].index(page), sections[page["page_id"]].index(section), span[0])
        placed.append((order_key, row, page, section, span))
    if errors:
        return {}, {"ok": False, "errors": errors}

    placed.sort(key=lambda x: (literature[x[2]["literature_id"]]["literature_priority"], x[0]))
    audit = load_definition_audit()
    legacy = load_json(LEGACY_CATALOG)["records"] if LEGACY_CATALOG.exists() else []
    scans = {sid: load_transcription(registry, sid) for sid in {p["source_id"] for p in registry["pages"]
                                                                 if p["kind"] == "transcription"}}
    check_texts = defaultdict(list)  # literature_id -> [(page_id, section, normalized text)] of check witnesses
    for p in registry["pages"]:
        if p["kind"] == "transcription" and scans[p["source_id"]]["scan"]["role"] == "check_witness":
            # One concatenated text per check copy: heading choices of different transcribers
            # must not split a sentence the other transcriber kept whole.
            joined = "".join(variant_key(sec["heading"]) + variant_key(sec["text"]) for sec in sections[p["page_id"]])
            check_texts[p["literature_id"]].append((p["page_id"], {"heading": "(whole transcription)",
                                                                   "start_page": None}, joined))
    by_text, passages, eligible_keys = {}, [], []
    per_lit = Counter()
    for order_key, row, page, section, span in placed:
        lit = literature[page["literature_id"]]
        per_lit[lit["literature_id"]] += 1
        conditions = [audit_condition(dict(c), audit) for c in row["conditions"]]
        exact = sorted({c["label"] for c in conditions if c["status"] == "exact" and c["label"] in vocab["labels"]})
        pending = sorted({c["label"] for c in conditions if c["status"] == "exact" and c["label"] in vocab["pending"]})
        reasons = []
        if pending:
            reasons.append("LABEL_PENDING_REGISTRY:" + ",".join(pending))
        if not conditions:
            reasons.append("NO_OBSERVABLE_MORPHOLOGY_CONDITION")
        for status in CONDITION_STATUSES[1:]:
            if any(c["status"] == status for c in conditions):
                reasons.append(f"CONDITION_{status.upper()}")
        features = [l.split(":", 1)[0] for l in exact + pending]
        if len(set(features)) != len(features):
            reasons.append("CONTRADICTORY_CLASSES_FOR_ONE_FEATURE")
        sensitive = sorted(set(row.get("claim_topics", [])) & set(SENSITIVE_TOPICS))
        if sensitive:
            reasons.append("SENSITIVE_HISTORICAL_CLAIM:" + ",".join(sensitive))
        key = text_key(row["original_text"])
        if key in by_text:
            reasons.append("DUPLICATE_TEXT_OF:" + by_text[key])
        else:
            by_text[key] = row["passage_id"]
        reasons += [f"ANNOTATOR_EXCLUSION:{r}" for r in row.get("exclusions", [])]
        check = None
        if page["kind"] == "transcription":
            if "〓" in row["original_text"]:
                reasons.append("ILLEGIBLE_CHARACTER_IN_TRANSCRIPTION")
            check = transcription_agreement(row["original_text"], check_texts.get(page["literature_id"], []))
            if check["status"] != "agrees_with_check_witness":
                reasons.append("TRANSCRIPTION_UNCONFIRMED")
        if not reasons:  # parallel witness of an already eligible statement with the same labels
            for other_key, other_id, other_labels in eligible_keys:
                if other_labels == exact and SequenceMatcher(None, key, other_key).ratio() >= NEAR_DUPLICATE_RATIO:
                    reasons.append("NEAR_DUPLICATE_OF:" + other_id)
                    break
        if not reasons:
            eligible_keys.append((key, row["passage_id"], exact))
        scan = scan_page_at(section, span[0])
        scan_end = scan_page_at(section, span[1] - 1)
        locator = f"section {section['heading']}"
        if section["occurrence"] > 1:
            locator += f" (occurrence {section['occurrence']})"
        if row.get("section_title"):
            locator += f" / {row['section_title']}"
        locator += f" / chars {span[0]}-{span[1]} of section text"
        if scan:
            locator += f" / scan {scan}" + (f" - {scan_end}" if scan_end != scan else "")
        if page["kind"] == "transcription":
            source = transcription_source(page, lit, section, span, scans[page["source_id"]], row)
            source["transcription_check"] = check
        else:
          source = {
            "title": lit["title"], "url": page_url(page["title"], page["revid"]),
            "chapter": page["chapter"], "locator": locator,
            "literature_title": lit["title"], "passage_no": row.get("passage_no") or section["heading"],
            "passage_no_kind": row.get("passage_no_kind", "section_heading"),
            "section_heading": section["heading"], "section_occurrence": section["occurrence"],
            "char_span": list(span), "scan_page": scan, "scan_page_end": scan_end,
            "wikisource_title": page["title"], "wikisource_revid": page["revid"],
            "snapshot_file": f"snapshots/{page['page_id']}.html",
            "snapshot_sha256": manifest["pages"][page["page_id"]]["snapshot_sha256"],
            "text_witness": lit["text_witness"],
            "license": "public domain original; Wikisource transcription CC BY-SA 4.0"}
        source["page_id"] = page["page_id"]
        if row.get("attributed_to"):
            source["attributed_to"] = row["attributed_to"]
        passages.append({
            "passage_id": row["passage_id"], "literature_id": lit["literature_id"],
            "original_text": row["original_text"], "translation": row["translation"],
            "commentary": row["commentary"], "source": source, "required_labels": exact,
            "pending_labels": pending,
            "literature_priority": lit["literature_priority"],
            "passage_order": per_lit[lit["literature_id"]] * 10,
            "annotation_status": ANNOTATION_STATUS, "service_eligible": not reasons,
            "exclusion_reasons": reasons, "region": row.get("region"),
            "observable_conditions": conditions,
            "claim_topics": sorted(set(row.get("claim_topics", []))),
            "translation_kind": "korean_literal_ai_draft", "commentary_kind": "historical_note_ai_draft",
            "display_policy": "historical_text_only_not_a_personal_prediction",
            "legacy_record_ids": legacy_ids(legacy, page, section, row["original_text"]),
            "annotation_file": f"annotations/{row['_file']}",
            **({"review_note": row["review_note"]} if row.get("review_note") else {})})
    corpus = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/release/build_reference_corpus.py build",
        "label_vocabulary": {"source": "data/label_schema_kface_v3.json", "version": vocab["version"],
                             "sources": vocab["sources"], "feature_origins": vocab["feature_origins"],
                             "format": "feature:class", "withheld_features": vocab["withheld"],
                             "canonical_label_count": len(vocab["labels"]),
                             "definition_audit": {"source": "data/release/" + str(audit["source"]),
                                                  "version": audit.get("version"),
                                                  "quantile_features": sorted(audit["quantile_features"]),
                                                  "high_prevalence_labels": audit["majority"],
                                                  "rules": ["literal relations never exact for quantile classes",
                                                            "class prevalence is descriptive metadata, not a semantic exclusion",
                                                            "unmeasured compound qualifiers downgrade to approximate"]},
                             "pending_labels_source": "data/release/literature/pending_labels.json",
                             "pending_labels": sorted(vocab["pending"])},
        "annotation_policy": {
            "annotation_status": ANNOTATION_STATUS,
            "human_review": "none; no passage has been reviewed by a human expert",
            "translation": "Korean literal translation drafted by an AI model from the quoted original",
            "commentary": "historical/terminology note drafted by an AI model; no personal inference",
            "eligibility": "service_eligible only if every observable condition maps exactly to one canonical release label (v3 schema, training registry, confirmed extended schema; pending labels keep the passage ineligible), scan-transcribed text agrees with an independent second-copy transcription and has no illegible character, no unobservable/approximate/ambiguous/unmapped condition remains, the passage is not a textual duplicate, and it carries no sensitive historical claim (" + ", ".join(SENSITIVE_TOPICS) + ")",
            "display": "historical text only; not a judgement of character, intelligence, health or fate"},
        "sources": [dict(l, source_urls=[page_url(p["title"], p["revid"]) if p["kind"] != "transcription"
                                         else scans[p["source_id"]]["scan"]["file_page"]
                                         for p in registry["pages"] if p["literature_id"] == l["literature_id"]])
                    for l in registry["literature"]],
        "scan_transcriptions": [{
            "source_id": sid, "literature_id": t["scan"]["literature_id"], "role": t["scan"]["role"],
            "commons_title": t["scan"]["commons_title"], "file_page": t["scan"]["file_page"],
            "pdf_sha256": t["scan"]["sha256"], "pdf_pages": t["scan"]["pages"],
            "transcribed_pages": len(t["pages"]), "missing_pages": t["missing_pages"],
            "transcription_files": t["files"], "transcription_sha256": t["sha256"],
            "transcription_status": "ai_visual_transcription_not_human_proofread",
            "page_types": dict(Counter(p["page_type"] for p in t["pages"].values())),
            "characters": sum(len(text_key(b["text"])) for p in t["pages"].values() for b in p.get("blocks", [])),
            "illegible_marks": sum(b["text"].count("〓") for p in t["pages"].values() for b in p.get("blocks", [])),
            "uncertain_entries": sum(len(p.get("uncertain", [])) for p in t["pages"].values()),
            **({"window20_agreement_with_check_witness": window_agreement(t, scans)} if t["scan"]["role"] == "primary_witness" else {})}
            for sid, t in sorted(scans.items())],
        "literature_coverage": registry["coverage_only"],
        "source_sections": archive_sections(registry, manifest, sections, scans),
        "passages": passages}
    report = validate_corpus_document(corpus, vocab)
    return corpus, report


VARIANTS = str.maketrans({"爲": "為", "眞": "真", "靑": "青", "淸": "清", "衆": "眾", "敎": "教", "兇": "凶",
                          "髮": "髪", "脣": "唇", "顴": "顴", "隂": "陰", "陽": "陽", "㑹": "會", "黒": "黑",
                          "竪": "豎", "啓": "啟", "鬭": "鬥", "閒": "間", "卽": "即", "旣": "既", "眥": "眦", "愼": "慎", "硏": "研",
                          "絛": "縧", "鷰": "燕", "閗": "鬥", "麤": "粗", "麄": "粗", "峯": "峰", "羣": "群"})


def variant_key(text: str) -> str:
    """Comparison key across copies: no punctuation/whitespace, common glyph variants folded."""
    return text_key(text).translate(VARIANTS)


def transcription_agreement(original: str, check_sections: list) -> dict:
    key = variant_key(original)
    if not check_sections:
        return {"status": "no_check_witness_transcribed"}
    for page_id, section, text in check_sections:
        if key and key in text:
            return {"status": "agrees_with_check_witness", "check_page_id": page_id,
                    "check_offset": text.find(key),
                    "comparison": "exact substring of the independent second-copy transcription after removing punctuation/whitespace and folding glyph variants"}
    best = max(((SequenceMatcher(None, key, t).find_longest_match(0, len(key), 0, len(t)).size, pid, sec)
                for pid, sec, t in check_sections), key=lambda x: x[0])
    return {"status": "not_found_in_check_witness", "longest_common_run": best[0], "passage_chars": len(key)}


def window_agreement(t: dict, scans: dict, width: int = 20) -> dict:
    """Share of consecutive 20-character windows of a primary transcription found verbatim in the check copy."""
    def body(x):
        return "".join(variant_key(b["text"]) for p in x["pages"].values() if p["page_type"] in ("text", "toc")
                       for b in p.get("blocks", []))
    check = "".join(body(x) for x in scans.values()
                    if x["scan"]["role"] == "check_witness" and x["scan"]["literature_id"] == t["scan"]["literature_id"])
    text = body(t)
    windows = [text[i:i + width] for i in range(0, len(text) - width, width)]
    hit = sum(w in check for w in windows)
    return {"windows": len(windows), "found": hit, "rate": round(hit / len(windows), 4) if windows else None}


def transcription_source(page: dict, lit: dict, section: dict, span: tuple, t: dict, row: dict) -> dict:
    first = int((scan_page_at(section, span[0]) or "pdf_page:0").split(":")[1])
    last = int((scan_page_at(section, span[1] - 1) or "pdf_page:0").split(":")[1])
    scan = t["scan"]
    locator = f"section {section['heading']}"
    if section["occurrence"] > 1:
        locator += f" (occurrence {section['occurrence']})"
    if row.get("section_title"):
        locator += f" / inline title {row['section_title']}"
    locator += f" / chars {span[0]}-{span[1]} of section text / PDF page {first}" + (f"-{last}" if last != first else "")
    printed = [t["pages"][n].get("printed_page") for n in range(first, last + 1) if n in t["pages"]]
    if any(printed):
        locator += " / printed folio " + "-".join(x for x in printed if x)
    source = {
        "title": lit["title"], "url": f"{scan['url']}#page={first}", "chapter": scan["chapter"], "locator": locator,
        "literature_title": lit["title"],
        "passage_no": row.get("passage_no") or row.get("section_title") or section["heading"],
        "passage_no_kind": row.get("passage_no_kind", "inline_title" if row.get("section_title") else "section_heading"),
        "section_heading": section["heading"], "section_occurrence": section["occurrence"],
        "char_span": list(span), "scan_page": f"pdf_page:{first}", "scan_page_end": f"pdf_page:{last}",
        "pdf_page": first, "pdf_page_end": last, "printed_folio": printed,
        "wikisource_title": None, "wikisource_revid": None,
        "commons_title": scan["commons_title"], "commons_file_page": scan["file_page"],
        "commons_sha1": scan["commons_sha1"], "snapshot_file": scan["snapshot_file"],
        "snapshot_sha256": scan["sha256"], "scan_source_id": scan["source_id"],
        "transcription_files": sorted({t["pages"][n]["_file"] for n in range(first, last + 1) if n in t["pages"]}),
        "transcription_sha256": t["sha256"],
        "transcription_status": "ai_visual_transcription_not_human_proofread",
        "text_witness": lit["text_witness"],
        "license": "Ming-period text; 1925 printing scanned by the National Library of China and hosted on Wikimedia Commons (see file page)"}
    if row.get("attributed_to"):
        source["attributed_to"] = row["attributed_to"]
    return source


def archive_sections(registry, manifest, sections, scans):
    """Preserve all acquired text, including material not translated or service-mapped.

    This private research archive is never a candidate passage. Transcription and
    translation completeness are separate facts; no placeholder translation is made.
    """
    literature = {row["literature_id"]: row for row in registry["literature"]}
    records = []
    for page in registry["pages"]:
        lit = literature[page["literature_id"]]
        for index, section in enumerate(sections[page["page_id"]], 1):
            text = section["text"]
            if not text.strip():
                continue
            if page["kind"] == "transcription":
                source = transcription_source(page, lit, section, (0, len(text)), scans[page["source_id"]], {})
            else:
                source = {"title": lit["title"], "url": page_url(page["title"], page["revid"]),
                          "chapter": page["chapter"], "wikisource_revid": page["revid"],
                          "snapshot_file": f"snapshots/{page['page_id']}.html",
                          "snapshot_sha256": manifest["pages"][page["page_id"]]["snapshot_sha256"],
                          "scan_page": section["start_page"],
                          "license": "public domain original; Wikisource transcription CC BY-SA 4.0"}
            records.append({"section_id": f"{page['page_id']}:{section['occurrence']}:{index}",
                            "literature_id": page["literature_id"], "page_id": page["page_id"],
                            "heading": section["heading"], "occurrence": section["occurrence"],
                            "original_text": text, "text_sha256": sha256_bytes(text.encode("utf-8")),
                            "source": source, "annotation_status": "source_archive_unreviewed",
                            "service_eligible": False})
    return records


def legacy_ids(records: list[dict], page: dict, section: dict, original: str) -> list[str]:
    """Link to data/interpretation_catalog_v1.json records quoting the same section heading."""
    if page["kind"] != "gj":
        return []
    volume = page["page_id"][2:]
    target = squash(original)
    out = []
    for r in records:
        if f"/第{volume}卷" not in r.get("source_url", "") or r.get("name_zh") != section["heading"]:
            continue
        legacy = squash(r.get("original_section", ""))
        if legacy and (legacy in target or target in legacy):
            out.append(r["record_id"])
    return out


NEAR_DUPLICATE_RATIO = 0.75


def text_key(text: str) -> str:
    """Duplicate key: ignores whitespace, punctuation and editorial brackets."""
    return PUNCT.sub("", text)


PUNCT = re.compile(r"[\s\u3000\u200b，。、；：！？「」『』《》〈〉（）()〔〕·,.;:!?\"'○□]")


def section_char_coverage(sections_by_page: dict, passages: list[dict]) -> list[dict]:
    """Share of each annotated section's non-punctuation characters covered by passages."""
    spans = defaultdict(list)
    for p in passages:
        src = p["source"]
        spans[(src["page_id"], src["section_heading"], src["section_occurrence"])].append(src["char_span"])
    out = []
    for (page_id, heading, occ), ranges in sorted(spans.items()):
        section = next(s for s in sections_by_page[page_id] if s["heading"] == heading and s["occurrence"] == occ)
        text = section["text"]
        covered = set()
        for a, b in ranges:
            covered.update(range(a, b))
        content = [i for i, ch in enumerate(text) if not PUNCT.fullmatch(ch)]
        hit = sum(1 for i in content if i in covered)
        missing, run = [], []
        for i in content:
            if i in covered:
                if run:
                    missing.append("".join(text[j] for j in run))
                    run = []
            else:
                run.append(i)
        if run:
            missing.append("".join(text[j] for j in run))
        out.append({"page_id": page_id, "heading": heading, "occurrence": occ, "content_chars": len(content),
                    "covered_chars": hit, "coverage": round(hit / len(content), 4) if content else 1.0,
                    "uncovered_fragments": missing[:20]})
    return out


def validate_corpus_document(corpus: dict, vocab: dict) -> dict:
    errors = []
    if corpus.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    lit_ids = [s.get("literature_id") for s in corpus.get("sources", [])]
    if len(set(lit_ids)) != len(lit_ids):
        errors.append("duplicate literature_id")
    seen, texts = set(), defaultdict(list)
    for p in corpus.get("passages", []):
        pid = p.get("passage_id")
        if pid in seen:
            errors.append(f"duplicate passage_id {pid}")
        seen.add(pid)
        if p.get("literature_id") not in lit_ids:
            errors.append(f"{pid}: dangling literature_id")
        for f in ("original_text", "translation", "commentary"):
            if not isinstance(p.get(f), str) or not p[f].strip():
                errors.append(f"{pid}: empty {f}")
        src = p.get("source") or {}
        for f in ("title", "url", "chapter", "locator"):
            if not isinstance(src.get(f), str) or not src[f].strip():
                errors.append(f"{pid}: source.{f} missing")
        labels = p.get("required_labels", [])
        if labels != sorted(set(labels)):
            errors.append(f"{pid}: required_labels must be sorted and unique")
        for l in labels:
            if l not in vocab["labels"]:
                errors.append(f"{pid}: non-canonical label {l}")
        if not isinstance(p.get("service_eligible"), bool):
            errors.append(f"{pid}: service_eligible not boolean")
        if p.get("service_eligible"):
            if not labels or p.get("exclusion_reasons"):
                errors.append(f"{pid}: eligible passage needs labels and no exclusion reasons")
            if any(c["status"] != "exact" for c in p.get("observable_conditions", [])):
                errors.append(f"{pid}: eligible passage has non-exact condition")
            feats = [l.split(":")[0] for l in labels]
            if len(set(feats)) != len(feats):
                errors.append(f"{pid}: eligible passage has contradictory classes")
            exact = sorted({c["label"] for c in p["observable_conditions"] if c["status"] == "exact"})
            if exact != labels or p.get("pending_labels"):
                errors.append(f"{pid}: required_labels differ from exact conditions")
        elif not p.get("exclusion_reasons"):
            errors.append(f"{pid}: excluded passage needs a reason")
        if p.get("annotation_status") != ANNOTATION_STATUS:
            errors.append(f"{pid}: annotation_status must stay {ANNOTATION_STATUS}")
        for n in ("literature_priority", "passage_order"):
            if isinstance(p.get(n), bool) or not isinstance(p.get(n), int):
                errors.append(f"{pid}: {n} must be int")
        texts[text_key(p.get("original_text", ""))].append(p)
    for group in texts.values():
        if sum(bool(p.get("service_eligible")) for p in group) > 1:
            errors.append("duplicate original_text eligible twice: " + ",".join(p["passage_id"] for p in group))
    passages = corpus.get("passages", [])
    eligible = [p for p in passages if p.get("service_eligible")]
    reasons = Counter(r.split(":")[0] for p in passages for r in p.get("exclusion_reasons", []))
    label_use = Counter(l for p in eligible for l in p["required_labels"])
    pending_use = Counter(l for p in passages for l in p.get("pending_labels", []))
    return {
        "ok": not errors, "errors": errors,
        "passages": len(passages), "eligible": len(eligible), "excluded": len(passages) - len(eligible),
        "by_literature": {l: {"passages": sum(p["literature_id"] == l for p in passages),
                              "eligible": sum(p["literature_id"] == l for p in eligible)} for l in lit_ids},
        "by_region": {r: {"passages": sum(p.get("region") == r for p in passages),
                          "eligible": sum(p.get("region") == r for p in eligible)}
                      for r in sorted({p.get("region") for p in passages if p.get("region")})},
        "exclusion_reason_counts": dict(reasons.most_common()),
        "condition_status_counts": dict(Counter(c["status"] for p in passages for c in p.get("observable_conditions", []))),
        "eligible_label_usage": dict(sorted(label_use.items())),
        "pending_label_usage": dict(sorted(pending_use.items())),
        "canonical_labels_without_eligible_passage": sorted(set(vocab["labels"]) - set(label_use)),
        "eligible_rule_sizes": dict(Counter(len(p["required_labels"]) for p in eligible))}


def section_coverage(registry: dict, manifest: dict, corpus: dict) -> list[dict]:
    sections = load_snapshot_sections(registry, manifest)
    used = Counter((p["source"]["page_id"], p["source"]["section_heading"], p["source"]["section_occurrence"])
                   for p in corpus["passages"])
    out = []
    for page in registry["pages"]:
        for s in sections[page["page_id"]]:
            out.append({"page_id": page["page_id"], "heading": s["heading"], "occurrence": s["occurrence"],
                        "chars": len(squash(s["text"])),
                        "passages": used.get((page["page_id"], s["heading"], s["occurrence"]), 0)})
    return out


def check_transcription(doc: dict, registry: dict) -> list[str]:
    """Format check for one AI visual transcription file of an image-only scan."""
    errors = []
    scans = {s["source_id"]: s for s in registry.get("scan_sources", [])}
    scan = scans.get(doc.get("source_id"))
    if scan is None:
        return [f"unknown source_id {doc.get('source_id')!r}"]
    first, last = doc.get("page_range", [0, -1])
    if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last <= scan["pages"]):
        errors.append(f"bad page_range {doc.get('page_range')}")
    if "not human-proofread" not in str(doc.get("transcriber", "")):
        errors.append("transcriber must state the transcription is not human-proofread")
    got = [p.get("pdf_page") for p in doc.get("pages", [])]
    if got != list(range(first, last + 1)):
        errors.append(f"pages must be exactly {first}..{last} in order")
    for page in doc.get("pages", []):
        n = page.get("pdf_page")
        if page.get("page_type") not in PAGE_TYPES:
            errors.append(f"p{n}: bad page_type {page.get('page_type')!r}")
        if page.get("printed_page") is not None and not isinstance(page["printed_page"], str):
            errors.append(f"p{n}: printed_page must be string or null")
        for b in page.get("blocks", []):
            if b.get("type") not in BLOCK_TYPES:
                errors.append(f"p{n}: bad block type {b.get('type')!r}")
            text = b.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"p{n}: empty block")
            elif re.search(r"[A-Za-z0-9]", text):
                errors.append(f"p{n}: ASCII letters/digits in block text {text[:20]!r}")
        if page.get("page_type") == "text" and not page.get("blocks"):
            errors.append(f"p{n}: text page without blocks")
        if not isinstance(page.get("uncertain", []), list):
            errors.append(f"p{n}: uncertain must be a list")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "inventory", "build", "validate", "check", "check-transcription"))
    parser.add_argument("--file", type=Path, help="check: validate a single annotation file")
    parser.add_argument("--refresh", action="store_true", help="fetch: re-download existing snapshots")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    registry = load_json(REGISTRY)
    if args.command == "check-transcription":
        errors = check_transcription(load_json(args.file), registry)
        print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, indent=1))
        return 0 if not errors else 1
    if args.command == "fetch":
        fetch(registry, refresh=args.refresh)
        return 0
    manifest = load_json(SNAPSHOT_MANIFEST)
    vocab = load_label_vocabulary()
    if args.command == "inventory":
        dump_json(INVENTORY, inventory(registry, manifest))
        return 0
    if args.command == "validate":
        report = validate_corpus_document(load_json(args.corpus), vocab)
        print(json.dumps({k: report[k] for k in ("ok", "passages", "eligible", "errors")}, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.command == "check":
        rows = [dict(r, _file=args.file.name, page_id=r.get("page_id", doc.get("page_id")))
                for doc in [load_json(args.file)] for r in doc["passages"]]
        corpus, report = build(registry, manifest, rows, vocab)
        if report["ok"]:
            cov = section_char_coverage(load_snapshot_sections(registry, manifest), corpus["passages"])
            report = {k: report[k] for k in ("ok", "passages", "eligible", "exclusion_reason_counts",
                                              "condition_status_counts", "eligible_label_usage")}
            report["section_coverage"] = cov
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    corpus, report = build(registry, manifest, load_annotations(), vocab)
    if not report["ok"]:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 1
    report["section_coverage"] = section_coverage(registry, manifest, corpus)
    report["annotated_section_char_coverage"] = section_char_coverage(
        load_snapshot_sections(registry, manifest), corpus["passages"])
    report["corpus_sha256"] = sha256_bytes(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    dump_json(args.corpus, corpus)
    dump_json(VALIDATION, report)
    print(json.dumps({k: report[k] for k in ("ok", "passages", "eligible", "excluded", "by_literature")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
