# FaceTeller literature annotation guide (corpus v1)

Annotations are AI drafts (`annotation_status = ai_annotated_unreviewed`). Never claim
human/expert review. Never invent text, sources, citations or translations of text that
is not in the snapshot. The build rejects any `original_text` or condition `evidence`
that is not a verbatim (whitespace-insensitive) substring of the snapshot section.

Section text to annotate: `data/release/literature/section_inventory.json`
(`pages[].sections[].text`, keyed by `page_id` + `heading` + `occurrence`).
Exemplar file: `annotations/xf01_eye.json`. Check a file with

    .venv/Scripts/python.exe scripts/release/build_reference_corpus.py check --file data/release/literature/annotations/<file>.json

## File format

```json
{"page_id": "gj634", "annotator": "AI model draft (Claude), single pass; no human review",
 "sections_complete": ["龍鼻", "虎鼻"],
 "passages": [ {passage}, ... ]}
```

Passage:

| field | rule |
|---|---|
| passage_id | ASCII, stable, unique across ALL files: `<PREFIX>-<region>-<nnn>` using the prefix you were assigned |
| section | exact `heading` string from the inventory (keep 《》 if the inventory has them) |
| section_occurrence | only if the heading occurs more than once on the page (default 1) |
| section_title | only for unheaded 太清神鑑 pages: the inline title (e.g. `論眉部`) that precedes the passage |
| attributed_to | only when the compilation itself names an author for the piece (e.g. `許負`, `袁柳莊`, `達摩`) |
| original_text | contiguous verbatim text copied from the section (keep punctuation exactly; you may omit a leading/trailing 「 」) |
| translation | Korean literal translation (직역) of exactly that original_text |
| commentary | Korean historical/terminology note, 1-3 sentences |
| region | main region: eye, brow, nose, mouth, face, philtrum, ear, forehead, glabella, under_eye, cheekbone, chin, mole, teeth, tongue, other |
| conditions | list of observable-form conditions (see below); `[]` if the passage has no face-form statement |
| claim_topics | list from: criminality, sexual_conduct, health_disease, death_lifespan, intelligence, violence, kinship_harm, malice_deceit, fertility_offspring (any prediction about one's own children/heirs: existence, number, sex, timing), wealth, rank_office, character, fortune_general, family_children, marriage, none, other |

## Segmentation

* Cover each assigned section completely and in order: every character of the section
  belongs to exactly one passage (`check` prints `section_coverage`; target 1.0).
  List each fully covered heading in `sections_complete`.
* Named type entries (e.g. 龍鼻, 鬼眉; ~32-character poems) = ONE passage each: the type
  is defined by all of its conditions together.
* Essays / rhymed formula lists: one passage per self-contained statement, normally one
  couplet or sentence (`形態 → 주장`). Do not merge unrelated statements; do not split a
  condition from the claim it supports.
* A passage that only states general doctrine, colour/qi, voice, conduct etc. is still
  a passage (translated, commentary) with `conditions: []` or unobservable conditions.

## Translation (직역)

* Literal and complete: render every clause, keep the order, no embellishment, no
  softening of negative/stigmatizing claims (they are excluded from service later by
  `claim_topics`, not by rewriting).
* Technical terms: Korean reading + hanja + short gloss at first use in the passage,
  e.g. `준두(準頭, 코끝)`, `산근(山根, 콧뿌리)`, `인당(印堂, 두 눈썹 사이)`,
  `난대·정위(蘭臺·廷尉, 좌우 콧방울)`, `지각(地閣, 턱)`, `관골(顴骨, 광대뼈)`.
* If a phrase is uncertain, translate the most literal reading and say so in commentary.
  Do not guess variant characters; do not "correct" the transcription.

## Commentary

* Explain terms, which body part/form the text describes, and why it does or does not
  map to a canonical label. State claims as the text's claim: `문헌은 …라고 적었다/연결했다`.
* Never address the reader, never assert the claim is true, never infer anything about a
  living person's character, intelligence, health or fate.
* Do not invent historical facts (dates, authorship, editions) beyond what the snapshot
  itself says.

## Conditions and the exact-mapping lexicon

Each condition: `{"region", "evidence" (verbatim substring of original_text),
"observable_ko", "status", "label" (exact only), "candidate_labels" (canonical, optional),
"reason" (required unless exact)}`.

Status values:

* `exact` - the text directly states the same dimension and direction as ONE canonical
  label below. Only these make a passage service-eligible.
* `approximate` - related but not the same measure (figurative size, absolute units,
  side-specific, 豐/厚 for a tip, 鼻高 without saying root vs tip...). Give candidates.
* `ambiguous` - several readings (animal metaphors like 龍/鳳/虎/牛 without explicit form;
  秀/清/好 quality words used as form).
* `unmapped` - observable form but no canonical v3 label (ear, forehead, mole, wrinkles,
  under-eye/臥蠶, philtrum depth/width, nose dorsum/鼻梁 shape, teeth, hair, unibrow 眉交).
* `unobservable` - colour, luster, qi, gaze/神, movement, voice, conduct, sex/age of the
  subject (`region: "subject"`), anything a still photo morphology label cannot show.

Every face-form statement in the passage must appear as a condition. A passage with
several forms has several conditions (AND). If a passage is about a female/male subject
only, add a `subject` unobservable condition.

Canonical labels (`feature:class` from `data/label_schema_kface_v3.json`) and the ONLY
classical expressions that count as `exact`:

| label | exact expressions (examples) |
|---|---|
| eye_size:large / small | 目大·眼大·目長·眼長 / 目小·眼小·目短 |
| eye_roundness:rounded / narrow | 目圓·眼圓 (the eye, not 睛 pupil) / 目細·眼細·細長(細 part) |
| eye_tilt:up / down | 目尾朝天·眼尾上·尾向上·眼梢上翹 / 眼尾垂·目尾下垂·眼角垂 |
| eye_asymmetry:asymmetric | 雌雄眼, 兩目大小不一 (not side-specific 左/右目小 = approximate) |
| sclera_show:lower / upper | 下三白 / 上三白 (三白 alone or 四白 = ambiguous/unmapped) |
| eye_depth:deep / prominent | 目深·眼深·眼陷·深陷 / 目突·眼突·睛凸·眼凸 (露 alone = approximate) |
| eye_spacing:close / wide | 兩目相近·眼近 / 兩目相遠·眼遠 |
| eyelid:single / double | explicit 單/雙 eyelid statements only |
| brow_shape:straight / arched / angled | 一字·眉直·平直 / 彎·如新月·如初月·如彎弓 / 眉有角·眉折 |
| brow_tail:descending / ascending | 八字·尾垂·尾下垂 / 尾上·尾揚·尾昂 |
| brow_thickness:thick / thin | 眉粗·眉麄 / 眉細 (眉薄 = approximate: thin vs sparse) |
| brow_density:dense / sparse | 眉濃·眉密 / 眉疏·眉散·眉稀·眉淡 |
| brow_length:long / short | 眉長·過目·過眼 / 眉短·不及目·不覆眼 |
| brow_eye_gap:close / wide | 壓眼·壓目·眉低近目 / 眉高 (眉高居額) |
| brow_tail_clarity:fading / clear | 尾散·尾疏·後疏 / 尾聚·尾清 |
| brow_gap:wide / narrow | 印堂廣·印堂闊·印堂寬·兩眉相遠 / 印堂狹·印堂窄·兩眉相近 (眉交/連 = unmapped unibrow) |
| nose_length:long / short | 鼻長 / 鼻短 |
| nose_width:wide / narrow | 鼻闊·鼻廣·鼻寬 / 鼻狹·鼻窄 (鼻大/小 = approximate) |
| nose_tip:rounded / tapered | 準圓·準頭圓·準頭豐圓·準頭肥圓 / 準尖·準頭尖·鼻尖 (準豐/厚 alone = approximate) |
| nose_projection:high / low | 準頭高·準頭起·準頭聳 / 準頭低·準頭陷 (鼻高/鼻低/塌 = approximate) |
| nose_bridge:high / low | 山根高·山根聳·山根起·山根隆·山根直聳 / 山根低·山根陷·山根塌·山根平 (斷/折 = approximate) |
| nostril_show:visible / hidden | 孔露·孔仰·竅露·露竈·孔昂 / 孔不露·不見孔·孔藏 |
| alar_fullness:full / flat | 蘭臺廷尉豐·蘭廷豐·竈豐·蘭臺廷尉大 / 蘭廷小·蘭臺廷尉薄 |
| mouth_width:wide / narrow | 口大·口闊·口廣·口寬 / 口小·口窄 (口容拳 = approximate) |
| lip_thickness:full / thin | 脣厚·唇厚·脣豐·兩脣豐厚 / 脣薄·唇薄 |
| lip_balance:upper_thin / upper_full | 上薄下厚 / 上厚下薄 (上脣厚 alone = approximate) |
| mouth_corner:up / down | 角上·角仰·口角朝上·仰月 / 角垂·角下·口角垂·覆船 |
| lip_projection:protruded / retruded | 脣突·口突·脣掀 (吹火 = approximate) / 脣縮·口縮 |
| lip_bow:pronounced / flat | 脣有稜·脣稜分明 / 脣無稜 |
| philtrum_length:long / short | 人中長 / 人中短 (深/淺/闊/狹/偏/平 = unmapped) |
| face_shape:round / angular / oval | 面圓 / 面方 / 面如卵 (面長/短 -> face_length) |
| face_length:long / short | 面長 / 面短 |
| jaw_width:wide / narrow | 腮闊·兩腮寬 / 腮削·腮窄 |
| chin_shape:pointed / rounded / square | 頦尖·地閣尖·頤尖 / 地閣圓·頦圓 / 地閣方·頦方 |
| chin_projection:prominent / receding | 地閣朝·頦翹 = approximate; only explicit 頦突/頦縮 are exact |
| cheekbone:prominent / flat | 顴高·顴聳·顴起·顴露 / 顴低·顴平·無顴 |
| cheek_fullness:full / hollow | 頰豐·腮豐·頤豐 / 頰陷·頰削·腮陷 |
| lower_face_ratio:long / short | 下停長 / 下停短 |

Only labels in this table may be used. Medium/intermediate classes are never exact.
No two classes of the same feature in one passage. When unsure, do not use `exact`.

### Release vocabulary update (second dispatch, 2026-10-14)

forehead_height, forehead_width, ear_length, ear_width (data/release/extended_schema.json /
training_registry.json, visible-geometry definitions), under_eye and philtrum_definition
(training_registry.json) are now CANONICAL: exact use makes a passage eligible like any v3
label. Current pending status is read from pending_labels.json. pigmented_marks is promoted only after the automated400 annotation readiness check; vocabulary registration is not independent accuracy validation. The expression tables below still define what
counts as exact. ear_width is width/height of the visible ear (耳闊/耳寬 = wide, 耳窄/耳狹 =
narrow); forehead_* measure the VISIBLE forehead (skin to hair transition), so statements
about bone (額骨), the hairline itself or forehead protrusion are approximate.

### Additional morphology definitions

Labels currently listed in pending_labels.json may use `status: "exact"`; the build keeps those passages ineligible
(`LABEL_PENDING_REGISTRY`) and stores the label in `pending_labels`
(see `pending_labels.json`). Never equate a partly visible forehead with the hairline.

| label | exact expressions |
|---|---|
| forehead_width:wide / narrow | 額廣·額闊·額寬 / 額狹·額窄 (額小 = approximate) |
| forehead_height:high / low | 額長 / 額短·額低 (額高·額聳 = approximate: height vs projection) |
| ear_length:long / short | 耳長 / 耳短 (耳大/耳小 = approximate) |
| ear_width:wide / narrow | 耳闊·耳寬 / 耳窄·耳狹 |
| pigmented_marks:present / absent | Broad whole-face absence (面無痣) is approximate: the target covers only exposed forehead/upper cheeks, not all facial skin. A mark at a specific location is approximate (candidate present) unless the actual target records that location. |
| under_eye:visible / flat | 臥蠶·有臥蠶 / explicit 無臥蠶 (目下平 = approximate) |
| philtrum_definition:clear / faint | 人中深 / 人中淺·人中平·人中漫 |

### Release definition audit (2026-10-14, data/release/landmark_labels.json) — overrides the tables above

* Length/width/size/gap labels (eye_size, eye_roundness, brow_length, brow_eye_gap, nose_width,
  nose_length, mouth_width, lip_thickness, face_length, jaw_width, brow_gap, eye_spacing,
  philtrum_length, lower_face_ratio) are COHORT-RELATIVE train-270 quantile classes. They are
  exact only for plain relative words (長/短/大/小/闊/狹/厚/薄/高 of the part itself). A LITERAL
  RELATION to another part or an absolute amount is never guaranteed by a quantile class and is
  `approximate`: 過目/過眼/長於目/不及目/短於目/不覆目/不覆眼/壓眼/壓目/居額/一寸 … (e.g. brow_length
  short still has brow length ≈1.7× eye width, so 眉不及目 contradicts rather than matches it).
* Signed classes: eye_tilt and mouth_corner encode the physical sign (± a repeat tolerance), and
  lip_balance the 1.0 ratio. A plain direction (尾上/尾垂/角垂/角上/上薄下厚) may be exact.
  Class frequency is descriptive metadata only: common physical directions remain literal
  directions. Do not downgrade a condition merely because more than90% of training faces share it.
  Additional qualitative requirements such as extreme angle still need explicit measurement.
* Every form qualifier in the evidence must be measured. If a label covers only part of a
  compound (山根連額起, 地閣豐圓, 準頭尖薄, 顴骨尖高, 薄弱 …) the condition is `approximate`, or split
  into separate conditions where the extra part is approximate/unmapped.
* forehead_* / ear_* / under_eye / philtrum_definition are canonical vocabulary; runtime shows a
  passage only when every required feature decision is CONFIRMED at runtime. Canonical vocabulary is not proof of model accuracy; commentary must not hard-code a temporary "pending" state.
* A generic visible dark mark is not automatically the historical 黑子/痣 category.
  Keep this correspondence approximate unless the actual annotation contract establishes it.
  This is an appearance-category mismatch, not a demand for a clinical diagnosis.
  Upper-face presence does imply presence somewhere on the face; that location implication
  is valid, whereas upper-face absence never implies whole-face absence.
* Lip measurements must retain their actual meaning: 脣掀/翻 (lifted/turned lips) is not
  forward lip_projection; 有稜/棱 (defined rim) is not specifically a Cupid's-bow M shape;
  上下俱厚 needs both lips individually thick, not just a high combined thickness ratio.
  Those correspondences are approximate and keep the complete passage out of service matching.
