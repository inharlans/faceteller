# 柳莊相法 scan transcription guide (NLC / Wikimedia Commons PDFs)

The three PDFs in `snapshots/nlc/` are image-only scans (no text layer, no Wikisource
transcription exists). Text is produced by AI visual reading of rendered page images.
It is a transcription of what is printed on the page, never a reconstruction:

* Transcribe ONLY characters you can see on that page image. Never fill gaps from memory
  of 柳莊相法, 神相全編 or any other edition. If a character is unreadable write `〓`
  (one per character). If you are unsure, write your best reading and add an entry to the
  page's `uncertain` list.
* Keep printed glyph forms (爲 stays 爲, 眞 stays 眞). Do not modernise or correct.
* Reading order: vertical columns, right column first, top to bottom.
* Join the columns of one paragraph into one `text` block without line breaks. Start a
  new block where the print starts a new paragraph or a new heading.
* Headings (chapter titles such as 眉, 論眼, 乾坤賦; section titles printed larger or set
  apart) are `heading` blocks.
* Omit the small emphasis/punctuation circles and dots printed BESIDE characters
  (。 、 ﹅ marks in the margin of a column). Keep full-size characters printed in the
  column itself, including `○` used as a separator (e.g. `○解曰`).
* Interlinear small double-column notes: put them inline as `〈…〉` at their position.
* Running heads in the margin (e.g. `柳莊相法`, `柳莊相法 目錄`) and printed folio numbers
  (e.g. `七七`, `四`) are NOT text: put the folio number in `printed_page` (as printed,
  e.g. "七七") and drop the running head.
* Library stamps, watermarks, handwriting: do not transcribe; mention in `note`.
* Figure pages (face charts, palm charts): `page_type` "figure"; transcribe printed labels
  as `figure_label` blocks in reading order (right to left, top to bottom), or describe in
  `note` if labels are too dense/illegible.
* Blank, cover, colophon (版權頁) and table-of-contents pages: set `page_type` accordingly
  and transcribe any printed text as `text`/`heading` blocks.

## File format

One file per assigned page range:
`data/release/literature/transcriptions/<source_id>/p<first:03d>-p<last:03d>.json`

```json
{"source_id": "nlc511_63880",
 "page_range": [1, 25],
 "transcriber": "AI visual transcription (Claude) from 200-dpi renders; not OCR; not human-proofread",
 "pages": [
  {"pdf_page": 1, "page_type": "cover|blank|toc|text|figure|colophon|other",
   "printed_page": null,
   "blocks": [{"type": "heading", "text": "柳莊相法目錄"},
              {"type": "text", "text": "…"}],
   "uncertain": [{"context": "…眼〓深…", "reading": "深", "alternatives": ["溪"]}],
   "note": ""}
 ]}
```

`pdf_page` is the 1-based page index of the PDF (= image file `pNNN.png`). Every page of
the range must be present exactly once, in order, even if blank. Block `type` is one of
heading, text, figure_label. Block text must be nonempty and must not contain ASCII
letters or digits.

Check a file with

    .venv/Scripts/python.exe scripts/release/build_reference_corpus.py check-transcription --file <file>
