# YOLO Trainer Annotator Operation Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a polished MiSans Word manual with real step-by-step screenshots that lets a zero-background annotator complete annotation, train a model, inspect predictions, and iterate on weak results.

**Architecture:** Keep product code and existing datasets unchanged. Build an isolated demonstration dataset from copied workspace images, capture the current Chinese UI through the local application, annotate screenshots non-destructively, and generate one DOCX from structured content plus those assets. Render the DOCX to page PNGs and revise until every page is readable and clean.

**Tech Stack:** YOLO Trainer local frontend/backend, in-app browser automation, bundled Python runtime, python-docx, Pillow, LibreOffice renderer, MiSans font.

---

### Task 1: Prepare the manual workspace and font

**Files:**
- Create: `artifacts/annotator-manual/source/`
- Create: `artifacts/annotator-manual/screenshots/raw/`
- Create: `artifacts/annotator-manual/screenshots/annotated/`
- Create: `artifacts/annotator-manual/rendered/`
- Create: `artifacts/annotator-manual/fonts/`
- Create: `artifacts/annotator-manual/manual_manifest.json`

- [ ] **Step 1: Confirm application and document tool availability**

Run:

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:5173/
test -x /Applications/LibreOffice.app/Contents/MacOS/soffice || command -v soffice
```

Expected: backend reports `status: ok`, frontend returns HTML, and LibreOffice is available.

- [ ] **Step 2: Create isolated artifact directories**

Run:

```bash
mkdir -p artifacts/annotator-manual/{source,screenshots/raw,screenshots/annotated,rendered,fonts}
```

Expected: all five directories exist under `artifacts/annotator-manual`.

- [ ] **Step 3: Acquire and verify MiSans**

Download MiSans from Xiaomi's official font distribution, place the regular and semibold font files under `artifacts/annotator-manual/fonts/`, and register them for the current user when necessary for LibreOffice rendering.

Run:

```bash
fc-scan artifacts/annotator-manual/fonts/* 2>/dev/null | rg 'family|style' || mdls artifacts/annotator-manual/fonts/*
```

Expected: output identifies the font family as `MiSans` and includes regular and semibold weights.

- [ ] **Step 4: Record reproducible inputs**

Create `artifacts/annotator-manual/manual_manifest.json` with the application URL, capture date, source dataset ID, demonstration dataset name, screenshot filenames, font filenames, and final DOCX path.

- [ ] **Step 5: Commit the implementation plan**

```bash
git add docs/superpowers/plans/2026-07-17-annotator-operation-manual.md
git commit -m "docs: plan annotator operation manual"
```

### Task 2: Build an isolated demonstration dataset

**Files:**
- Create: `artifacts/annotator-manual/source/images/`
- Modify: `artifacts/annotator-manual/manual_manifest.json`

- [ ] **Step 1: Inspect active datasets and choose source images**

Run:

```bash
curl -fsS http://127.0.0.1:8000/api/storage/items | jq '.items[] | select(.entity_type == "dataset") | {id: .entity_id, name: .display_name, size: .size_bytes}'
```

Expected: at least one active dataset is available without opening the trash lifecycle.

- [ ] **Step 2: Copy a small diverse image subset**

Select 12-24 images that include easy targets, small targets, clutter, and partial occlusion. Copy only image files into `artifacts/annotator-manual/source/images/`; do not copy or edit source labels.

Run:

```bash
find artifacts/annotator-manual/source/images -type f | wc -l
```

Expected: between 12 and 24 images are present and the source dataset file count is unchanged.

- [ ] **Step 3: Import the demonstration dataset through the UI**

Use project name `标注培训演示` and dataset name `无人机目标演示集`. Scan the copied image directory, verify the preview count, and import it as a new dataset.

Expected: the application automatically proceeds to `类别管理` and the dataset summary shows the copied image count.

- [ ] **Step 4: Create demonstration classes**

Create only classes supported by the selected images, using clear names such as `无人机` and `固定翼目标`. Avoid synonyms or duplicate category meanings.

Expected: at least one active class exists and the annotation step becomes available.

- [ ] **Step 5: Annotate enough examples for the complete workflow**

Create tight boxes on at least six images, including one small target and one occluded or difficult target. Save every image before switching.

Expected: the quality page reports at least six annotated images and at least two images are available for deterministic train/validation splitting.

### Task 3: Capture the real application workflow

**Files:**
- Create: `artifacts/annotator-manual/screenshots/raw/01-navigation.png`
- Create: `artifacts/annotator-manual/screenshots/raw/02-dataset-scan.png`
- Create: `artifacts/annotator-manual/screenshots/raw/03-dataset-preview.png`
- Create: `artifacts/annotator-manual/screenshots/raw/04-dataset-imported.png`
- Create: `artifacts/annotator-manual/screenshots/raw/05-class-create.png`
- Create: `artifacts/annotator-manual/screenshots/raw/06-annotation-overview.png`
- Create: `artifacts/annotator-manual/screenshots/raw/07-create-box.png`
- Create: `artifacts/annotator-manual/screenshots/raw/08-small-target-zoom.png`
- Create: `artifacts/annotator-manual/screenshots/raw/09-box-properties.png`
- Create: `artifacts/annotator-manual/screenshots/raw/10-save-next.png`
- Create: `artifacts/annotator-manual/screenshots/raw/11-quality-review.png`
- Create: `artifacts/annotator-manual/screenshots/raw/12-version-create.png`
- Create: `artifacts/annotator-manual/screenshots/raw/13-version-result.png`
- Create: `artifacts/annotator-manual/screenshots/raw/14-training-config.png`
- Create: `artifacts/annotator-manual/screenshots/raw/15-training-progress.png`
- Create: `artifacts/annotator-manual/screenshots/raw/16-training-result.png`
- Create: `artifacts/annotator-manual/screenshots/raw/17-prediction-config.png`
- Create: `artifacts/annotator-manual/screenshots/raw/18-prediction-results.png`
- Create: `artifacts/annotator-manual/screenshots/raw/19-failure-sample.png`
- Create: `artifacts/annotator-manual/screenshots/raw/20-gt-pred-overlay.png`
- Create: `artifacts/annotator-manual/screenshots/raw/21-return-to-correct.png`
- Create: `artifacts/annotator-manual/screenshots/raw/22-run-comparison.png`

- [ ] **Step 1: Standardize capture conditions**

Use one desktop viewport large enough to show labels clearly, keep the YOLO Trainer tab focused, close dialogs unrelated to the target step, and preserve the same demonstration project throughout.

- [ ] **Step 2: Capture intake and class setup**

Capture the six-step navigation, scan form, preview result, imported dataset state, and class creation state. Do not expose unrelated local paths or historical records.

- [ ] **Step 3: Capture annotation actions**

Capture the annotation workbench overview, a newly created box, zoomed small-target view, selected-box properties, edge tag controls, and saved-next state. Verify every saved box remains visible after image reselection.

- [ ] **Step 4: Capture quality and version creation**

Capture quality issue review, training readiness, version settings, and the completed version row with train/validation/test counts.

- [ ] **Step 5: Capture training states**

Create a fast demonstration run with a Nano model and conservative settings. Capture the initial configuration, active progress/log state, and completed metrics/artifacts state. If the run fails, fix the environment or dataset issue before continuing; do not document a failed run as the normal path.

- [ ] **Step 6: Capture evaluation and iteration**

Capture prediction configuration, summary results, failure filters, GT/Pred overlay, opening a failure sample in annotation, and a comparison between the original and iterated runs.

- [ ] **Step 7: Validate screenshot set**

Run:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
paths = sorted(Path('artifacts/annotator-manual/screenshots/raw').glob('*.png'))
assert len(paths) >= 22, len(paths)
for path in paths:
    with Image.open(path) as image:
        assert image.width >= 1200 and image.height >= 700, (path, image.size)
print(f'{len(paths)} screenshots validated')
PY
```

Expected: at least 22 screenshots validate at a readable desktop resolution.

### Task 4: Add instructional callouts to screenshots

**Files:**
- Create: `artifacts/annotator-manual/annotate_screenshots.py`
- Create: `artifacts/annotator-manual/screenshots/annotated/*.png`
- Modify: `artifacts/annotator-manual/manual_manifest.json`

- [ ] **Step 1: Define callout metadata**

In `annotate_screenshots.py`, define each source image, crop region, numbered marker center, arrow endpoint, highlight rectangle, and short caption as structured data rather than hard-coded one-off drawing calls.

- [ ] **Step 2: Implement restrained visual annotation**

Use Pillow to draw red numbered circles, 3-pixel arrows, and translucent red highlight rectangles. Keep markers outside UI text whenever possible and preserve an unmarked raw screenshot for every annotated output.

- [ ] **Step 3: Generate annotated screenshots**

Run with the bundled Python runtime:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 artifacts/annotator-manual/annotate_screenshots.py
```

Expected: every screenshot listed in the manifest has a corresponding annotated output.

- [ ] **Step 4: Inspect all annotated screenshots**

Open each annotated PNG at 100% and verify markers identify the intended control, do not hide labels, and remain legible after scaling to the document width.

### Task 5: Author the MiSans Word manual

**Files:**
- Create: `artifacts/annotator-manual/build_manual.py`
- Create: `artifacts/YOLO_Trainer_零基础标注员操作手册.docx`

- [ ] **Step 1: Resolve the compact reference guide style sheet**

Use A4 portrait geometry, MiSans body and heading styles, a restrained blue-green palette, numbered chapter headings, real Word list numbering, fixed image width, consistent figure captions, and page header/footer with page fields.

- [ ] **Step 2: Implement document building helpers**

In `build_manual.py`, provide focused helpers for cover, chapter opener, numbered step, screenshot figure, completion check, tip/warning callout, glossary item, and checklist item. Set east-Asia and Latin run fonts to MiSans explicitly.

- [ ] **Step 3: Write the full workflow content**

Cover dataset import, class creation, box rules, image zoom/pan, save behavior, quality review, version freezing, Nano-model training, prediction review, failure classification, correction, new-version creation, retraining, and run comparison. Every screenshot must be introduced by an action and followed by an observable completion condition.

- [ ] **Step 4: Add zero-background explanations**

Define bounding box, class, GT, Pred, confidence, false positive, false negative, train/validation/test split, dataset version, and iteration at first use or in the glossary. Keep explanations operational and no longer than needed to perform the task.

- [ ] **Step 5: Build the DOCX**

Run:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 artifacts/annotator-manual/build_manual.py
```

Expected: `artifacts/YOLO_Trainer_零基础标注员操作手册.docx` exists and all screenshots are embedded.

### Task 6: Render, inspect, and revise the document

**Files:**
- Modify: `artifacts/annotator-manual/build_manual.py`
- Replace: `artifacts/YOLO_Trainer_零基础标注员操作手册.docx`
- Create: `artifacts/annotator-manual/rendered/page-*.png`

- [ ] **Step 1: Render the DOCX to PNG pages**

Run:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  ~/.codex/plugins/cache/openai-primary-runtime/documents/26.715.12143/skills/documents/render_docx.py \
  artifacts/YOLO_Trainer_零基础标注员操作手册.docx \
  --output_dir artifacts/annotator-manual/rendered \
  --emit_pdf
```

Expected: one PNG per page plus a QA PDF.

- [ ] **Step 2: Run structural audits**

Run:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  ~/.codex/plugins/cache/openai-primary-runtime/documents/26.715.12143/skills/documents/scripts/style_lint.py \
  artifacts/YOLO_Trainer_零基础标注员操作手册.docx
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  ~/.codex/plugins/cache/openai-primary-runtime/documents/26.715.12143/skills/documents/scripts/images_audit.py \
  artifacts/YOLO_Trainer_零基础标注员操作手册.docx
```

Expected: no missing images, broken relationships, fake headings, or unexplained font/style drift.

- [ ] **Step 3: Inspect every rendered page at 100%**

Check cover balance, MiSans glyph rendering, heading hierarchy, image legibility, figure-caption pairing, callout spacing, page breaks, headers/footers, and absence of clipping or overlap.

- [ ] **Step 4: Revise and re-render until clean**

Adjust image sizing, paragraph spacing, keep-with-next rules, page breaks, and callout geometry in `build_manual.py`. Rebuild and re-render after each revision; do not edit the generated DOCX manually.

- [ ] **Step 5: Verify the operator workflow against the live app**

Follow the manual once from dataset import through prediction review using the demonstration project. Confirm button names, navigation labels, status meanings, and error guidance match the current UI.

- [ ] **Step 6: Final artifact checks**

Run:

```bash
test -s artifacts/YOLO_Trainer_零基础标注员操作手册.docx
unzip -p artifacts/YOLO_Trainer_零基础标注员操作手册.docx word/styles.xml | rg -q 'MiSans'
unzip -p artifacts/YOLO_Trainer_零基础标注员操作手册.docx word/document.xml | rg -q 'MiSans'
git status --short
```

Expected: the DOCX is non-empty, MiSans is present in styles and document runs, and existing unrelated untracked files remain untouched.
