# Minimum Training Split Validation Design

## Problem

Dataset version 4 contains one annotated image. The exporter placed that image in
`train` and left `val` empty, while still writing `val: images/val` to
`data.yaml`. Ultralytics therefore spent time preparing the run and then failed
while loading the empty validation split.

Reusing the same image for training and validation would make the experiment
metrics misleading. The local single-user MVP should reject this input early
and explain how to fix it.

## Required Behavior

### Dataset version creation

- Count distinct images containing annotations for the selected active classes.
- Require at least two annotated images before creating a frozen version.
- When fewer than two images are available, return an actionable validation
  error that includes the current count and the minimum count.
- Preserve the current deterministic split behavior for two or more images. Two
  images produce one training image and one validation image.
- Do not create a database version row or partial artifact directory when this
  validation fails.

### Training preflight

- Before importing or invoking Ultralytics, inspect the selected version's split
  manifest and artifact directories.
- Require at least one training image and one validation image.
- Reject legacy, manually edited, or incomplete versions immediately with an
  actionable error instructing the user to annotate at least two images and
  create a new dataset version.
- Keep the failed run record and log so the experiment history remains
  auditable.

### User workflow

For the current dataset, the user must annotate at least one additional image,
return to Quality and Versions, create a new version, and then create a new
training task. Existing failed runs and version 4 remain unchanged.

## Error Handling

Version creation should report a message equivalent to:

> Cannot create dataset version: at least 2 annotated images are required for
> training and validation; current selection has 1.

Training preflight should report a message equivalent to:

> Dataset version requires at least one training image and one validation image.
> Annotate at least 2 images and create a new version.

The backend API remains the source of truth. The frontend can display the API
message through its existing error handling without introducing a new workflow.

## Testing

- Unit-test split behavior for zero, one, two, and representative multi-image
  inputs.
- API-test that version creation rejects one annotated image without persisting
  a version or artifacts.
- Preserve the existing two-image version export test and assert non-empty
  training and validation splits.
- Runner-test that an existing version with an empty validation split fails
  before Ultralytics training is invoked.
- Run the complete backend test suite after implementation.

## Non-Goals

- No single-image debug mode.
- No train/validation image duplication.
- No changes to augmentation, model configuration, or prediction workflows.
- No automatic mutation or deletion of existing dataset versions and runs.
