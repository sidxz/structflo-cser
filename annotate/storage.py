"""Annotation persistence — load/save in both GT JSON and YOLO formats.

Ground-truth JSON schema (pair format):
    [
      {
        "struct_bbox": [x1, y1, x2, y2],   # pixel coords of chemical structure
        "label_bbox":  [x1, y1, x2, y2],   # pixel coords of label ID  (null = skipped)
        "label_text":  "",                  # filled in by post-processing, not annotator
        "smiles":      ""                   # filled in by post-processing, not annotator
      },
      ...
    ]

YOLO .txt (written only when pairs are non-empty):
    0  cx  cy  w  h   (normalised 0-1; class 0 = chemical_structure)
    1  cx  cy  w  h   (normalised 0-1; class 1 = compound_label)

Annotation states:
    - GT JSON absent  → page not yet visited
    - GT JSON = []    → page explicitly marked as "no panels"
    - GT JSON = [...]  → page annotated with N pairs
"""

import json
from pathlib import Path


def gt_path(page_id: str, output_dir: Path) -> Path:
    return output_dir / "ground_truth" / f"{page_id}.json"


def lbl_path(page_id: str, output_dir: Path) -> Path:
    return output_dir / "labels" / f"{page_id}.txt"


def load(page_id: str, output_dir: Path) -> list[dict] | None:
    """Return pairs as [{struct_bbox, label_bbox, ...}], or None if not yet annotated."""
    p = gt_path(page_id, output_dir)
    if not p.exists():
        return None                          # not yet visited
    return json.loads(p.read_text())


def save(page_id: str, pairs: list[dict], img_w: int, img_h: int,
         output_dir: Path) -> None:
    """Persist annotations.

    GT JSON is *always* written (even for empty pages) so the page is
    tracked as 'done'.  YOLO .txt is only written when pairs are present.

    YOLO labels: class 0 = chemical_structure, class 1 = compound_label.
    """
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Ensure each record has the full schema
    records = [
        {
            "struct_bbox": pair["struct_bbox"],
            "label_bbox":  pair.get("label_bbox"),      # None = skipped
            "label_text":  pair.get("label_text", ""),
            "smiles":      pair.get("smiles", ""),
        }
        for pair in pairs
    ]
    gt_path(page_id, output_dir).write_text(json.dumps(records, indent=2))

    lbl = lbl_path(page_id, output_dir)
    if not pairs:
        lbl.unlink(missing_ok=True)         # no YOLO file for empty pages
        return

    lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl, "w") as f:
        for pair in pairs:
            s = pair["struct_bbox"]          # [x1, y1, x2, y2]
            l = pair.get("label_bbox")       # [x1, y1, x2, y2] or None

            # class 0 = chemical_structure
            sx1, sy1, sx2, sy2 = s
            f.write(
                f"0 {(sx1 + sx2) / 2 / img_w:.6f} {(sy1 + sy2) / 2 / img_h:.6f} "
                f"{(sx2 - sx1) / img_w:.6f} {(sy2 - sy1) / img_h:.6f}\n"
            )

            # class 1 = compound_label
            if l:
                lx1, ly1, lx2, ly2 = l
                f.write(
                    f"1 {(lx1 + lx2) / 2 / img_w:.6f} {(ly1 + ly2) / 2 / img_h:.6f} "
                    f"{(lx2 - lx1) / img_w:.6f} {(ly2 - ly1) / img_h:.6f}\n"
                )
