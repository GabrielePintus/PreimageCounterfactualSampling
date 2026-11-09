"""Versioned numeric region artifacts: JSON + NPZ, never Python pickle."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .backend import BuildResult
from certcf.geometry.regions import CertifiedRegion, NormBall, RegionGeometry


def save_regions(result: BuildResult, directory, *, settings=None):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays, records = {}, []
    for i, r in enumerate(result.regions):
        prefix = f"r{i}_"
        g = r.geometry
        for key in ("A", "b", "lower", "upper"):
            arrays[prefix + key] = getattr(g, key)
        arrays[prefix + "representative"] = r.representative
        ball = None
        if g.ball is not None:
            arrays[prefix + "center"] = g.ball.center
            ball = {"radius": float(g.ball.radius), "p": "inf" if g.ball.p == np.inf else float(g.ball.p)}
        records.append({
            "region_id": r.region_id, "target_class": int(r.target_class),
            "source_domain_id": r.source_domain_id, "certificate": r.certificate,
            "prefix": prefix, "ball": ball,
        })
    # Immutable payload + atomically replaced manifest: interrupted writes leave
    # the previous checkpoint readable. Old payloads can be cleaned separately.
    filename = f"regions-{uuid.uuid4().hex}.npz"
    target = root / filename
    with target.open("wb") as f:
        np.savez_compressed(f, **arrays)
    manifest = {
        "format_version": 2, "kind": "certcf_regions", "file": filename,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "class_labels": list(result.class_labels), "input_shape": list(result.input_shape),
        "status": result.status, "diagnostics": result.diagnostics,
        "regions": records, "settings": settings or {},
    }
    temporary = root / f".manifest-{uuid.uuid4().hex}.json"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, root / "manifest.json")
    return root / "manifest.json"


def load_regions(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2 or manifest.get("kind") != "certcf_regions":
        raise ValueError("Unsupported region artifact format")
    filename = manifest["file"]
    if Path(filename).name != filename:
        raise ValueError("Invalid artifact payload path")
    target = root / filename
    if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Region artifact checksum mismatch")
    regions = []
    with np.load(target, allow_pickle=False) as data:
        for record in manifest["regions"]:
            p = record["prefix"]
            ball_record = record["ball"]
            ball = None if ball_record is None else NormBall(
                data[p + "center"], ball_record["radius"], float(ball_record["p"]),
            )
            geometry = RegionGeometry(*(data[p + k] for k in ("A", "b", "lower", "upper")), ball)
            regions.append(CertifiedRegion(
                record["region_id"], record["target_class"], geometry,
                record["source_domain_id"], data[p + "representative"], record["certificate"],
            ))
    result = BuildResult(tuple(regions), tuple(manifest["class_labels"]), tuple(manifest["input_shape"]), manifest["status"], manifest["diagnostics"])
    return result, manifest["settings"]
