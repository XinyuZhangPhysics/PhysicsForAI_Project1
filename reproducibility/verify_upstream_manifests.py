from __future__ import annotations
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

DATASET_ID = "DKYoon/SlimPajama-6B"
DATASET_REVISION = "b5f90f419b7489cdba26fdbc8c022fcb5562f968"
ROW_RE = re.compile(r"^train-(\d{5}):row-(\d+)$")

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def collect(root: Path):
    expected = {}
    refs = set()
    files = sorted(root.rglob("*.jsonl"))
    for path in files:
        for row in load_jsonl(path):
            rid = row.get("train_row_id")
            if rid:
                refs.add(rid)
                if row.get("text_sha256"):
                    expected[rid] = row["text_sha256"]
            for prefix in ("source", "target"):
                rid = row.get(f"{prefix}_train_row_id")
                if rid:
                    refs.add(rid)
                    if row.get(f"{prefix}_text_sha256"):
                        expected[rid] = row[f"{prefix}_text_sha256"]
            for doc in row.get("source_documents", []):
                rid = doc["train_row_id"]
                refs.add(rid)
                if doc.get("text_sha256"):
                    expected[rid] = doc["text_sha256"]
    missing = sorted(refs - set(expected))
    if missing:
        raise RuntimeError(
            f"{len(missing)} referenced rows have no published text hash; first={missing[0]}"
        )
    return files, expected

def parquet_for(data: Path, shard: int) -> Path:
    matches = []
    for root in (data, data / "data"):
        matches.extend(root.glob(f"train-{shard:05d}-of-00048-*.parquet"))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet for shard {shard}, found {len(matches)}"
        )
    return matches[0]

def verify(data: Path, expected: dict[str, str]) -> int:
    by_shard = defaultdict(dict)
    for rid, digest in expected.items():
        match = ROW_RE.match(rid)
        if not match:
            raise ValueError(rid)
        by_shard[int(match.group(1))][int(match.group(2))] = (rid, digest)
    checked = 0
    for shard, rows in sorted(by_shard.items()):
        parquet = pq.ParquetFile(parquet_for(data, shard))
        offset = 0
        pending = set(rows)
        for row_group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(row_group).num_rows
            local = [ordinal for ordinal in pending if offset <= ordinal < offset + size]
            if local:
                column = parquet.read_row_group(row_group, columns=["text"]).column(0)
                for ordinal in local:
                    rid, expected_digest = rows[ordinal]
                    got = sha_text(column[ordinal - offset].as_py())
                    if got != expected_digest:
                        raise RuntimeError(
                            f"hash mismatch {rid}: {got} != {expected_digest}"
                        )
                    pending.remove(ordinal)
                    checked += 1
            offset += size
        if pending:
            raise RuntimeError(f"missing rows in shard {shard}: {len(pending)}")
        print(f"verified shard {shard:02d}: {len(rows)} rows", flush=True)
    return checked

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="local DKYoon/SlimPajama-6B checkout at the pinned revision",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path(__file__).resolve().parent / "upstream_selection",
    )
    args = parser.parse_args()
    files, expected = collect(args.selection_root)
    checked = verify(args.data, expected)
    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "manifest_files": len(files),
                "unique_rows": checked,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
