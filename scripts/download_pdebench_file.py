from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URL = ""
DEFAULT_OUTPUT = Path("data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5")
DEFAULT_MD5 = "44cb784d5a07aa2b1c864cabdcf625f9"
DEFAULT_EXPECTED_SIZE_GB = 7.7
CHUNK_SIZE = 1024 * 1024 * 8


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_can_write(output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite")


def download_file(url: str, output: Path, force: bool) -> None:
    if not url:
        raise ValueError("No download URL configured; pass --url explicitly")

    ensure_can_write(output, force=force)
    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_suffix(output.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    try:
        with urlopen(url) as response, part_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=CHUNK_SIZE)
        os.replace(part_path, output)
    finally:
        if part_path.exists():
            part_path.unlink()


def inspect_hdf5_metadata(path: Path) -> dict[str, object]:
    import h5py

    datasets: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:

        def collect_dataset(name: str, item: object) -> None:
            if isinstance(item, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    }
                )

        handle.visititems(collect_dataset)
    return {"datasets": datasets}


def preflight_file(
    path: Path,
    expected_md5: str | None,
    min_size_bytes: int,
    inspect_hdf5: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return result

    size_bytes = path.stat().st_size
    result["size_bytes"] = size_bytes
    result["size_ok"] = size_bytes >= min_size_bytes

    if expected_md5:
        md5 = compute_md5(path)
        result["md5"] = md5
        result["md5_ok"] = md5 == expected_md5

    if inspect_hdf5:
        result["hdf5"] = inspect_hdf5_metadata(path)

    return result


def _size_gb_to_bytes(size_gb: float) -> int:
    return int(size_gb * 1024 * 1024 * 1024)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preflight one PDEBench HDF5 file."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md5", default=DEFAULT_MD5)
    parser.add_argument("--expected-size-gb", type=float, default=DEFAULT_EXPECTED_SIZE_GB)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--inspect-hdf5", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    output = args.output
    if not output.is_absolute():
        output = repo_root() / output

    min_size_bytes = _size_gb_to_bytes(args.expected_size_gb)
    if not args.preflight_only:
        download_file(args.url, output, force=args.force)

    result = preflight_file(
        output,
        expected_md5=args.md5,
        min_size_bytes=min_size_bytes,
        inspect_hdf5=args.inspect_hdf5,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        result.get("exists") is False
        or result.get("size_ok") is False
        or result.get("md5_ok") is False
    ):
        print("Preflight failed: file did not meet size or MD5 requirements.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
