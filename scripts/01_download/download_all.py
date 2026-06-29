from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import argparse
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


OPENNEURO_BUCKET = "https://s3.amazonaws.com/openneuro.org"
MAX_WORKERS = 8
RAW = PROJECT_ROOT / "data" / "raw"
MANIFESTS = PROJECT_ROOT / "outputs" / "manifests"
LOGS = PROJECT_ROOT / "outputs" / "logs"
AUDIT = PROJECT_ROOT / "outputs" / "audit"


@dataclass
class DownloadItem:
    dataset_id: str
    source: str
    url: str
    relative_path: str
    size_bytes: int | None = None
    checksum_type: str = ""
    checksum: str = ""
    status: str = "pending"
    error: str = ""


def log(message: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line)
    with (LOGS / "download_all.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def urlopen_with_retries(url: str, timeout: int = 120, retries: int = 3):
    last_error: Exception | None = None
    request = urllib.request.Request(url, headers={"User-Agent": "state-capacity-tinyrnn/0.1"})
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except ssl.SSLCertVerificationError:
            # The official TU Berlin dataset host currently fails certificate-chain
            # verification in this Windows Python environment. We still keep direct
            # URL provenance and use this fallback only after verification fails.
            context = ssl._create_unverified_context()
            return urllib.request.urlopen(request, timeout=timeout, context=context)
        except Exception as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                context = ssl._create_unverified_context()
                return urllib.request.urlopen(request, timeout=timeout, context=context)
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to open {url}: {last_error}") from last_error


def fetch_text(url: str, timeout: int = 120) -> str:
    with urlopen_with_retries(url, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict:
    return json.loads(fetch_text(url))


def head_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "state-capacity-tinyrnn/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def openneuro_items(dataset_id: str) -> list[DownloadItem]:
    prefix = f"{dataset_id}/"
    continuation: str | None = None
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    items: list[DownloadItem] = []
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if continuation:
            query["continuation-token"] = continuation
        xml_text = fetch_text(f"{OPENNEURO_BUCKET}?{urllib.parse.urlencode(query)}")
        root = ET.fromstring(xml_text)
        for obj in root.findall("s3:Contents", namespace):
            key = obj.findtext("s3:Key", default="", namespaces=namespace)
            if not key or key.endswith("/"):
                continue
            size = int(obj.findtext("s3:Size", default="0", namespaces=namespace))
            etag = obj.findtext("s3:ETag", default="", namespaces=namespace).strip('"')
            relative = key.removeprefix(prefix)
            items.append(
                DownloadItem(
                    dataset_id=dataset_id,
                    source="OpenNeuro S3",
                    url=f"{OPENNEURO_BUCKET}/{urllib.parse.quote(key)}",
                    relative_path=f"openneuro/{dataset_id}/{relative}",
                    size_bytes=size,
                    checksum_type="s3_etag",
                    checksum=etag,
                )
            )
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace) == "true"
        if not truncated:
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not continuation:
            raise RuntimeError(f"{dataset_id} listing truncated without continuation token")
    return items


def zenodo_items(record_id: str, dataset_id: str = "cog_bci") -> list[DownloadItem]:
    record = fetch_json(f"https://zenodo.org/api/records/{record_id}")
    items: list[DownloadItem] = []
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename")
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        if not key or not url:
            continue
        checksum = item.get("checksum", "")
        checksum_type = ""
        checksum_value = ""
        if isinstance(checksum, str) and ":" in checksum:
            checksum_type, checksum_value = checksum.split(":", 1)
        elif isinstance(checksum, str):
            checksum_type = "unknown"
            checksum_value = checksum
        items.append(
            DownloadItem(
                dataset_id=dataset_id,
                source=f"Zenodo record {record_id}",
                url=url,
                relative_path=f"{dataset_id}/{key}",
                size_bytes=item.get("size"),
                checksum_type=checksum_type,
                checksum=checksum_value,
            )
        )
    return items


def extract_zip_links_from_html(html: str, base_url: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+\.zip)["\']', html, flags=re.IGNORECASE)
    links.extend(re.findall(r'(?:https?://[^\s"\']+\.zip)', html, flags=re.IGNORECASE))
    urls: list[str] = []
    seen: set[str] = set()
    for link in links:
        url = urllib.parse.urljoin(base_url, link)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def tu_berlin_items() -> list[DownloadItem]:
    base = "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/"
    html = fetch_text(base)
    urls = extract_zip_links_from_html(html, base)
    candidate_urls = [
        "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG/EEG_01-26_MATLAB.zip",
        "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/NIRS/NIRS_01-26_MATLAB.zip",
    ]
    for url in candidate_urls:
        if url not in urls:
            urls.append(url)

    items: list[DownloadItem] = []
    for url in urls:
        name = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
        if not name.lower().endswith(".zip"):
            continue
        size = head_size(url)
        items.append(
            DownloadItem(
                dataset_id="tu_berlin_eeg_nirs",
                source="TU Berlin simultaneous EEG-NIRS website",
                url=url,
                relative_path=f"tu_berlin_eeg_nirs/{name}",
                size_bytes=size,
                checksum_type="",
                checksum="",
            )
        )
    # Keep only links that at least responded to HEAD or are scraped from the official page.
    unique: dict[str, DownloadItem] = {}
    for item in items:
        if item.size_bytes is not None or item.url in urls:
            unique[item.relative_path] = item
    return list(unique.values())


def compute_hash(path: Path, kind: str) -> str:
    hasher = hashlib.new(kind)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_item(item: DownloadItem) -> DownloadItem:
    target = RAW / item.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (item.size_bytes is None or target.stat().st_size == item.size_bytes):
        item.status = "skipped_existing"
        return item

    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen_with_retries(item.url, timeout=240) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if item.size_bytes is not None and tmp.stat().st_size != item.size_bytes:
            raise RuntimeError(f"size mismatch expected={item.size_bytes} got={tmp.stat().st_size}")
        tmp.replace(target)
        if item.checksum_type in {"md5", "sha256"} and item.checksum:
            observed = compute_hash(target, item.checksum_type)
            if observed.lower() != item.checksum.lower():
                raise RuntimeError(f"{item.checksum_type} mismatch expected={item.checksum} got={observed}")
        item.status = "downloaded"
    except Exception as exc:
        item.status = "failed"
        item.error = str(exc)
        if tmp.exists():
            tmp.unlink()
    return item


def target_complete(item: DownloadItem) -> bool:
    if not item.url:
        return False
    target = RAW / item.relative_path
    return target.exists() and (item.size_bytes is None or target.stat().st_size == item.size_bytes)


def parse_deadline_hhmm(value: str | None) -> datetime | None:
    if not value:
        return None
    hour_text, minute_text = value.split(":", 1)
    now = datetime.now()
    return now.replace(hour=int(hour_text), minute=int(minute_text), second=0, microsecond=0)


def write_manifest(items: list[DownloadItem]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with (MANIFESTS / "download_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in items], handle, indent=2)
    with (MANIFESTS / "checksums.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_id", "relative_path", "size_bytes", "checksum_type", "checksum", "status"],
            delimiter="\t",
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "dataset_id": item.dataset_id,
                    "relative_path": item.relative_path,
                    "size_bytes": item.size_bytes,
                    "checksum_type": item.checksum_type,
                    "checksum": item.checksum,
                    "status": item.status,
                }
            )


def downloaded_subject_count(dataset_id: str) -> int:
    root_map = {
        "ds007554": RAW / "openneuro" / "ds007554",
        "hbn_release_4": RAW / "openneuro" / "ds005508",
        "cog_bci": RAW / "cog_bci",
        "tu_berlin_eeg_nirs": RAW / "tu_berlin_eeg_nirs",
    }
    root = root_map[dataset_id]
    if not root.exists():
        return 0
    if dataset_id in {"ds007554", "hbn_release_4"}:
        return len([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if dataset_id == "cog_bci":
        return len(list(root.glob("sub-*.zip"))) + len([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if dataset_id == "tu_berlin_eeg_nirs":
        names = set()
        for path in root.glob("*.zip"):
            for match in re.findall(r"(?:VP|sub-)?0?([0-9]{1,2})", path.name, flags=re.IGNORECASE):
                number = int(match)
                if 1 <= number <= 26:
                    names.add(f"{number:02d}")
        if any(path.name.lower().startswith(("eeg_01-26", "nirs_01-26")) for path in root.glob("*.zip")):
            return 26
        return len(names)
    return 0


def write_eligibility(items: list[DownloadItem]) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[DownloadItem]] = {}
    for item in items:
        by_dataset.setdefault(item.dataset_id, []).append(item)

    specs = [
        ("ds007554", "OpenNeuro", "uvx openneuro-py@latest download --dataset=ds007554 --target-dir=data/raw/openneuro/ds007554", None, True),
        ("cog_bci", "Zenodo record 6874129", "python scripts/01_download/download_zenodo_record.py --record-id 6874129 --target-dir data/raw/cog_bci --verify true", 29, True),
        ("tu_berlin_eeg_nirs", "TU Berlin simultaneous EEG-NIRS", "python scripts/01_download/download_tu_berlin_eeg_nirs.py --target-dir data/raw/tu_berlin_eeg_nirs --verify true", 26, True),
        ("hbn_release_4", "OpenNeuro ds005508", "uvx openneuro-py@latest download --dataset=ds005508 --target-dir=data/raw/openneuro/ds005508", 324, True),
    ]
    with (AUDIT / "dataset_eligibility.tsv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset_id",
            "source",
            "download_command",
            "manual_login_required",
            "controlled_access",
            "browser_only",
            "raw_size_gb",
            "expected_subjects",
            "downloaded_subjects",
            "included_main",
            "exclusion_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for dataset_id, source, command, expected, included in specs:
            dataset_items = by_dataset.get(dataset_id, [])
            size = sum(item.size_bytes or 0 for item in dataset_items)
            failures = [item for item in dataset_items if item.status == "failed"]
            downloaded = downloaded_subject_count(dataset_id)
            exclusion = ""
            included_main = included
            if failures:
                included_main = False
                exclusion = f"download_failed_for_{len(failures)}_files"
            writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "source": source,
                    "download_command": command,
                    "manual_login_required": "false",
                    "controlled_access": "false",
                    "browser_only": "false",
                    "raw_size_gb": f"{size / (1024**3):.3f}" if size else "",
                    "expected_subjects": "" if expected is None else expected,
                    "downloaded_subjects": downloaded,
                    "included_main": str(included_main).lower(),
                    "exclusion_reason": exclusion,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deadline-hhmm",
        default=None,
        help="Local same-day HH:MM deadline. Stop after finishing the current batch.",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_full_run_allowed()
    deadline = parse_deadline_hhmm(args.deadline_hhmm)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "download_all.log").write_text("", encoding="utf-8")

    all_items: list[DownloadItem] = []
    datasets = [
        ("ds007554", lambda: openneuro_items("ds007554")),
        ("hbn_release_4", lambda: openneuro_items("ds005508")),
        ("cog_bci", lambda: zenodo_items("6874129", "cog_bci")),
        ("tu_berlin_eeg_nirs", tu_berlin_items),
    ]

    for dataset_id, builder in datasets:
        log(f"DISCOVER {dataset_id}")
        try:
            items = builder()
            log(f"DISCOVERED {dataset_id} files={len(items)} size_gb={sum(i.size_bytes or 0 for i in items)/(1024**3):.3f}")
            all_items.extend(items)
        except Exception as exc:
            failed = DownloadItem(dataset_id=dataset_id, source="discovery", url="", relative_path="", status="failed", error=str(exc))
            all_items.append(failed)
            log(f"DISCOVER_FAILED {dataset_id}: {exc}")

    write_manifest(all_items)

    for item in all_items:
        if target_complete(item):
            item.status = "skipped_existing"

    downloadable = [item for item in all_items if item.url and item.status != "skipped_existing"]
    log(f"DOWNLOAD_PHASE files={len(downloadable)} workers={args.workers} deadline={args.deadline_hhmm or ''}")
    completed = 0
    stopped_for_deadline = False
    for start in range(0, len(downloadable), args.workers):
        if deadline and datetime.now() >= deadline:
            stopped_for_deadline = True
            log("DEADLINE_REACHED before next batch; stopping without launching more files")
            break
        batch = downloadable[start : start + args.workers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_item, item) for item in batch]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                if completed <= 20 or completed % 25 == 0 or result.status == "failed":
                    log(
                        f"STATUS {completed}/{len(downloadable)} {result.status} "
                        f"{result.dataset_id} {result.relative_path} {result.error}"
                    )
        write_manifest(all_items)
        write_eligibility(all_items)
        if deadline and datetime.now() >= deadline:
            stopped_for_deadline = True
            log("DEADLINE_REACHED after current batch; stopping cleanly")
            break

    write_manifest(all_items)
    write_eligibility(all_items)
    failures = [item for item in all_items if item.status == "failed"]
    if stopped_for_deadline:
        log(f"DOWNLOAD_ALL paused_at_deadline failures={len(failures)}")
        return
    if failures:
        raise SystemExit(f"Download completed with {len(failures)} failed items; see outputs/manifests/download_manifest.json")
    log("DOWNLOAD_ALL complete")


if __name__ == "__main__":
    main()
