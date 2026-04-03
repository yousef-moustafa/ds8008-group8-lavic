import os
import json
from tqdm import tqdm

# =========================
# Config
# =========================
BASE_DATA_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "train_images")
CATEGORY_DIRS = [
    "all_beauty",
    "amazon_fashion",
    "amazon_home",
]
INPUT_FILENAME = "train.jsonl"
OUTPUT_FILENAME = "train_clean.jsonl"
CANDIDATE_KEY = "candidates_st"

# =========================
# Helpers
# =========================
def build_available_asins(image_dir):
    """
    Build a fast lookup set of ASINs from filenames like:
    B00ARY7CEG_0.jpg -> B00ARY7CEG
    """
    all_files = os.listdir(image_dir)
    available_asins = set()

    for name in tqdm(all_files, desc="Indexing image files"):
        if "_" in name:
            asin = name.split("_", 1)[0]
            available_asins.add(asin)

    return available_asins, len(all_files)


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def clean_jsonl_file(input_path, output_path, available_asins, candidate_key):
    """
    Clean one JSONL file by removing missing-image ASIN candidates.
    Keeps a row only if at least one candidate remains.
    """
    total_rows = count_lines(input_path)

    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_dropped": 0,
        "matched_references": 0,
        "missing_references": 0,
        "candidates_removed": 0,
        "unique_matched_asins": set(),
        "unique_missing_asins": set(),
    }

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_rows, desc=f"Cleaning {os.path.basename(os.path.dirname(input_path))}/{os.path.basename(input_path)}"):
            stats["rows_total"] += 1
            row = json.loads(line)

            old_candidates = row.get(candidate_key, [])
            new_candidates = []

            for asin in old_candidates:
                if not isinstance(asin, str):
                    stats["candidates_removed"] += 1
                    stats["missing_references"] += 1
                    continue

                if asin in available_asins:
                    new_candidates.append(asin)
                    stats["matched_references"] += 1
                    stats["unique_matched_asins"].add(asin)
                else:
                    stats["candidates_removed"] += 1
                    stats["missing_references"] += 1
                    stats["unique_missing_asins"].add(asin)

            if len(new_candidates) > 0:
                row[candidate_key] = new_candidates
                fout.write(json.dumps(row) + "\n")
                stats["rows_kept"] += 1
            else:
                stats["rows_dropped"] += 1

    return stats


def verify_clean_file(clean_path, available_asins, candidate_key):
    """
    Double-check the cleaned file:
    - every candidate must exist in available_asins
    - no row should have zero candidates
    """
    total_rows = count_lines(clean_path)

    verify_stats = {
        "rows_checked": 0,
        "bad_rows_empty_candidates": 0,
        "bad_candidate_references": 0,
        "bad_asins": set(),
    }

    with open(clean_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_rows, desc=f"Verifying {os.path.basename(os.path.dirname(clean_path))}/{os.path.basename(clean_path)}"):
            verify_stats["rows_checked"] += 1
            row = json.loads(line)

            candidates = row.get(candidate_key, [])

            if len(candidates) == 0:
                verify_stats["bad_rows_empty_candidates"] += 1

            for asin in candidates:
                if not isinstance(asin, str) or asin not in available_asins:
                    verify_stats["bad_candidate_references"] += 1
                    if isinstance(asin, str):
                        verify_stats["bad_asins"].add(asin)

    return verify_stats


def print_clean_summary(category, stats):
    print(f"\n=== CLEAN SUMMARY: {category} ===")
    print(f"Rows total            : {stats['rows_total']:,}")
    print(f"Rows kept             : {stats['rows_kept']:,}")
    print(f"Rows dropped          : {stats['rows_dropped']:,}")
    print(f"Matched references    : {stats['matched_references']:,}")
    print(f"Missing references    : {stats['missing_references']:,}")
    print(f"Candidates removed    : {stats['candidates_removed']:,}")
    print(f"Unique matched ASINs  : {len(stats['unique_matched_asins']):,}")
    print(f"Unique missing ASINs  : {len(stats['unique_missing_asins']):,}")

    if stats["unique_missing_asins"]:
        print("Sample missing ASINs:")
        for asin in list(stats["unique_missing_asins"])[:10]:
            print(" ", asin)


def print_verify_summary(category, verify_stats):
    print(f"\n=== VERIFY SUMMARY: {category} ===")
    print(f"Rows checked              : {verify_stats['rows_checked']:,}")
    print(f"Rows with empty candidates: {verify_stats['bad_rows_empty_candidates']:,}")
    print(f"Bad candidate references  : {verify_stats['bad_candidate_references']:,}")

    if verify_stats["bad_asins"]:
        print("Sample bad ASINs after cleaning:")
        for asin in list(verify_stats["bad_asins"])[:10]:
            print(" ", asin)

    if verify_stats["bad_rows_empty_candidates"] == 0 and verify_stats["bad_candidate_references"] == 0:
        print("Verification passed.")


# =========================
# Main
# =========================
def main():
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    print(f"Working directory : {os.getcwd()}")
    print(f"Image directory   : {IMAGE_DIR}")

    available_asins, num_files = build_available_asins(IMAGE_DIR)

    print(f"\nIndexed files              : {num_files:,}")
    print(f"Unique ASINs with images   : {len(available_asins):,}")

    grand = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_dropped": 0,
        "matched_references": 0,
        "missing_references": 0,
        "candidates_removed": 0,
    }

    for category in CATEGORY_DIRS:
        input_path = os.path.join(BASE_DATA_DIR, category, INPUT_FILENAME)
        output_path = os.path.join(BASE_DATA_DIR, category, OUTPUT_FILENAME)

        if not os.path.exists(input_path):
            print(f"\nSkipping {category}: file not found -> {input_path}")
            continue

        stats = clean_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            available_asins=available_asins,
            candidate_key=CANDIDATE_KEY,
        )

        print_clean_summary(category, stats)

        verify_stats = verify_clean_file(
            clean_path=output_path,
            available_asins=available_asins,
            candidate_key=CANDIDATE_KEY,
        )

        print_verify_summary(category, verify_stats)

        grand["rows_total"] += stats["rows_total"]
        grand["rows_kept"] += stats["rows_kept"]
        grand["rows_dropped"] += stats["rows_dropped"]
        grand["matched_references"] += stats["matched_references"]
        grand["missing_references"] += stats["missing_references"]
        grand["candidates_removed"] += stats["candidates_removed"]

    print("\n=== GRAND TOTAL ===")
    print(f"Rows total         : {grand['rows_total']:,}")
    print(f"Rows kept          : {grand['rows_kept']:,}")
    print(f"Rows dropped       : {grand['rows_dropped']:,}")
    print(f"Matched references : {grand['matched_references']:,}")
    print(f"Missing references : {grand['missing_references']:,}")
    print(f"Candidates removed : {grand['candidates_removed']:,}")


if __name__ == "__main__":
    main()