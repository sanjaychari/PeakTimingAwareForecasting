from pathlib import Path
import shutil

ROOT = Path("results")

KEEP_PREFIXES = (
    # Current ESNet experiments
    "esnet__r-nova-fd-01__ifHCOutOctets__",

    # Current MIT workload
    "mit_gpu__67691045781943-r7343737-n685852__utilization_gpu_pct__",

    # Current TelecomTS workload
    "telecomts__normal_stationary_Zone_B_no_congestion_File__TX_Bytes__",
)

KEEP_TOP_LEVEL = {
    "correlation",
    "paper_figures",
}

KEEP_LOSSES = {"mse", "ep", "mep"}
KEEP_HORIZONS = {"h96", "h192", "h336", "h720"}


def relevant_run(name: str) -> bool:
    if not name.startswith(KEEP_PREFIXES):
        return False

    parts = set(name.split("__"))

    if not (parts & KEEP_LOSSES):
        return False

    if not (parts & KEEP_HORIZONS):
        return False

    return True


def main():
    to_delete = []

    for path in ROOT.iterdir():
        if path.name in KEEP_TOP_LEVEL:
            continue

        if path.is_dir() and relevant_run(path.name):
            continue

        to_delete.append(path)

    print("Would delete:")
    for path in to_delete:
        print(" ", path)

    print(f"\nTotal: {len(to_delete)} entries")
    print("\nNo files were deleted.")
    print("Run again with --delete to actually remove them.")

    import sys
    if "--delete" in sys.argv:
        for path in to_delete:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"\nDeleted {len(to_delete)} entries.")


if __name__ == "__main__":
    main()
