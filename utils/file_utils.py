from typing import Optional

def find_behavior_files(rat: str, repo_root: Optional[str] = None) -> list[str]:
    """
    Find all .tsv and .txt files for a given rat identifier.

    Behavior folder base is read from config/default_paths.json under the key
    "behavior". If that path is relative it is interpreted relative to the
    repository root (detected from this module if not provided).

    Args:
        rat: Rat identifier (string). This name is appended as a subfolder
             under the behavior folder.
        repo_root: Optional repository root Path (if not provided the parent of
                   the `config` folder is used).

    Returns:
        Sorted list of absolute file path strings for all matching .tsv and .txt
        files found recursively under the rat's behavior folder. Returns an
        empty list if no folder or files are found.
    """
    import json
    from pathlib import Path

    # determine repo root
    repo_root = Path(__file__).resolve().parents[1]

    defaults_path = repo_root / "config" / "default_paths.json"
    if not defaults_path.exists():
        raise FileNotFoundError(f"default_paths.json not found at {defaults_path}")

    with defaults_path.open("r", encoding="utf-8") as fh:
        defaults = json.load(fh)

    behavior_rel = defaults.get("behavior", "data/behavior")
    behavior_path = Path(behavior_rel)
    if not behavior_path.is_absolute():
        behavior_path = repo_root / behavior_path

    rat_folder = behavior_path / rat
    if not rat_folder.exists():
        return []

    exts = ("*.tsv", "*.txt")
    found: list[str] = []
    for ext in exts:
        for p in rat_folder.rglob(ext):
            if p.is_file():
                found.append(str(p.resolve()))

    # remove duplicates and return sorted list for deterministic order
    return sorted(dict.fromkeys(found))


def find_ephys_subfolders(rat: str, repo_root: Optional[str] = None) -> list[str]:
    """Return immediate subfolder paths under the ephys folder for a rat.

    Reads the `ephys` setting from `config/default_paths.json` and resolves
    it relative to the repository root. Returns a sorted list of absolute
    directory paths (strings). If the rat folder does not exist, returns
    an empty list.
    """
    import json
    from pathlib import Path

    # determine repo root
    repo_root = Path(__file__).resolve().parents[1]

    defaults_path = repo_root / "config" / "default_paths.json"
    if not defaults_path.exists():
        raise FileNotFoundError(f"default_paths.json not found at {defaults_path}")

    with defaults_path.open("r", encoding="utf-8") as fh:
        defaults = json.load(fh)

    ephys_rel = defaults.get("ephys", "data/ephys")
    ephys_path = Path(ephys_rel)
    if not ephys_path.is_absolute():
        ephys_path = repo_root / ephys_path

    rat_folder = ephys_path / rat
    if not rat_folder.exists():
        return []

    subfolders = [str(p.resolve()) for p in rat_folder.iterdir() if p.is_dir()]
    return sorted(subfolders)


def search_files(rat: str, date: str, repo_root: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Find a behavior file and ephys subfolder for a given rat and date.

    `date` is expected in MMDDYYYY format. The function searches behavior
    files for filenames containing the date string and ephys subfolders
    whose names contain the date. Returns a tuple (behavior_file, ephys_folder)
    where each element is an absolute path string or None if not found.
    """
    from pathlib import Path

    # convert MMDDYYYY into the two expected formats
    # behavior files: YYYY-MM-DD
    # ephys folders: YYYYMMDD
    if len(date) == 8 and date.isdigit():
        mm = date[0:2]
        dd = date[2:4]
        yyyy = date[4:8]
        behavior_date_token = f"{yyyy}-{mm}-{dd}"
        ephys_date_token = f"{yyyy}{mm}{dd}"
    else:
        # fallback: use the raw token for both
        behavior_date_token = date
        ephys_date_token = date

    # find behavior files and pick the first one whose filename contains the YYYY-MM-DD token
    behavior_files = find_behavior_files(rat, repo_root=repo_root)
    behavior_file = None
    for p in behavior_files:
        if behavior_date_token in Path(p).name:
            behavior_file = p
            break

    # find ephys subfolders and pick one whose folder name contains YYYYMMDD token
    ephys_subs = find_ephys_subfolders(rat, repo_root=repo_root)
    ephys_folder = None
    for d in ephys_subs:
        if ephys_date_token in Path(d).name:
            ephys_folder = d
            break

    return behavior_file, ephys_folder
