import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
EXERCISE_ROOT = ROOT / "cute_exercise"

EXERCISE_ENTRY_RE = re.compile(
    r"^(?P<number>\d+)\. \[(?P<title>[^\]]+)\]\((?P<path>cute_exercise/ex(?P<dir_number>\d+)_[^)]+)/\)$"
)


def readme_exercise_entries() -> list[tuple[int, str, int]]:
    entries = []
    in_exercises = False

    for line in README.read_text().splitlines():
        if line == "## Exercises":
            in_exercises = True
            continue
        if in_exercises and line.startswith("## "):
            break
        if not in_exercises:
            continue

        match = EXERCISE_ENTRY_RE.match(line)
        if match:
            entries.append(
                (
                    int(match.group("number")),
                    match.group("path"),
                    int(match.group("dir_number")),
                )
            )

    return entries


def exercise_readme_dirs() -> list[str]:
    dirs = [
        path.parent
        for path in EXERCISE_ROOT.glob("ex[0-9]*_*/README.md")
    ]
    return [
        str(path.relative_to(ROOT))
        for path in sorted(
            dirs,
            key=lambda path: int(re.match(r"ex(\d+)_", path.name).group(1)),
        )
    ]


def test_top_level_readme_exercise_index_matches_exercise_dirs() -> None:
    entries = readme_exercise_entries()
    entry_numbers = [number for number, _, _ in entries]
    entry_paths = [path for _, path, _ in entries]
    dir_numbers = [dir_number for _, _, dir_number in entries]

    assert entry_paths == exercise_readme_dirs()
    assert len(entry_numbers) == len(set(entry_numbers))
    assert len(dir_numbers) == len(set(dir_numbers))
    assert len(entry_paths) == len(set(entry_paths))

    for number, path, dir_number in entries:
        assert number == dir_number, f"{path} is listed as {number}"
