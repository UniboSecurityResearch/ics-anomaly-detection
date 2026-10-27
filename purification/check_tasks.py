"""Check task files for purification.sbatch before queueing them.

    python -m purification.check_tasks TASKFILE [TASKFILE ...]   (from the repository root)

Every line is "<output folder> <script> <arguments>". Checks: the script is
main_purification.py, its argument parser accepts the arguments (the same parser the
job will use, so a missing required option such as --out fails here instead of on the
cluster), --out equals the output folder, and no two lines share an output folder.
"""

import shlex
import sys


def check(paths) -> int:
    import main_purification
    seen, errors, n = {}, 0, 0
    for path in paths:
        with open(path) as fd:
            for k, line in enumerate(fd, 1):
                if not line.strip():
                    continue
                n += 1
                parts = shlex.split(line)
                folder, script, argv = parts[0], parts[1], parts[2:]
                where = f"{path}:{k}"
                if script != "main_purification.py":
                    print(f"ERROR {where}: unexpected script {script}")
                    errors += 1
                    continue
                try:
                    args = main_purification.parse_args(argv)
                except SystemExit:
                    print(f"ERROR {where}: arguments rejected by main_purification.py (see above)")
                    errors += 1
                    continue
                if args.out != folder:
                    print(f"ERROR {where}: --out {args.out} differs from the task folder {folder}")
                    errors += 1
                if folder in seen:
                    print(f"ERROR {where}: output folder already used at {seen[folder]}")
                    errors += 1
                seen[folder] = where
    print(f"checked {n} tasks in {len(paths)} files: {'OK' if not errors else str(errors) + ' errors'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1:]))
