from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    out_dir = Path("data/output_llm_smoke/pdf")
    md_files = sorted(out_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"no md files under {out_dir.resolve()}")

    md = md_files[0]
    text = md.read_text(encoding="utf-8")

    refs = re.findall(
        r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        text,
    )
    md_dir = md.parent

    missing: list[str] = []
    for ref in refs:
        if ref.startswith("http://") or ref.startswith("https://"):
            continue
        if ref.startswith("file://"):
            p = Path(ref[len("file://") :])
        else:
            # relative path -> resolve against md dir
            if (len(ref) >= 3 and ref[1] == ":" and (ref[2] == "\\" or ref[2] == "/")) or ref.startswith("/") or ref.startswith("\\"):
                p = Path(ref)
            else:
                p = (md_dir / ref).resolve()
        if not p.exists():
            missing.append(str(p))

    print("md_path:", str(md))
    print("md_chars:", len(text))
    print("image_refs:", len(refs))
    print("missing:", len(missing))
    if missing:
        print("missing_first:", missing[:10])


if __name__ == "__main__":
    main()

