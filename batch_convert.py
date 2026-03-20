import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert local PDFs to Markdown.")
    parser.add_argument("--input-dir", required=True, help="输入目录，例如 ./docs")
    parser.add_argument("--output-dir", required=True, help="输出目录，例如 ./out_md")
    parser.add_argument("--pattern", default="*.pdf", help="匹配文件模式，默认 *.pdf")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出的 .md 已存在，则跳过该 PDF（默认会覆盖/重新转）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="强制覆盖已存在的输出文件（与 --skip-existing 可同时传，但覆盖优先生效）",
    )
    parser.add_argument(
        "--on-error",
        choices=["continue", "stop"],
        default="continue",
        help="单个文件转换失败时：continue=继续下一个，stop=中断",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理多少个 PDF；0 表示不限制",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="可选：写入日志到该文件路径（例如 ./convert.log）。留空则不写文件日志。",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入目录: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()

    pdfs = sorted(input_dir.rglob(args.pattern))
    if not pdfs:
        print(f"未找到匹配文件: {input_dir} / {args.pattern}")
        return

    if args.max_files and args.max_files > 0:
        pdfs = pdfs[: args.max_files]

    total = len(pdfs)
    log_path = Path(args.log_file) if args.log_file else None

    def log_line(s: str) -> None:
        print(s)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(s + "\n")

    for idx, pdf_path in enumerate(pdfs, start=1):
        rel = pdf_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path = out_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        should_skip = out_path.exists() and args.skip_existing and not args.overwrite
        if should_skip:
            percent = (idx - 1) / total * 100 if total else 0
            log_line(f"[{idx}/{total} {percent:5.1f}%] skip (exists): {pdf_path}")
            continue

        percent = idx / total * 100 if total else 0
        log_line(f"[{idx}/{total} {percent:5.1f}%] converting: {pdf_path}")
        try:
            result = converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()
            out_path.write_text(markdown, encoding="utf-8")
            log_line(f"           => {out_path}")
        except Exception as e:
            log_line(f"           !! failed: {pdf_path}")
            log_line(f"           !! error : {type(e).__name__}: {e}")
            if args.on_error == "stop":
                raise

    print(f"All done. Output in: {output_dir}")


if __name__ == "__main__":
    main()

