import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert local PDFs to Markdown.")
    parser.add_argument("--input-dir", required=True, help="输入目录，例如 ./docs")
    parser.add_argument("--output-dir", required=True, help="输出目录，例如 ./out_md")
    parser.add_argument("--pattern", default="*.pdf", help="匹配文件模式，默认 *.pdf")
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

    total = len(pdfs)
    for idx, pdf_path in enumerate(pdfs, start=1):
        rel = pdf_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path = out_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{total}] converting: {pdf_path}")
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        out_path.write_text(markdown, encoding="utf-8")
        print(f"           => {out_path}")

    print(f"All done. Output in: {output_dir}")


if __name__ == "__main__":
    main()

