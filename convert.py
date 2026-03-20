import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a local PDF to Markdown using docling."
    )
    parser.add_argument("--input", required=True, help="本地 PDF 路径，例如 ./input.pdf")
    parser.add_argument("--output", required=True, help="输出 Markdown 路径，例如 ./out.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    output_path = Path(args.output)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(str(input_path))

    markdown = result.document.export_to_markdown()
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Done: wrote markdown to {output_path}")


if __name__ == "__main__":
    main()

