# docling-demo（PDF -> Markdown）

这是一个最小可运行的 demo：使用 `docling` 将**本地 PDF** 转成 **Markdown**。

## 1. 准备

把你的 PDF 放到项目目录，例如：`./input.pdf`（也可以放到子目录，再把路径传给 `--input`）。

## 2. 安装依赖（Windows / PowerShell）

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 3. 运行

```powershell
python convert.py --input "input.pdf" --output "out.md"
```

## 4. 你会得到

`out.md`：docling 生成的 Markdown 内容（UTF-8 编码）。

## 批量转换（可选）

如果你有一个目录，包含多个 PDF，可以用 `batch_convert.py` 递归转换：

```powershell
python batch_convert.py --input-dir "./input_pdfs" --output-dir "./out_md"
```

## 命令行增强（推荐）

如果你希望“已生成的输出自动跳过、失败可继续、可选覆盖”，可以用下面这条：

```powershell
python batch_convert.py --input-dir "C:\Users\lenovo\Documents\水利\附件1、2020年黄河、渭河、沁河超标准洪水预案\附件1、2020年黄河、渭河、沁河超标准洪水预案" --output-dir "./out_md" --skip-existing --on-error continue --log-file "./out_md/convert.log"
```

