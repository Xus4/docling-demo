"""MinerU ZIP 解压后的目录布局归一化（外层同名 md 丢弃 + 内层目录扁平化）。"""

from __future__ import annotations

import shutil
from pathlib import Path


def _merge_dir_into(src: Path, dst: Path) -> None:
    """将目录 src 合并进已存在的目录 dst，然后删除 src。"""
    if not src.is_dir() or not dst.is_dir():
        raise ValueError("_merge_dir_into expects two directories")
    for child in list(src.iterdir()):
        target = dst / child.name
        if child.is_dir() and target.is_dir():
            _merge_dir_into(child, target)
        elif child.is_dir() and not target.exists():
            shutil.move(str(child), str(target))
        elif child.is_file():
            if target.exists() and target.is_file():
                target.unlink()
            shutil.move(str(child), str(target))
        else:
            shutil.move(str(child), str(target))
    shutil.rmtree(src, ignore_errors=True)


def normalize_mineru_zip_layout(out_root: Path, prefer_stem: str) -> bool:
    """若检测到 MinerU 典型「外层 {stem}.md + 内层 {stem}/」布局，则归一化并返回 True；否则不改动并返回 False。

    归一化步骤：
    1. 删除外层 ``out_root/{prefer_stem}.md``（若存在）。
    2. 将 ``out_root/{prefer_stem}/`` 下所有条目移至 ``out_root/``（目录冲突时合并）。
    3. 删除已空的内层目录。

    主文件最终应在 ``out_root/{prefer_stem}.md``，与 ``output_path`` 约定一致。
    """
    stem = prefer_stem.strip()
    if not stem:
        return False

    outer_md = out_root / f"{stem}.md"
    inner_dir = out_root / stem
    inner_md = inner_dir / f"{stem}.md"

    if not inner_dir.is_dir() or not inner_md.is_file():
        return False

    if outer_md.is_file():
        try:
            outer_md.unlink()
        except OSError:
            return False

    # 将 inner_dir 下所有条目移到 out_root
    for child in list(inner_dir.iterdir()):
        dest = out_root / child.name
        if child.is_dir() and dest.exists() and dest.is_dir():
            _merge_dir_into(child, dest)
        elif dest.exists() and dest.is_file() and child.is_file():
            dest.unlink(missing_ok=True)
            shutil.move(str(child), str(dest))
        else:
            shutil.move(str(child), str(dest))

    shutil.rmtree(inner_dir, ignore_errors=True)
    return True
