"""调用 MinerU `mineru-api`（FastAPI）与官方路由兼容的 HTTP 接口。"""

from __future__ import annotations

import io
import json
import mimetypes
import shutil
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from config import AppConfig

import httpx

from src.mineru_zip_layout import normalize_mineru_zip_layout


class MinerUError(Exception):
    """MinerU HTTP 或结果解析错误。"""


_T = TypeVar("_T")


def _sleep_before_http_retry(attempt_idx: int, *, max_sleep: float) -> None:
    """attempt_idx 为 0 表示首次失败后、第二次尝试前；max_sleep 不超过距重试截止的剩余时间。"""
    delay = min(12.0, 0.35 * (2**attempt_idx))
    delay = min(delay, max(0.0, max_sleep))
    if delay > 0:
        time.sleep(delay)


def _httpx_call_with_retry(
    op: Callable[[], _T],
    *,
    what: str,
    retry_duration_sec: float,
    cancel_check: Callable[[], bool] | None = None,
) -> _T:
    """对连接抖动、连不上接口等 ``RequestError``，在 ``retry_duration_sec`` 内持续重试。"""
    budget = max(1.0, float(retry_duration_sec))
    deadline = time.monotonic() + budget
    attempt = 0
    while True:
        try:
            return op()
        except httpx.RequestError as exc:
            now = time.monotonic()
            if now >= deadline:
                secs = int(round(budget))
                raise MinerUError(
                    f"MinerU {what}失败（网络不稳定，已持续重试约 {secs} 秒）: {exc!s}"
                ) from exc
            if cancel_check is not None and cancel_check():
                raise MinerUError("任务已取消") from exc
            remaining = deadline - now
            _sleep_before_http_retry(attempt, max_sleep=remaining)
            attempt += 1


def _form_bool(v: bool) -> str:
    return "true" if v else "false"


def _guess_media_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    if mt:
        return mt
    return "application/octet-stream"


def build_multipart_form_fields(
    *,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: str | None,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    response_format_zip: bool,
    return_original_file: bool,
    start_page_id: int,
    end_page_id: int,
    lang_list: tuple[str, ...],
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = [
        ("backend", backend),
        ("parse_method", parse_method),
        ("formula_enable", _form_bool(formula_enable)),
        ("table_enable", _form_bool(table_enable)),
        ("return_md", _form_bool(return_md)),
        ("return_middle_json", _form_bool(return_middle_json)),
        ("return_model_output", _form_bool(return_model_output)),
        ("return_content_list", _form_bool(return_content_list)),
        ("return_images", _form_bool(return_images)),
        ("response_format_zip", _form_bool(response_format_zip)),
        ("return_original_file", _form_bool(return_original_file)),
        ("start_page_id", str(int(start_page_id))),
        ("end_page_id", str(int(end_page_id))),
    ]
    for lang in lang_list:
        if lang.strip():
            items.append(("lang_list", lang.strip()))
    if server_url and server_url.strip():
        items.append(("server_url", server_url.strip()))
    return items


def build_multipart_request_items(
    *,
    form_fields: list[tuple[str, str]],
    file_name: str,
    file_bytes: bytes,
    media_type: str,
) -> list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]]:
    items: list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]] = []
    for key, value in form_fields:
        # httpx 的 multipart 文本字段使用 (None, value) 形式最稳定。
        items.append((key, (None, value)))
    items.append(("files", (file_name, file_bytes, media_type)))
    return items


def markdown_from_results_json(data: dict[str, Any], prefer_stem: str) -> str:
    results = data.get("results")
    if not isinstance(results, dict):
        raise MinerUError("MinerU 返回 JSON 缺少 results 字段或类型异常")

    if prefer_stem in results:
        entry = results[prefer_stem]
        if isinstance(entry, dict) and "md_content" in entry:
            raw = entry.get("md_content")
            if isinstance(raw, str):
                return raw
            raise MinerUError(f"MinerU results[{prefer_stem!r}].md_content 非字符串")

    if len(results) == 1:
        entry = next(iter(results.values()))
        if isinstance(entry, dict) and isinstance(entry.get("md_content"), str):
            return entry["md_content"]

    raise MinerUError(
        f"无法在 MinerU results 中定位 md_content（stem={prefer_stem!r}，键={list(results)!r}）"
    )


def markdown_from_zip_bytes(zb: bytes, prefer_stem: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            names = zf.namelist()
            preferred = [
                n
                for n in names
                if n.endswith(f"/{prefer_stem}.md") or n.rstrip("/").endswith(f"{prefer_stem}.md")
            ]
            pick = preferred[0] if preferred else None
            if pick is None:
                mds = sorted(n for n in names if n.lower().endswith(".md"))
                if not mds:
                    raise MinerUError("MinerU ZIP 结果中未找到 .md 文件")
                pick = mds[0]
            return zf.read(pick).decode("utf-8")
    except zipfile.BadZipFile as exc:
        raise MinerUError("MinerU 返回的 ZIP 无效") from exc


def _safe_zip_rel_path(name: str) -> Path | None:
    rel = Path(name)
    if rel.is_absolute():
        return None
    if ".." in rel.parts:
        return None
    return rel


def persist_zip_artifacts(
    *,
    zip_bytes: bytes,
    output_path: Path,
    prefer_stem: str,
) -> None:
    out_root = output_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = _safe_zip_rel_path(info.filename)
                if rel is None:
                    continue
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)
    except zipfile.BadZipFile as exc:
        raise MinerUError("MinerU 返回的 ZIP 无效") from exc

    normalize_mineru_zip_layout(out_root, prefer_stem)

    md_paths = sorted(
        (p for p in out_root.rglob("*.md") if p.is_file()),
        key=lambda p: p.as_posix(),
    )

    if not md_paths:
        raise MinerUError("MinerU ZIP 结果中未找到 .md 文件")
    preferred = [
        p for p in md_paths if p.name == f"{prefer_stem}.md" or p.as_posix().endswith(f"/{prefer_stem}.md")
    ]
    chosen = preferred[0] if preferred else sorted(md_paths, key=lambda p: p.as_posix())[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if chosen.resolve() != output_path.resolve():
        shutil.copy2(chosen, output_path)

    # images 以“主 md 同级目录下的 images”为准，统一归并到 out_root/images。
    source_images = chosen.parent / "images"
    keep_images = out_root / "images"
    if source_images.is_dir():
        keep_images.mkdir(parents=True, exist_ok=True)
        for item in source_images.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(source_images)
            dst = keep_images / rel
            try:
                if item.resolve() == dst.resolve():
                    continue
            except OSError:
                pass
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    # 仅保留一个主 md 与 images 目录（若存在），其余 MinerU 过程文件全部清理。
    keep_md = output_path.resolve()

    for child in list(out_root.iterdir()):
        try:
            child_resolved = child.resolve()
        except OSError:
            continue
        if child_resolved == keep_md:
            continue
        if keep_images.is_dir() and child_resolved == keep_images.resolve():
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _parse_error_body(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            msg = data.get("message") or data.get("detail") or data.get("error")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            return json.dumps(data, ensure_ascii=False)[:2000]
    except Exception:  # noqa: BLE001
        pass
    return (resp.text or resp.reason_phrase or "")[:2000]


@dataclass(frozen=True)
class MinerUClientConfig:
    base_url: str
    api_key: str | None
    timeout_sec: float
    poll_interval_sec: float
    max_wait_sec: float
    verify_ssl: bool
    parse_mode: Literal["async", "sync"]
    backend: str
    parse_method: str
    formula_enable: bool
    table_enable: bool
    server_url: str | None
    lang_list: tuple[str, ...]
    return_md: bool
    return_middle_json: bool
    return_model_output: bool
    return_content_list: bool
    return_images: bool
    response_format_zip: bool
    return_original_file: bool
    start_page_id: int
    end_page_id: int
    connect_retry_duration_sec: float


def _emit_processing_stage(
    cb: Callable[[str], None] | None, name: str
) -> None:
    if cb is not None:
        cb(name)


def _headers(api_key: str | None) -> dict[str, str]:
    if api_key and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


def run_mineru_convert(
    *,
    input_path: Path,
    output_path: Path,
    cfg: MinerUClientConfig,
    backend_override: str | None,
    resume_task_id: str | None,
    on_remote_task_id: Callable[[str], None] | None,
    progress_callback: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    on_processing_stage: Callable[[str], None] | None = None,
) -> None:
    base = cfg.base_url.rstrip("/")
    if not base:
        raise MinerUError("未配置 MINERU_BASE_URL（环境变量为空）")

    stem = input_path.stem
    selected_backend = (backend_override or "").strip() or cfg.backend
    # 用户要求“不要过滤 MinerU 返回内容”：
    # 当返回 ZIP 时，尽可能请求 MinerU 包含全部可返回的产物文件。
    if cfg.response_format_zip:
        return_md = True
        return_middle_json = True
        return_model_output = True
        return_content_list = True
        return_images = True
        return_original_file = True
    else:
        return_md = cfg.return_md
        return_middle_json = cfg.return_middle_json
        return_model_output = cfg.return_model_output
        return_content_list = cfg.return_content_list
        return_images = cfg.return_images
        return_original_file = cfg.return_original_file
    form = build_multipart_form_fields(
        backend=selected_backend,
        parse_method=cfg.parse_method,
        formula_enable=cfg.formula_enable,
        table_enable=cfg.table_enable,
        server_url=cfg.server_url,
        return_md=return_md,
        return_middle_json=return_middle_json,
        return_model_output=return_model_output,
        return_content_list=return_content_list,
        return_images=return_images,
        response_format_zip=cfg.response_format_zip,
        return_original_file=return_original_file and cfg.response_format_zip,
        start_page_id=cfg.start_page_id,
        end_page_id=cfg.end_page_id,
        lang_list=cfg.lang_list,
    )

    timeout = httpx.Timeout(cfg.timeout_sec, connect=min(30.0, cfg.timeout_sec))
    headers = _headers(cfg.api_key)

    with httpx.Client(
        base_url=base,
        headers=headers,
        timeout=timeout,
        verify=cfg.verify_ssl,
    ) as client:
        file_bytes = input_path.read_bytes()
        multipart_items = build_multipart_request_items(
            form_fields=form,
            file_name=input_path.name,
            file_bytes=file_bytes,
            media_type=_guess_media_type(input_path),
        )

        if cfg.parse_mode == "sync":
            _run_sync_parse(
                client,
                multipart_items=multipart_items,
                stem=stem,
                output_path=output_path,
                connect_retry_duration_sec=cfg.connect_retry_duration_sec,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
                on_processing_stage=on_processing_stage,
            )
            return

        _run_async_parse(
            client,
            multipart_items=multipart_items,
            stem=stem,
            output_path=output_path,
            cfg=cfg,
            resume_task_id=resume_task_id,
            on_remote_task_id=on_remote_task_id,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            on_processing_stage=on_processing_stage,
        )


def _run_sync_parse(
    client: httpx.Client,
    *,
    multipart_items: list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]],
    stem: str,
    output_path: Path,
    connect_retry_duration_sec: float,
    cancel_check: Callable[[], bool] | None,
    progress_callback: Callable[[int, int], None] | None,
    on_processing_stage: Callable[[str], None] | None,
) -> None:
    if progress_callback:
        progress_callback(2, 100)
    _emit_processing_stage(on_processing_stage, "mineru_upload")
    resp = _httpx_call_with_retry(
        lambda: client.post("/file_parse", files=multipart_items),
        what="同步解析",
        retry_duration_sec=connect_retry_duration_sec,
        cancel_check=cancel_check,
    )
    if resp.status_code == 409:
        raise MinerUError(_parse_error_body(resp))
    if resp.status_code == 503:
        raise MinerUError(_parse_error_body(resp))
    resp.raise_for_status()
    if progress_callback:
        progress_callback(8, 100)
    ct = (resp.headers.get("content-type") or "").lower()
    body = resp.content
    if "application/zip" in ct or body.startswith(b"PK\x03\x04"):
        _emit_processing_stage(on_processing_stage, "mineru_materialize")
        persist_zip_artifacts(zip_bytes=body, output_path=output_path, prefer_stem=stem)
    else:
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise MinerUError("MinerU /file_parse 返回非 JSON 且非 ZIP") from exc
        md = markdown_from_results_json(data, stem)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_processing_stage(on_processing_stage, "mineru_materialize")
        output_path.write_text(md, encoding="utf-8")
    if progress_callback:
        progress_callback(100, 100)


def _run_async_parse(
    client: httpx.Client,
    *,
    multipart_items: list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]],
    stem: str,
    output_path: Path,
    cfg: MinerUClientConfig,
    resume_task_id: str | None,
    on_remote_task_id: Callable[[str], None] | None,
    progress_callback: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    on_processing_stage: Callable[[str], None] | None,
) -> None:
    if progress_callback:
        progress_callback(1, 100)
    task_id = str(resume_task_id or "").strip()
    if not task_id:
        _emit_processing_stage(on_processing_stage, "mineru_upload")
        resp = _httpx_call_with_retry(
            lambda: client.post("/tasks", files=multipart_items),
            what="提交解析任务",
            retry_duration_sec=cfg.connect_retry_duration_sec,
            cancel_check=cancel_check,
        )
        if resp.status_code not in (200, 202):
            raise MinerUError(_parse_error_body(resp))
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise MinerUError("MinerU /tasks 返回非 JSON") from exc
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise MinerUError(f"MinerU /tasks 未返回 task_id: {payload!r}")
        if on_remote_task_id is not None:
            on_remote_task_id(task_id)

    deadline = time.monotonic() + max(30.0, cfg.max_wait_sec)
    poll = max(0.2, cfg.poll_interval_sec)
    phase = 0
    remote_stage_emitted = False

    while time.monotonic() < deadline:
        if cancel_check is not None and cancel_check():
            raise MinerUError("任务已取消")

        if not remote_stage_emitted:
            _emit_processing_stage(on_processing_stage, "mineru_remote")
            remote_stage_emitted = True

        st_resp = _httpx_call_with_retry(
            lambda: client.get(f"/tasks/{task_id}"),
            what="查询任务状态",
            retry_duration_sec=cfg.connect_retry_duration_sec,
            cancel_check=cancel_check,
        )
        if st_resp.status_code == 404:
            raise MinerUError("MinerU 任务不存在（404）")
        st_resp.raise_for_status()
        try:
            st = st_resp.json()
        except json.JSONDecodeError as exc:
            raise MinerUError("MinerU 任务状态非 JSON") from exc
        status = st.get("status") if isinstance(st, dict) else None

        if status == "failed":
            err = st.get("error") if isinstance(st, dict) else None
            raise MinerUError(str(err).strip() if err else "MinerU 任务失败")

        if status == "completed":
            break

        phase += 1
        if progress_callback:
            # 轮询阶段仅占 2–10%，避免短时间等待却显示八九十
            pct = min(10, 2 + min(phase // 12, 8))
            progress_callback(pct, 100)
        time.sleep(poll)
    else:
        raise MinerUError(f"MinerU 任务在 {cfg.max_wait_sec:.0f}s 内未完成")

    if progress_callback:
        progress_callback(18, 100)

    res_resp = _httpx_call_with_retry(
        lambda: client.get(f"/tasks/{task_id}/result"),
        what="拉取解析结果",
        retry_duration_sec=cfg.connect_retry_duration_sec,
        cancel_check=cancel_check,
    )
    if res_resp.status_code == 409:
        raise MinerUError(_parse_error_body(res_resp))
    if res_resp.status_code == 202:
        raise MinerUError("MinerU 任务已完成但结果尚未就绪（202），请稍后重试")
    res_resp.raise_for_status()

    body = res_resp.content
    ct = (res_resp.headers.get("content-type") or "").lower()
    if "application/zip" in ct or body.startswith(b"PK\x03\x04"):
        _emit_processing_stage(on_processing_stage, "mineru_materialize")
        persist_zip_artifacts(zip_bytes=body, output_path=output_path, prefer_stem=stem)
    else:
        try:
            data = res_resp.json()
        except json.JSONDecodeError as exc:
            raise MinerUError("MinerU /tasks/.../result 返回非 JSON 且非 ZIP") from exc
        md = markdown_from_results_json(data, stem)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_processing_stage(on_processing_stage, "mineru_materialize")
        output_path.write_text(md, encoding="utf-8")
    if progress_callback:
        progress_callback(100, 100)


def mineru_client_config_from_app(app: "AppConfig") -> MinerUClientConfig:
    return MinerUClientConfig(
        base_url=app.mineru_base_url,
        api_key=app.mineru_api_key,
        timeout_sec=app.mineru_timeout_sec,
        poll_interval_sec=app.mineru_poll_interval_sec,
        max_wait_sec=app.mineru_max_wait_sec,
        connect_retry_duration_sec=app.mineru_connect_retry_duration_sec,
        verify_ssl=app.mineru_verify_ssl,
        parse_mode=app.mineru_parse_mode,
        backend=app.mineru_backend,
        parse_method=app.mineru_parse_method,
        formula_enable=app.mineru_formula_enable,
        table_enable=app.mineru_table_enable,
        server_url=app.mineru_server_url,
        lang_list=app.mineru_lang_list,
        return_md=app.mineru_return_md,
        return_middle_json=app.mineru_return_middle_json,
        return_model_output=app.mineru_return_model_output,
        return_content_list=app.mineru_return_content_list,
        return_images=app.mineru_return_images,
        response_format_zip=app.mineru_response_format_zip,
        return_original_file=app.mineru_return_original_file,
        start_page_id=app.mineru_start_page_id,
        end_page_id=app.mineru_end_page_id,
    )
