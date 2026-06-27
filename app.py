import base64
import hashlib
import html
import json
import re
import traceback
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from grader_module import (
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_RUBRIC,
    GradingError,
    extract_score,
    recognize_and_grade_image,
)


APP_NAME = "高中英语作文批改助手"
VERSION = "0625b_fixed"
DEFAULT_STANDARD_NAME = "默认读后续写评分标准"
STEPS = [
    "考试和题目信息录入",
    "评分标准",
    "上传并批量批改",
    "结果核对与导出",
]

LOG_DIR = Path("logs")
ERROR_LOG_JSONL = LOG_DIR / "processing_errors.jsonl"
ERROR_LOG_TEXT = LOG_DIR / "processing_errors.log"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="✍️",
    layout="wide",
)


def inject_custom_css() -> None:
    """Use a neutral blue UI style so normal actions do not look like error states."""
    st.markdown(
        """
        <style>
        :root {
            --app-primary: #2563eb;
            --app-primary-hover: #1d4ed8;
            --app-success-bg: #ecfdf5;
            --app-info-bg: #eff6ff;
        }
        div.stButton > button[kind="primary"] {
            background-color: var(--app-primary) !important;
            border-color: var(--app-primary) !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: var(--app-primary-hover) !important;
            border-color: var(--app-primary-hover) !important;
            color: #ffffff !important;
        }
        div.stProgress > div > div > div > div {
            background-color: var(--app-primary) !important;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 10px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_safe_key(*parts) -> str:
    text = "_".join(str(p) for p in parts)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def guess_student_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(\d+)", stem)
    if match:
        return match.group(1)
    return stem


def clamp_score(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        score = float(text)
    except ValueError:
        return text
    if score < 0:
        score = 0
    if score > 25:
        score = 25
    if score.is_integer():
        return str(int(score))
    return str(score)


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def safe_preview(text: Optional[str], limit: int = 2000) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...<已截断，原长度 {len(text)} 字符>"


def append_error_log(
    *,
    filename: str,
    student_id: str,
    stage: str,
    error_code: str,
    message: str,
    model: str,
    num_ctx: int,
    extra: Optional[Dict] = None,
    exc: Optional[BaseException] = None,
) -> None:
    """Append a structured JSONL log and a readable text log for model-processing failures."""
    ensure_log_dir()
    record = {
        "time": now_str(),
        "version": VERSION,
        "file_name": filename,
        "student_id": student_id,
        "stage": stage,
        "error_code": error_code,
        "message": message,
        "model": model,
        "num_ctx": int(num_ctx),
        "extra": extra or {},
    }
    if exc is not None:
        record["exception_type"] = exc.__class__.__name__
        record["traceback"] = traceback.format_exc()

    with ERROR_LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ERROR_LOG_TEXT.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"time: {record['time']}\n")
        f.write(f"version: {VERSION}\n")
        f.write(f"file_name: {filename}\n")
        f.write(f"student_id: {student_id}\n")
        f.write(f"stage: {stage}\n")
        f.write(f"error_code: {error_code}\n")
        f.write(f"message: {message}\n")
        f.write(f"model: {model}\n")
        f.write(f"num_ctx: {num_ctx}\n")
        if extra:
            f.write("extra:\n")
            f.write(json.dumps(extra, ensure_ascii=False, indent=2))
            f.write("\n")
        if exc is not None:
            f.write("traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n")


def read_log_file(path: Path) -> Optional[bytes]:
    try:
        if path.exists() and path.is_file():
            return path.read_bytes()
    except Exception:
        return None
    return None


def init_session_state() -> None:
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    if "exam" not in st.session_state:
        st.session_state.exam = {
            "exam_name": "",
            "essay_prompt": "",
            "original_text": "",
            "paragraph1_start": "",
            "paragraph2_start": "",
            "updated_at": "",
        }

    if "rubrics" not in st.session_state:
        st.session_state.rubrics = {
            DEFAULT_STANDARD_NAME: DEFAULT_RUBRIC,
        }

    if "selected_rubric_name" not in st.session_state:
        st.session_state.selected_rubric_name = DEFAULT_STANDARD_NAME

    if "answers" not in st.session_state:
        st.session_state.answers = {}

    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL

    if "num_ctx" not in st.session_state:
        st.session_state.num_ctx = DEFAULT_NUM_CTX

    if "last_batch_summary" not in st.session_state:
        st.session_state.last_batch_summary = ""

    if "snapshot_zip_data" not in st.session_state:
        st.session_state.snapshot_zip_data = None

    if "snapshot_zip_generated_at" not in st.session_state:
        st.session_state.snapshot_zip_generated_at = ""


def normalize_opening_sentence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(?:para\.?\s*[12]|paragraph\s*[12]|第一段|第二段|第[12]段|续写第一段|续写第二段)\s*[:：.．、-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def extract_prompt_parts(essay_prompt: str) -> Tuple[str, str, str, List[str]]:
    """
    Extract original text and two continuation opening sentences from a combined prompt.

    Supported marker formats:
    1. / I looked at the old photo...
    2. / on one line and the opening sentence on the next non-empty line.
    3. 第一段：... / 第二段：...
    """
    text = essay_prompt.strip()
    if not text:
        return "", "", "", ["作文题目为空。"]

    raw_lines = text.splitlines()
    openings: List[Tuple[int, str]] = []
    consumed_indices = set()

    marker_only_pattern = r"^\s*[/／|｜#＃>*]{1,3}\s*$"
    marker_same_line_pattern = r"^\s*[/／|｜#＃>*]{1,3}\s*(.+)$"
    labeled_line_pattern = (
        r"^\s*(?:para\.?\s*[12]|paragraph\s*[12]|第一段续写开头句|第二段续写开头句|"
        r"第一段|第二段|第[12]段|续写第一段|续写第二段)\s*[:：.．、-]\s*(.+)$"
    )

    for index, line in enumerate(raw_lines):
        if index in consumed_indices:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        if re.match(marker_only_pattern, stripped):
            next_index = index + 1
            while next_index < len(raw_lines) and not raw_lines[next_index].strip():
                next_index += 1
            if next_index < len(raw_lines):
                opening = normalize_opening_sentence(raw_lines[next_index])
                if opening:
                    openings.append((index, opening))
                    consumed_indices.add(next_index)
            continue

        same_line_match = re.match(marker_same_line_pattern, stripped, flags=re.IGNORECASE)
        if same_line_match:
            opening = normalize_opening_sentence(same_line_match.group(1))
            if opening:
                openings.append((index, opening))
            continue

        label_match = re.match(labeled_line_pattern, stripped, flags=re.IGNORECASE)
        if label_match:
            opening = normalize_opening_sentence(label_match.group(1))
            if opening:
                openings.append((index, opening))
            continue

    warnings: List[str] = []

    if len(openings) >= 2:
        first_marker_index = openings[0][0]
        original_lines = raw_lines[:first_marker_index]
        original_text = "\n".join(original_lines).strip()
        paragraph1_start = openings[0][1]
        paragraph2_start = openings[1][1]
        return original_text, paragraph1_start, paragraph2_start, warnings

    para1 = ""
    para2 = ""
    label_para1 = re.search(
        r"(?:第一段续写开头句|第一段|Para\.?\s*1|Paragraph\s*1)\s*[:：]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    label_para2 = re.search(
        r"(?:第二段续写开头句|第二段|Para\.?\s*2|Paragraph\s*2)\s*[:：]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if label_para1:
        para1 = normalize_opening_sentence(label_para1.group(1))
    if label_para2:
        para2 = normalize_opening_sentence(label_para2.group(1))

    if para1 and para2:
        original_text = re.split(
            r"(?:第一段续写开头句|第一段|Para\.?\s*1|Paragraph\s*1)\s*[:：]",
            text,
            flags=re.IGNORECASE,
        )[0].strip()
        return original_text, para1, para2, warnings

    warnings.append(
        "未能自动提取两段续写开头句。建议在两段续写开头句前分别另起一行，并以 / 开头。"
    )
    return text, "", "", warnings


NON_BLOCKING_PARSE_WARNING_MARKERS = [
    "模型返回总分",
    "与五项合计",
    "不一致",
    "已按五项合计修正",
]


def is_score_mismatch_warning(warning: str) -> bool:
    """Return True when the warning only says total score was corrected by sub-score sum.

    This warning is expected after 0625a: the backend keeps enforcing
    total_score = sum(five dimension scores). It should not make the frontend
    show “需核对” as long as five sub-scores are present and in range.
    """
    text = str(warning or "")
    return all(marker in text for marker in NON_BLOCKING_PARSE_WARNING_MARKERS)


def blocking_parse_warnings(answer: Dict) -> List[str]:
    """Warnings that should make the frontend display “需核对”.

    Non-blocking case: LLM gave a total score inconsistent with five sub-scores,
    and the backend already corrected the final score to the five-score sum.

    Blocking cases remain visible: missing five sub-scores, out-of-range values,
    missing grading comment, explicit model review flag, or any unknown parse note.
    """
    warnings = answer.get("parse_warnings", []) or []
    blocking = []
    for warning in warnings:
        if is_score_mismatch_warning(str(warning)):
            continue
        blocking.append(str(warning))
    return blocking


def non_blocking_parse_warnings(answer: Dict) -> List[str]:
    warnings = answer.get("parse_warnings", []) or []
    return [str(warning) for warning in warnings if is_score_mismatch_warning(str(warning))]


def needs_manual_review(answer: Dict) -> bool:
    return bool(blocking_parse_warnings(answer))


def answer_status(answer: Dict) -> str:
    explicit_status = answer.get("status", "")
    if explicit_status == "processing":
        return "处理中"
    if explicit_status == "queued":
        return "排队中"
    if answer.get("error"):
        return "处理失败"
    if answer.get("grading_comment") and answer.get("score") != "":
        if needs_manual_review(answer):
            return "需核对"
        return "已完成"
    if answer.get("grading_comment"):
        return "需核对"
    if answer.get("image_bytes"):
        return "待处理"
    return "未上传"


def status_counts() -> Dict[str, int]:
    total = len(st.session_state.answers)
    finished = sum(1 for item in st.session_state.answers.values() if answer_status(item) == "已完成")
    review = sum(1 for item in st.session_state.answers.values() if answer_status(item) == "需核对")
    processing = sum(1 for item in st.session_state.answers.values() if answer_status(item) in {"处理中", "排队中"})
    failed = sum(1 for item in st.session_state.answers.values() if answer_status(item) == "处理失败")
    pending = max(0, total - finished - review - processing - failed)
    done_like = finished + review + failed
    return {
        "total": total,
        "finished": finished,
        "review": review,
        "processing": processing,
        "failed": failed,
        "pending": pending,
        "done_like": done_like,
    }


def add_uploaded_files(uploaded_files) -> Tuple[int, int]:
    added_count = 0
    skipped_count = 0

    for uploaded_file in uploaded_files or []:
        filename = uploaded_file.name
        if filename in st.session_state.answers:
            skipped_count += 1
            continue

        file_bytes = uploaded_file.getvalue()
        st.session_state.answers[filename] = {
            "file_name": filename,
            "image_bytes": file_bytes,
            "student_id": guess_student_id_from_filename(filename),
            "recognized_text": "",
            "recognition_note": "",
            "grading_comment": "",
            "score": "",
            "raw_output": "",
            "error": "",
            "error_code": "",
            "error_stage": "",
            "parse_warnings": [],
            "ollama_metadata": {},
            "status": "pending",
            "updated_at": "",
        }
        added_count += 1

    return added_count, skipped_count


def build_status_dataframe() -> pd.DataFrame:
    summary = []
    for filename, answer in st.session_state.answers.items():
        summary.append(
            {
                "文件名": filename,
                "学生编号": answer.get("student_id", ""),
                "状态": answer_status(answer),
                "作文得分": answer.get("score", ""),
                "更新时间": answer.get("updated_at", ""),
                "失败原因": answer.get("error", ""),
            }
        )
    return pd.DataFrame(summary)


def build_export_dataframe() -> pd.DataFrame:
    rows = []
    exam_name = st.session_state.exam.get("exam_name", "")
    rubric_name = st.session_state.selected_rubric_name

    for index, filename in enumerate(st.session_state.answers.keys(), start=1):
        answer = st.session_state.answers[filename]
        rows.append(
            {
                "序号": index,
                "考试名称": exam_name,
                "学生编号": answer.get("student_id", ""),
                "评分标准名称": rubric_name,
                "评分意见": answer.get("grading_comment", ""),
                "作文得分": answer.get("score", ""),
                "更新时间": answer.get("updated_at", ""),
            }
        )

    return pd.DataFrame(rows)


def go_next() -> None:
    st.session_state.current_step = min(len(STEPS) - 1, st.session_state.current_step + 1)
    st.rerun()


def go_prev() -> None:
    st.session_state.current_step = max(0, st.session_state.current_step - 1)
    st.rerun()


def render_step_nav() -> None:
    col_prev, col_spacer, col_next = st.columns([1, 5, 1])
    with col_prev:
        if st.session_state.current_step > 0:
            if st.button("上一步", use_container_width=True):
                go_prev()
    with col_next:
        if st.session_state.current_step < len(STEPS) - 1:
            if st.button("下一步", type="primary", use_container_width=True):
                go_next()


def render_sidebar_progress(slots: Optional[Dict[str, object]] = None) -> None:
    counts = status_counts()
    total = counts["total"]
    progress_value = counts["done_like"] / total if total else 0

    if slots and slots.get("metrics") and slots.get("progress"):
        with slots["metrics"].container():
            st.write(f"已上传：{total}")
            st.write(f"已完成：{counts['finished']}")
            st.write(f"需核对：{counts['review']}")
            st.write(f"处理中：{counts['processing']}")
            st.write(f"待处理：{counts['pending']}")
            st.write(f"失败：{counts['failed']}")
        slots["progress"].progress(progress_value)


def render_sidebar() -> Dict[str, object]:
    with st.sidebar:
        st.header("当前步骤")
        for index, step_name in enumerate(STEPS):
            marker = "▶" if index == st.session_state.current_step else "○"
            st.write(f"{marker} {index + 1}. {step_name}")

        st.divider()
        st.caption("处理进度")
        metrics_slot = st.empty()
        progress_slot = st.empty()
        slots = {"metrics": metrics_slot, "progress": progress_slot}
        render_sidebar_progress(slots)
        return slots


def render_step_1() -> None:
    st.subheader("考试和题目信息录入")

    with st.form("exam_form"):
        exam_name = st.text_input(
            "考试名称",
            value=st.session_state.exam.get("exam_name", ""),
            placeholder="例如：Sample High School English Writing Task",
        )
        essay_prompt = st.text_area(
            "作文题目",
            value=st.session_state.exam.get("essay_prompt", ""),
            height=420,
            placeholder=(
                "请粘贴原文阅读材料和两段续写开头句。\n\n"
                "建议格式：\n"
                "原文阅读材料……\n\n"
                "/ 第一段续写开头句……\n"
                "/ 第二段续写开头句……"
            ),
        )
        save = st.form_submit_button("保存考试和题目信息", type="primary")

    if save:
        original_text, paragraph1_start, paragraph2_start, warnings = extract_prompt_parts(essay_prompt)
        st.session_state.exam.update(
            {
                "exam_name": exam_name.strip(),
                "essay_prompt": essay_prompt.strip(),
                "original_text": original_text,
                "paragraph1_start": paragraph1_start,
                "paragraph2_start": paragraph2_start,
                "updated_at": now_str(),
            }
        )
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("已保存并完成题目结构提取。")

    if st.session_state.exam.get("essay_prompt"):
        original_text, paragraph1_start, paragraph2_start, warnings = extract_prompt_parts(
            st.session_state.exam.get("essay_prompt", "")
        )
        st.markdown("### 后台提取结果")
        if warnings:
            for warning in warnings:
                st.warning(warning)
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("原文阅读材料", original_text, height=220, disabled=True)
        with col2:
            st.text_area("第一段续写开头句", paragraph1_start, height=90, disabled=True)
            st.text_area("第二段续写开头句", paragraph2_start, height=90, disabled=True)

    render_step_nav()


def render_step_2() -> None:
    st.subheader("评分标准")

    rubric_names = list(st.session_state.rubrics.keys())
    if st.session_state.selected_rubric_name not in rubric_names:
        st.session_state.selected_rubric_name = rubric_names[0]

    selected_name = st.selectbox(
        "当前使用的评分标准",
        options=rubric_names,
        index=rubric_names.index(st.session_state.selected_rubric_name),
    )
    st.session_state.selected_rubric_name = selected_name

    edited_text = st.text_area(
        "评分标准内容",
        value=st.session_state.rubrics[selected_name],
        height=420,
    )

    col_save_current, col_save_new = st.columns(2)
    with col_save_current:
        if st.button("保存当前评分标准", type="primary", use_container_width=True):
            st.session_state.rubrics[selected_name] = edited_text.strip()
            st.success("已保存当前评分标准。")
    with col_save_new:
        new_name = st.text_input("另存为新评分标准名称", placeholder="例如：校内作文评分标准-0620b")
        if st.button("保存为新标准", use_container_width=True):
            if not new_name.strip():
                st.warning("请输入新评分标准名称。")
            else:
                st.session_state.rubrics[new_name.strip()] = edited_text.strip()
                st.session_state.selected_rubric_name = new_name.strip()
                st.success("已保存为新的评分标准。")
                st.rerun()

    render_step_nav()


def render_log_downloads() -> None:
    st.markdown("### 错误日志")
    st.caption("当图片处理失败时，系统会在本地 logs 目录写入日志。页面失败原因用于快速查看，日志文件用于定位具体阶段和模型返回内容。")

    col1, col2 = st.columns(2)
    jsonl_data = read_log_file(ERROR_LOG_JSONL)
    text_data = read_log_file(ERROR_LOG_TEXT)
    with col1:
        if jsonl_data:
            st.download_button(
                "下载 JSONL 错误日志",
                data=jsonl_data,
                file_name="processing_errors.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )
        else:
            st.button("暂无 JSONL 错误日志", disabled=True, use_container_width=True)
    with col2:
        if text_data:
            st.download_button(
                "下载文本错误日志",
                data=text_data,
                file_name="processing_errors.log",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("暂无文本错误日志", disabled=True, use_container_width=True)


def update_answer_from_success(answer: Dict, result: Dict) -> None:
    answer["recognized_text"] = result.get("recognized_text", "")
    answer["recognition_note"] = result.get("recognition_note", "")
    answer["grading_comment"] = result.get("grading_comment", "")
    answer["raw_output"] = result.get("raw_output", "")
    answer["score"] = clamp_score(result.get("score"))
    answer["error"] = ""
    answer["error_code"] = ""
    answer["error_stage"] = ""
    answer["parse_warnings"] = result.get("parse_warnings", []) or []
    answer["ollama_metadata"] = result.get("ollama_metadata", {}) or {}
    answer["status"] = "review_required" if needs_manual_review(answer) else "done"
    answer["updated_at"] = now_str()


def update_answer_from_error(answer: Dict, exc: BaseException) -> Tuple[str, str, Dict]:
    if isinstance(exc, GradingError):
        error_code = exc.code
        error_stage = exc.stage
        details = exc.details or {}
        message = str(exc)

        partial = details.get("parsed_result") or {}
        if partial:
            answer["recognized_text"] = partial.get("recognized_text", answer.get("recognized_text", "")) or ""
            answer["recognition_note"] = partial.get("recognition_note", answer.get("recognition_note", "")) or ""
            answer["grading_comment"] = partial.get("grading_comment", answer.get("grading_comment", "")) or ""
            answer["raw_output"] = partial.get("raw_output", answer.get("raw_output", "")) or ""
            answer["ollama_metadata"] = details.get("ollama_metadata", details.get("ocr_metadata", answer.get("ollama_metadata", {}))) or {}
            score = partial.get("score")
            answer["score"] = clamp_score(score) if score is not None else ""
    else:
        error_code = "UNEXPECTED_ERROR"
        error_stage = "app"
        details = {}
        message = f"程序异常：{exc}"

    answer["error"] = message
    answer["error_code"] = error_code
    answer["error_stage"] = error_stage
    answer["status"] = "failed"
    answer["updated_at"] = now_str()
    return error_code, error_stage, details


def render_step_3(sidebar_slots: Optional[Dict[str, object]] = None) -> None:
    st.subheader("上传并批量批改")

    with st.expander("高级设置", expanded=False):
        st.session_state.model_name = st.text_input("Ollama 模型", value=st.session_state.model_name)
        st.session_state.num_ctx = st.number_input(
            "Ollama 上下文长度 num_ctx",
            min_value=2048,
            max_value=32768,
            value=int(st.session_state.num_ctx),
            step=1024,
        )

    uploaded_files = st.file_uploader(
        "上传学生答题卡图片",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("添加图片", type="primary", use_container_width=True):
            added_count, skipped_count = add_uploaded_files(uploaded_files)
            st.success(f"已添加 {added_count} 张图片；跳过重复文件 {skipped_count} 张。")
            st.rerun()
    with col_clear:
        if st.button("清空已上传图片", use_container_width=True):
            st.session_state.answers = {}
            st.rerun()

    if not st.session_state.answers:
        st.warning("当前还没有上传学生答题卡图片。")
        render_log_downloads()
        render_step_nav()
        return

    table_slot = st.empty()
    table_slot.dataframe(build_status_dataframe(), use_container_width=True)

    col_run_pending, col_run_all = st.columns([1, 1])
    run_pending = False
    run_all = False
    with col_run_pending:
        run_pending = st.button("处理未完成图片", type="primary", use_container_width=True)
    with col_run_all:
        run_all = st.button("重新处理全部图片", use_container_width=True)

    if run_all or run_pending:
        exam = st.session_state.exam
        if not exam.get("essay_prompt", "").strip():
            st.warning("请先在第一步录入作文题目。")
            return

        original_text, paragraph1_start, paragraph2_start, warnings = extract_prompt_parts(exam.get("essay_prompt", ""))
        st.session_state.exam["original_text"] = original_text
        st.session_state.exam["paragraph1_start"] = paragraph1_start
        st.session_state.exam["paragraph2_start"] = paragraph2_start
        if warnings:
            for warning in warnings:
                st.warning(warning)

        if not paragraph1_start or not paragraph2_start:
            st.warning("两段续写开头句未完整提取，建议先返回第一步调整作文题目格式。")

        rubric_name = st.session_state.selected_rubric_name
        rubric_text = st.session_state.rubrics.get(rubric_name, DEFAULT_RUBRIC)

        items = []
        for filename, answer in st.session_state.answers.items():
            if run_all or not answer.get("grading_comment") or answer.get("error"):
                items.append((filename, answer))

        if not items:
            st.info("没有待处理图片。")
        else:
            ensure_log_dir()
            for _, answer in items:
                answer["status"] = "queued"
            table_slot.dataframe(build_status_dataframe(), use_container_width=True)
            render_sidebar_progress(sidebar_slots)

            progress = st.progress(0)
            status_box = st.empty()
            detail_box = st.empty()

            success_count = 0
            failed_count = 0
            review_count = 0
            model_name = st.session_state.model_name.strip() or DEFAULT_MODEL
            num_ctx = int(st.session_state.num_ctx)

            stop_batch = False
            for index, (filename, answer) in enumerate(items, start=1):
                answer["status"] = "processing"
                answer["error"] = ""
                answer["error_code"] = ""
                answer["error_stage"] = ""
                answer["updated_at"] = now_str()
                status_box.write(f"正在处理 {index}/{len(items)}：{filename}")
                table_slot.dataframe(build_status_dataframe(), use_container_width=True)
                render_sidebar_progress(sidebar_slots)

                try:
                    result = recognize_and_grade_image(
                        image_bytes=answer["image_bytes"],
                        original_text=original_text,
                        paragraph1_start=paragraph1_start,
                        paragraph2_start=paragraph2_start,
                        scoring_rubric=rubric_text,
                        scoring_standard_name=rubric_name,
                        model=model_name,
                        num_ctx=num_ctx,
                    )
                    update_answer_from_success(answer, result)
                    blocking_warnings = blocking_parse_warnings(answer)
                    non_blocking_warnings = non_blocking_parse_warnings(answer)
                    if blocking_warnings:
                        review_count += 1
                        detail_box.warning(f"已完成但需核对：{filename}。" + "；".join(blocking_warnings))
                    else:
                        success_count += 1
                        if non_blocking_warnings:
                            detail_box.info(f"已完成：{filename}。总分已按五项合计自动修正。")
                        else:
                            detail_box.success(f"已完成：{filename}")
                except Exception as exc:
                    error_code, error_stage, details = update_answer_from_error(answer, exc)
                    failed_count += 1
                    append_error_log(
                        filename=filename,
                        student_id=answer.get("student_id", ""),
                        stage=error_stage,
                        error_code=error_code,
                        message=str(exc),
                        model=model_name,
                        num_ctx=num_ctx,
                        extra={
                            "details": details,
                            "raw_output_preview": safe_preview(answer.get("raw_output", "")),
                            "recognized_text_preview": safe_preview(answer.get("recognized_text", ""), 800),
                            "grading_comment_preview": safe_preview(answer.get("grading_comment", ""), 1200),
                            "answer_ollama_metadata": answer.get("ollama_metadata", {}),
                        },
                        exc=exc,
                    )
                    detail_box.error(f"处理失败：{filename}。原因：{answer.get('error', '')}")
                    if error_code in {"OLLAMA_CONNECTION_ERROR", "OLLAMA_RESOURCE_EXHAUSTED"}:
                        remaining = items[index:]
                        for _, pending_answer in remaining:
                            if pending_answer.get("status") in {"queued", "processing"}:
                                pending_answer["status"] = "pending"
                                pending_answer["error"] = "Ollama 服务不可用或资源耗尽，本轮已暂停，未继续处理该图片。"
                                pending_answer["updated_at"] = now_str()
                        stop_batch = True

                table_slot.dataframe(build_status_dataframe(), use_container_width=True)
                render_sidebar_progress(sidebar_slots)
                progress.progress(index / len(items))
                if stop_batch:
                    status_box.warning("检测到 Ollama 连接中断或资源耗尽，本轮批处理已暂停。请重启 Ollama 或降低 num_ctx 后重试未完成图片。")
                    break

            status_box.write("本批次处理完成。")
            st.session_state.last_batch_summary = (
                f"本批次共 {len(items)} 张：已完成 {success_count} 张，需核对 {review_count} 张，失败 {failed_count} 张。"
            )
            if failed_count:
                st.warning(st.session_state.last_batch_summary)
            else:
                st.success(st.session_state.last_batch_summary)

    if st.session_state.last_batch_summary:
        st.caption(st.session_state.last_batch_summary)

    render_log_downloads()
    render_step_nav()


def reextract_score_callback(filename: str, score_key: str, comment_key: str, message_key: str) -> None:
    """Safely update score widget state from a button callback.

    Streamlit does not allow assigning st.session_state[score_key] after the
    text_input with the same key has been instantiated in the same run. Button
    callbacks run before the script is re-rendered, so this avoids
    StreamlitAPIException.
    """
    answer = st.session_state.answers.get(filename, {})
    score = extract_score(st.session_state.get(comment_key, ""))
    if score is None:
        st.session_state[message_key] = {"type": "warning", "text": "未能从评分意见中提取作文总分。"}
        return
    value = clamp_score(score)
    st.session_state[score_key] = value
    answer["score"] = value
    answer["updated_at"] = now_str()
    if answer.get("grading_comment") or st.session_state.get(comment_key, "").strip():
        answer["status"] = "done"
        answer["error"] = ""
    st.session_state[message_key] = {"type": "success", "text": "已重新提取得分。"}


def render_result_card(filename: str, answer: Dict) -> None:
    safe_key = make_safe_key(filename)
    with st.expander(f"{filename} | 学生编号：{answer.get('student_id', '')} | {answer_status(answer)}", expanded=False):
        col_img, col_text = st.columns([1, 1])
        with col_img:
            try:
                image = Image.open(BytesIO(answer["image_bytes"]))
                st.image(image, caption=filename, width="stretch")
            except Exception:
                st.error("图片无法显示。")

            student_id = st.text_input(
                "学生编号",
                value=answer.get("student_id", ""),
                key=f"student_id_{safe_key}",
            )
            if st.button("保存学生编号", key=f"save_student_id_{safe_key}"):
                answer["student_id"] = student_id.strip()
                answer["updated_at"] = now_str()
                st.success("已保存学生编号。")

        with col_text:
            if answer.get("error"):
                st.error(answer["error"])
                if answer.get("error_code") or answer.get("error_stage"):
                    st.caption(f"错误代码：{answer.get('error_code', '')}；阶段：{answer.get('error_stage', '')}")
            blocking_warnings = blocking_parse_warnings(answer)
            non_blocking_warnings = non_blocking_parse_warnings(answer)
            for warning in blocking_warnings:
                st.warning(warning)
            for warning in non_blocking_warnings:
                st.info(f"系统已自动处理：{warning}")

            st.text_area(
                "学生手写文本识别结果",
                value=answer.get("recognized_text", ""),
                height=180,
                disabled=True,
                key=f"recognized_text_{safe_key}",
            )
            st.text_area(
                "识别备注",
                value=answer.get("recognition_note", ""),
                height=80,
                disabled=True,
                key=f"recognition_note_{safe_key}",
            )

        st.divider()
        score_key = f"score_{safe_key}"
        comment_key = f"grading_comment_{safe_key}"
        if score_key not in st.session_state:
            st.session_state[score_key] = str(answer.get("score", ""))
        if comment_key not in st.session_state:
            st.session_state[comment_key] = answer.get("grading_comment", "")

        st.text_input("作文得分", key=score_key, placeholder="例如：18.5")
        st.text_area("教师评分意见", key=comment_key, height=360)

        col_save, col_reextract = st.columns([1, 1])
        with col_save:
            if st.button("保存评分意见和得分", type="primary", key=f"save_result_{safe_key}"):
                answer["grading_comment"] = st.session_state[comment_key].strip()
                answer["score"] = clamp_score(st.session_state[score_key])
                answer["error"] = "" if answer["grading_comment"] and answer["score"] != "" else answer.get("error", "")
                answer["status"] = "done" if answer["grading_comment"] and answer["score"] != "" and not needs_manual_review(answer) else "review_required"
                answer["updated_at"] = now_str()
                st.success("已保存。")
        with col_reextract:
            message_key = f"reextract_message_{safe_key}"
            st.button(
                "从评分意见重新提取得分",
                key=f"reextract_score_{safe_key}",
                on_click=reextract_score_callback,
                args=(filename, score_key, comment_key, message_key),
            )
            msg = st.session_state.get(message_key)
            if isinstance(msg, dict) and msg.get("text"):
                if msg.get("type") == "success":
                    st.success(msg["text"])
                else:
                    st.warning(msg["text"])

        if answer.get("raw_output"):
            with st.expander("查看模型原始输出", expanded=False):
                st.text_area("模型原始输出", value=answer.get("raw_output", ""), height=260, disabled=True)
        if answer.get("ollama_metadata"):
            with st.expander("查看 Ollama 调用元数据", expanded=False):
                st.json(answer.get("ollama_metadata", {}))


def guess_image_mime(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip('.')
    if suffix in {'jpg', 'jpeg'}:
        return 'image/jpeg'
    if suffix == 'webp':
        return 'image/webp'
    return 'image/png'


def build_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='results')
    return buffer.getvalue()


def render_manual_blob_download_button(label: str, data: bytes, file_name: str, mime: str) -> None:
    '''Render a browser-side download button without exposing a zip URL on page load.

    st.download_button registers a downloadable media endpoint as soon as the
    widget is rendered. Download managers such as IDM may scan and prefetch that
    endpoint when the user enters the page, which can look like an automatic
    download. This helper embeds the bytes as base64 and creates the Blob URL
    only inside the user's click handler.
    '''
    if not data:
        st.button(label, disabled=True)
        return

    button_id = 'manual-download-' + make_safe_key(file_name, len(data), label)[:12]
    data_b64 = base64.b64encode(data).decode('ascii')
    js_file_name = json.dumps(file_name, ensure_ascii=False)
    js_mime = json.dumps(mime)
    js_label = html_text(label)

    components.html(
        f'''
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,'Microsoft YaHei',sans-serif;">
          <button id="{button_id}" type="button"
            style="width:100%;border:1px solid #2563eb;background:#2563eb;color:#fff;border-radius:8px;padding:10px 16px;font-size:14px;cursor:pointer;">
            {js_label}
          </button>
        </div>
        <script>
        (function() {{
          const btn = document.getElementById({json.dumps(button_id)});
          const fileName = {js_file_name};
          const mime = {js_mime};
          const base64Data = {json.dumps(data_b64)};
          function base64ToBlob(b64, type) {{
            const chars = atob(b64);
            const chunkSize = 8192;
            const chunks = [];
            for (let offset = 0; offset < chars.length; offset += chunkSize) {{
              const slice = chars.slice(offset, offset + chunkSize);
              const bytes = new Uint8Array(slice.length);
              for (let i = 0; i < slice.length; i++) bytes[i] = slice.charCodeAt(i);
              chunks.push(bytes);
            }}
            return new Blob(chunks, {{type}});
          }}
          btn.addEventListener('click', function() {{
            const blob = base64ToBlob(base64Data, mime);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            setTimeout(function() {{
              URL.revokeObjectURL(url);
              a.remove();
            }}, 1500);
          }});
        }})();
        </script>
        ''',
        height=52,
    )


def html_text(value) -> str:
    return html.escape('' if value is None else str(value))


def html_pre(value) -> str:
    return html.escape('' if value is None else str(value))


def build_html_table(rows: List[Dict], columns: List[str]) -> str:
    if not rows:
        return '<p class="muted">暂无数据。</p>'
    head = ''.join(f'<th>{html_text(col)}</th>' for col in columns)
    body_parts = []
    for row in rows:
        cells = ''.join(f'<td>{html_text(row.get(col, ""))}</td>' for col in columns)
        body_parts.append(f'<tr>{cells}</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_parts)}</tbody></table></div>'


def build_frontend_snapshot_zip() -> bytes:
    """Build an offline static snapshot of the four-step frontend state.

    The snapshot is designed for GitHub demos: open index.html locally, use
    previous/next buttons to switch the four pages, and download CSV/Excel
    result files from the last page without a Python backend.
    """
    df = build_export_dataframe()
    status_df = build_status_dataframe()
    csv_bytes = df.to_csv(index=False).encode('utf-8-sig')
    try:
        excel_bytes = build_excel_bytes(df)
        excel_available = True
    except Exception:
        excel_bytes = b''
        excel_available = False

    counts = status_counts()
    exam = st.session_state.exam
    rubric_name = st.session_state.selected_rubric_name
    rubric_text = st.session_state.rubrics.get(rubric_name, '')

    answers_snapshot = []
    image_files = []
    for filename, answer in st.session_state.answers.items():
        image_bytes = answer.get('image_bytes', b'') or b''
        image_mime = guess_image_mime(filename)
        image_b64 = base64.b64encode(image_bytes).decode('ascii') if image_bytes else ''
        safe_image_name = re.sub(r'[^0-9A-Za-z._-]+', '_', Path(filename).name)
        image_path = f'assets/images/{safe_image_name}'
        if image_bytes:
            image_files.append((image_path, image_bytes))
        answers_snapshot.append({
            'file_name': filename,
            'student_id': answer.get('student_id', ''),
            'status': answer_status(answer),
            'score': answer.get('score', ''),
            'updated_at': answer.get('updated_at', ''),
            'error': answer.get('error', ''),
            'recognized_text': answer.get('recognized_text', ''),
            'recognition_note': answer.get('recognition_note', ''),
            'grading_comment': answer.get('grading_comment', ''),
            'raw_output': answer.get('raw_output', ''),
            'parse_warnings': answer.get('parse_warnings', []) or [],
            'ollama_metadata': answer.get('ollama_metadata', {}) or {},
            'image_path': image_path if image_bytes else '',
            'image_data_uri': f'data:{image_mime};base64,{image_b64}' if image_b64 else '',
        })

    status_rows = status_df.to_dict(orient='records')
    export_rows = df.to_dict(orient='records')
    csv_b64 = base64.b64encode(csv_bytes).decode('ascii')
    excel_b64 = base64.b64encode(excel_bytes).decode('ascii') if excel_bytes else ''

    snapshot_data = {
        'app_name': APP_NAME,
        'version': VERSION,
        'created_at': now_str(),
        'steps': STEPS,
        'exam': exam,
        'rubric_name': rubric_name,
        'rubric_text': rubric_text,
        'counts': counts,
        'status_rows': status_rows,
        'export_rows': export_rows,
        'answers': answers_snapshot,
        'excel_available': excel_available,
    }

    step1_html = f'''
      <section class="card">
        <h2>1. 考试和题目信息录入</h2>
        <div class="kv"><b>考试名称</b><span>{html_text(exam.get('exam_name', ''))}</span></div>
        <h3>作文题目</h3>
        <pre>{html_pre(exam.get('essay_prompt', ''))}</pre>
        <h3>后台提取结果</h3>
        <div class="grid-2">
          <div><h4>原文阅读材料</h4><pre>{html_pre(exam.get('original_text', ''))}</pre></div>
          <div><h4>第一段续写开头句</h4><pre>{html_pre(exam.get('paragraph1_start', ''))}</pre><h4>第二段续写开头句</h4><pre>{html_pre(exam.get('paragraph2_start', ''))}</pre></div>
        </div>
      </section>
    '''

    step2_html = f'''
      <section class="card">
        <h2>2. 评分标准</h2>
        <div class="kv"><b>当前使用的评分标准</b><span>{html_text(rubric_name)}</span></div>
        <pre>{html_pre(rubric_text)}</pre>
      </section>
    '''

    step3_html = f'''
      <section class="card">
        <h2>3. 上传并批量批改</h2>
        <div class="metrics">
          <div><b>{counts['total']}</b><span>已上传</span></div>
          <div><b>{counts['finished']}</b><span>已完成</span></div>
          <div><b>{counts['review']}</b><span>需核对</span></div>
          <div><b>{counts['failed']}</b><span>失败</span></div>
        </div>
        <div class="progress"><span style="width:{int((counts['done_like'] / counts['total'] * 100) if counts['total'] else 0)}%"></span></div>
        <h3>处理状态表</h3>
        {build_html_table(status_rows, ['文件名', '学生编号', '状态', '作文得分', '更新时间', '失败原因'])}
        <p class="muted">这是批改完成后的静态快照，不能重新调用模型，但保留了运行结果、日志和导出文件。</p>
      </section>
    '''

    cards = []
    for ans in answers_snapshot:
        warnings = ''.join(f'<li>{html_text(w)}</li>' for w in ans.get('parse_warnings', []))
        metadata = html_pre(json.dumps(ans.get('ollama_metadata', {}), ensure_ascii=False, indent=2))
        img_html = f'<img src="{ans["image_data_uri"]}" alt="{html_text(ans["file_name"])}" />' if ans.get('image_data_uri') else '<p class="muted">无图片数据。</p>'
        cards.append(f'''
          <details class="result-card">
            <summary>{html_text(ans['file_name'])}｜学生编号：{html_text(ans['student_id'])}｜{html_text(ans['status'])}｜得分：{html_text(ans['score'])}</summary>
            <div class="result-body">
              <div class="grid-2">
                <div>{img_html}</div>
                <div>
                  <h4>学生手写文本识别结果</h4><pre>{html_pre(ans.get('recognized_text', ''))}</pre>
                  <h4>识别备注</h4><pre>{html_pre(ans.get('recognition_note', ''))}</pre>
                </div>
              </div>
              <h4>教师评分意见</h4><pre>{html_pre(ans.get('grading_comment', ''))}</pre>
              {f'<div class="error">失败原因：{html_text(ans.get("error", ""))}</div>' if ans.get('error') else ''}
              {f'<div class="notice"><b>解析提示</b><ul>{warnings}</ul></div>' if warnings else ''}
              <details><summary>查看模型原始输出</summary><pre>{html_pre(ans.get('raw_output', ''))}</pre></details>
              <details><summary>查看 Ollama 调用元数据</summary><pre>{metadata}</pre></details>
            </div>
          </details>
        ''')

    excel_button = (
        f'<a class="btn" download="高中英语作文批改结果_{VERSION}.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}">下载 Excel 结果表</a>'
        if excel_available
        else '<button class="btn disabled" disabled>Excel 结果表不可用</button>'
    )
    step4_html = f'''
      <section class="card">
        <h2>4. 结果核对与导出</h2>
        <h3>结果表预览</h3>
        {build_html_table(export_rows, ['序号', '考试名称', '学生编号', '评分标准名称', '评分意见', '作文得分', '更新时间'])}
        <div class="download-row">
          <a class="btn" download="高中英语作文批改结果_{VERSION}.csv" href="data:text/csv;charset=utf-8;base64,{csv_b64}">下载 CSV 结果表</a>
          {excel_button}
        </div>
        <h3>逐个核对</h3>
        {''.join(cards) if cards else '<p class="muted">暂无学生结果。</p>'}
      </section>
    '''

    data_json = json.dumps(snapshot_data, ensure_ascii=False, indent=2)
    safe_data_json = data_json.replace('</', '<\\/')
    html_doc = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_text(APP_NAME)} 运行快照 {html_text(VERSION)}</title>
  <style>
    :root {{ --primary:#2563eb; --bg:#f8fafc; --border:#e5e7eb; --text:#111827; --muted:#64748b; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Microsoft YaHei",sans-serif; color:var(--text); background:#fff; }}
    .layout {{ display:grid; grid-template-columns:260px 1fr; min-height:100vh; }}
    aside {{ background:var(--bg); border-right:1px solid var(--border); padding:28px 22px; position:sticky; top:0; height:100vh; }}
    main {{ padding:36px 52px; overflow:auto; }}
    h1 {{ margin:0 0 8px; font-size:34px; }} h2 {{ margin-top:0; }} h3 {{ margin-top:26px; }}
    .caption,.muted {{ color:var(--muted); }} .step-list div {{ padding:8px 0; }} .active-step {{ font-weight:700; color:var(--primary); }}
    .topbar {{ margin:18px 0 28px; }} .progress {{ height:8px; background:#eef2ff; border-radius:999px; overflow:hidden; }} .progress span {{ display:block; height:100%; background:var(--primary); }}
    .nav {{ display:flex; justify-content:space-between; align-items:center; margin:24px 0; gap:12px; }}
    button,.btn {{ border:1px solid var(--border); background:#fff; color:#111827; border-radius:8px; padding:10px 16px; cursor:pointer; text-decoration:none; display:inline-block; font-size:14px; }}
    button.primary,.btn {{ background:var(--primary); border-color:var(--primary); color:#fff; }} button:disabled,.disabled {{ opacity:.45; cursor:not-allowed; }}
    .card {{ border:1px solid var(--border); border-radius:12px; padding:24px; background:#fff; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .page {{ display:none; }} .page.active {{ display:block; }}
    .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }} .metrics div {{ background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:14px; }} .metrics b {{ display:block; font-size:24px; }} .metrics span {{ color:var(--muted); }}
    .kv {{ display:grid; grid-template-columns:180px 1fr; border:1px solid var(--border); border-radius:10px; overflow:hidden; margin:12px 0; }} .kv b,.kv span {{ padding:12px; }} .kv b {{ background:var(--bg); }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:12px; line-height:1.6; max-height:420px; overflow:auto; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:10px; }} table {{ width:100%; border-collapse:collapse; font-size:14px; }} th,td {{ border-bottom:1px solid var(--border); padding:10px; text-align:left; vertical-align:top; }} th {{ background:#f8fafc; position:sticky; top:0; }}
    .result-card {{ border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin:10px 0; }} .result-card summary {{ cursor:pointer; font-weight:600; }} .result-body {{ margin-top:14px; }} img {{ max-width:100%; border:1px solid var(--border); border-radius:8px; }}
    .download-row {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0; }} .error {{ background:#fef2f2; color:#991b1b; padding:12px; border-radius:8px; margin:12px 0; }} .notice {{ background:#eff6ff; color:#1e3a8a; padding:12px; border-radius:8px; margin:12px 0; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ position:static; height:auto; }} main {{ padding:24px; }} .grid-2,.metrics {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h2>当前步骤</h2>
      <div class="step-list" id="sideSteps"></div>
      <hr />
      <p class="caption">处理进度</p>
      <p>已上传：{counts['total']}</p><p>已完成：{counts['finished']}</p><p>需核对：{counts['review']}</p><p>处理中：{counts['processing']}</p><p>待处理：{counts['pending']}</p><p>失败：{counts['failed']}</p>
      <div class="progress"><span style="width:{int((counts['done_like'] / counts['total'] * 100) if counts['total'] else 0)}%"></span></div>
    </aside>
    <main>
      <h1>{html_text(APP_NAME)}</h1>
      <div class="caption">静态运行快照｜版本 {html_text(VERSION)}｜生成时间 {html_text(now_str())}</div>
      <div class="topbar"><div class="progress"><span id="stepProgress"></span></div></div>
      <div id="page0" class="page">{step1_html}</div>
      <div id="page1" class="page">{step2_html}</div>
      <div id="page2" class="page">{step3_html}</div>
      <div id="page3" class="page">{step4_html}</div>
      <div class="nav"><button id="prevBtn" onclick="prevPage()">上一步</button><span id="stepCaption" class="caption"></span><button id="nextBtn" class="primary" onclick="nextPage()">下一步</button></div>
    </main>
  </div>
  <script id="snapshot-data" type="application/json">{safe_data_json}</script>
  <script>
    const SNAPSHOT = JSON.parse(document.getElementById('snapshot-data').textContent);
    let current = 0;
    function render() {{
      for (let i=0; i<SNAPSHOT.steps.length; i++) {{
        document.getElementById('page'+i).classList.toggle('active', i===current);
      }}
      const side = document.getElementById('sideSteps');
      side.innerHTML = SNAPSHOT.steps.map((s,i)=>`<div class="${{i===current?'active-step':''}}">${{i===current?'▶':'○'}} ${{i+1}}. ${{s}}</div>`).join('');
      document.getElementById('stepProgress').style.width = ((current+1)/SNAPSHOT.steps.length*100)+'%';
      document.getElementById('stepCaption').textContent = `步骤 ${{current+1}}/${{SNAPSHOT.steps.length}}：${{SNAPSHOT.steps[current]}}`;
      document.getElementById('prevBtn').disabled = current===0;
      document.getElementById('nextBtn').disabled = current===SNAPSHOT.steps.length-1;
      window.scrollTo(0,0);
    }}
    function prevPage() {{ if (current>0) {{ current--; render(); }} }}
    function nextPage() {{ if (current<SNAPSHOT.steps.length-1) {{ current++; render(); }} }}
    render();
  </script>
</body>
</html>
'''

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('index.html', html_doc.encode('utf-8'))
        zf.writestr('data.json', data_json.encode('utf-8'))
        zf.writestr(f'export/高中英语作文批改结果_{VERSION}.csv', csv_bytes)
        if excel_bytes:
            zf.writestr(f'export/高中英语作文批改结果_{VERSION}.xlsx', excel_bytes)
        for image_path, image_bytes in image_files:
            zf.writestr(image_path, image_bytes)
        jsonl_data = read_log_file(ERROR_LOG_JSONL)
        text_data = read_log_file(ERROR_LOG_TEXT)
        if jsonl_data:
            zf.writestr('logs/processing_errors.jsonl', jsonl_data)
        if text_data:
            zf.writestr('logs/processing_errors.log', text_data)
        zf.writestr('README_snapshot.txt', (
            f'{APP_NAME} 前端运行快照\n'
            f'版本：{VERSION}\n'
            '使用方法：解压本 zip 后，用浏览器打开 index.html。\n'
            'index.html 是纯静态页面，可通过“上一步 / 下一步”切换四个页面；第 4 页保留 CSV 和 Excel 下载按钮。\n'
        ).encode('utf-8'))
    return zip_buffer.getvalue()

def render_step_4() -> None:
    st.subheader("结果核对与导出")

    if not st.session_state.answers:
        st.warning("当前还没有学生结果。请先上传并处理图片。")
        render_log_downloads()
        render_step_nav()
        return

    st.dataframe(build_status_dataframe(), use_container_width=True)

    st.markdown("### 逐个核对")
    for filename, answer in st.session_state.answers.items():
        render_result_card(filename, answer)

    st.markdown("### 导出结果")
    df = build_export_dataframe()
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载 CSV 结果表",
        data=csv_data,
        file_name=f"高中英语作文批改结果_{VERSION}.csv",
        mime="text/csv",
    )

    excel_buffer = BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
        st.download_button(
            label="下载 Excel 结果表",
            data=excel_buffer.getvalue(),
            file_name=f"高中英语作文批改结果_{VERSION}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ModuleNotFoundError:
        st.warning("当前环境未安装 openpyxl，暂时无法导出 Excel。")

    st.markdown("### 前端运行快照")
    st.caption("快照包可离线打开。解压后打开 index.html，可以通过“上一步 / 下一步”查看四个页面，并在第 4 页继续下载 CSV 和 Excel 结果表。")
    st.info("为避免下载管理器在进入本页时自动抓取 zip，本区域先手动生成快照，再点击页面内下载按钮保存文件。")

    if st.button("生成 / 刷新前端运行快照包", type="primary", use_container_width=True):
        try:
            st.session_state.snapshot_zip_data = build_frontend_snapshot_zip()
            st.session_state.snapshot_zip_generated_at = now_str()
            st.success(f"已生成前端运行快照包：{st.session_state.snapshot_zip_generated_at}")
        except Exception as exc:
            st.session_state.snapshot_zip_data = None
            st.session_state.snapshot_zip_generated_at = ""
            st.warning(f"生成前端运行快照失败：{exc}")

    snapshot_data = st.session_state.get("snapshot_zip_data")
    if snapshot_data:
        generated_at = st.session_state.get("snapshot_zip_generated_at", "")
        st.caption(f"已准备快照包：{generated_at}。点击下方按钮后才会触发浏览器下载。")
        render_manual_blob_download_button(
            label="下载前端运行快照包",
            data=snapshot_data,
            file_name=f"frontend_snapshot_{VERSION}.zip",
            mime="application/zip",
        )
    else:
        st.caption("尚未生成快照包。")

    render_log_downloads()
    render_step_nav()


init_session_state()
inject_custom_css()
sidebar_slots = render_sidebar()

st.title(APP_NAME)

current_step = st.session_state.current_step
st.caption(f"步骤 {current_step + 1}/{len(STEPS)}：{STEPS[current_step]}")
st.progress((current_step + 1) / len(STEPS))
st.divider()

if current_step == 0:
    render_step_1()
elif current_step == 1:
    render_step_2()
elif current_step == 2:
    render_step_3(sidebar_slots)
elif current_step == 3:
    render_step_4()
