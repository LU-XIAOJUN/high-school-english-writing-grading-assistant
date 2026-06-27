import base64
import json
import re
import time
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image, UnidentifiedImageError


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT_OCR = 2048
DEFAULT_NUM_PREDICT_GRADING = 1536
DEFAULT_NUM_PREDICT_GRADING_FALLBACK = 1024
FALLBACK_NUM_CTX = 4096


DEFAULT_RUBRIC = """
读后续写总分为25分，分为五个维度，每个维度5分。

一、内容创作与逻辑（5分）
5分：情节丰富合理，逻辑严密，续写完整，与原文情境高度融合，细节饱满且符合人物设定与故事走向，能自然推动情节发展，形成闭环。
4分：情节较丰富合理，逻辑清晰，续写较完整，与原文情境契合度较高，无明显逻辑漏洞，能体现故事的合理推进。
3分：情节基本合理，有一定逻辑性，续写基本完整，与原文情境相关，无严重逻辑矛盾，故事推进方向基本合理。
2分：情节存在一定不合理之处，逻辑不够连贯，续写不够完整，与原文情境有一定脱节，故事推进存在小漏洞。
1分：情节存在较多重大问题，或抄袭原文内容，续写不完整，与原文情境基本脱节，逻辑混乱，无法形成完整故事线。

二、语言表达准确性（5分）
5分：词汇多样精准，语法结构丰富且运用恰当，语言错误极少，表达流畅自然，完全不影响理解，能灵活使用高级词汇与复杂句式。
4分：词汇较丰富恰当，语法结构多样，表达较流畅，仅有个别错误，不影响整体理解，能合理使用不同句式。
3分：词汇和语法结构以简单形式为主，存在少量错误或不恰当之处，但基本不影响理解，能满足基础表达需求。
2分：词汇有限，语法结构单调，错误较多，对理解造成一定影响，句式单一，存在较多基础错误。
1分：词汇非常有限，语法结构混乱，错误很多，严重影响理解，句式重复，大量基础语法和拼写错误。

三、篇章结构与衔接（5分）
5分：自然有效地使用了多样的语句间衔接手段，段落层次清晰，前后呼应，全文逻辑连贯，过渡自然，衔接词运用精准且丰富。
4分：较有效地使用了语句间衔接手段，全文结构较清晰，意义较连贯，有一定的过渡衔接，逻辑链条较完整。
3分：基本有效地使用了语句间衔接手段，全文结构基本清晰，意义基本连贯，有基础的衔接词，无明显逻辑断裂。
2分：未能有效地使用语句间衔接手段，全文结构不够清晰，意义不够连贯，过渡生硬，衔接词使用不当或缺失。
1分：几乎没有使用语句间衔接手段，全文结构不清晰，意义不连贯，上下文逻辑断裂，无法形成完整的篇章结构。

四、情境契合度（5分）
5分：续写内容完全贴合原文的人物性格、场景设定与情感基调，人物行为符合原文设定，场景氛围与原文高度统一，浑然一体。
4分：续写内容较贴合原文情境，人物行为、场景设定与原文契合度较高，无明显违和感，能呼应原文的情感基调。
3分：续写内容基本贴合原文情境，人物行为、场景设定无明显矛盾，与原文的情感基调基本一致。
2分：续写内容与原文情境有一定脱节，人物行为或场景设定与原文存在一定违和感，情感基调呼应不足。
1分：续写内容与原文情境严重脱节，人物行为违背原文设定，场景或情感基调与原文完全不符，无法呼应原文。

五、写作规范性（5分）
5分：词数符合题目要求，格式规范，书写清晰工整，无拼写、标点等低级错误，内容完整，符合读后续写的格式与字数要求。
4分：词数基本符合要求，格式规范，书写较清晰，仅有个别低级错误，不影响整体阅读，格式无明显问题。
3分：词数与要求略有偏差，格式无明显错误，存在少量低级错误，基本不影响阅读，内容基本完整。
2分：词数偏差较大，格式存在一定问题，书写潦草，低级错误较多，对阅读造成一定影响，内容不够完整。
1分：词数严重不足或远超要求，格式混乱，书写无法辨认，低级错误泛滥，内容未完成或与题目要求完全不符。

评分注意事项：
1. 内容创作与逻辑主要评价故事是否完整、合理、连贯、形成闭环。
2. 情境契合度主要评价续写是否贴合原文人物性格、场景设定和情感基调。
3. 不要因为同一个问题在多个维度中重复严重扣分。
4. 图片识别错误不应视为学生语言错误。
5. 写作规范性中的书写清晰度只能作为辅助判断，不能仅凭图片识别效果直接扣分。
""".strip()


class GradingError(RuntimeError):
    """LLM recognition/grading failed with structured diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "GRADING_ERROR",
        stage: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


def remove_think_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return cleaned.strip()


def preview_text(text: Optional[str], limit: int = 2000) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...<已截断，原长度 {len(text)} 字符>"


def compact_text(text: str, limit: int = 2200) -> str:
    """Limit long prompt sections to reduce Ollama runner pressure on small local models."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n……（此处已截断，原长度 {len(text)} 字符）"


def compact_rubric_for_prompt(scoring_rubric: str) -> str:
    """Keep grading criteria useful but short enough for stable local inference."""
    rubric = (scoring_rubric or "").strip()
    default_norm = re.sub(r"\s+", "", DEFAULT_RUBRIC)
    rubric_norm = re.sub(r"\s+", "", rubric)
    if not rubric or rubric_norm == default_norm:
        return (
            "读后续写总分25分，分为五个维度，每项5分："
            "1. 内容创作与逻辑：故事完整、合理、连贯，能形成闭环；"
            "2. 语言表达准确性：词汇、语法、句式准确自然，错误不影响理解；"
            "3. 篇章结构与衔接：两段结构清晰，过渡自然，前后呼应；"
            "4. 情境契合度：人物性格、情节走向、情感基调与原文一致；"
            "5. 写作规范性：词数、格式、书写、拼写和标点基本符合要求。"
            "评分时不要因同一问题在多个维度重复严重扣分；图片识别不确定处不直接作为学生语言错误。"
        )
    return compact_text(rubric, 1600)


def is_retryable_ollama_error(exc: BaseException) -> bool:
    if not isinstance(exc, GradingError):
        return False
    if exc.code in {"OLLAMA_RESOURCE_EXHAUSTED", "OLLAMA_TIMEOUT", "OLLAMA_MODEL_ERROR"}:
        return True
    if exc.code == "OLLAMA_HTTP_ERROR":
        text = json.dumps(exc.details, ensure_ascii=False).lower()
        return any(token in text for token in ["insufficient buffer", "queue was full", "tokenize", "runner", "server error"])
    return False


def is_fatal_ollama_availability_error(exc: BaseException) -> bool:
    return isinstance(exc, GradingError) and exc.code in {"OLLAMA_CONNECTION_ERROR", "OLLAMA_RESOURCE_EXHAUSTED"}


def image_bytes_to_base64(image_bytes: bytes, max_side: int = 1800, jpeg_quality: int = 92) -> str:
    """
    Convert uploaded image bytes to a compact JPEG base64 string for Ollama vision models.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise GradingError(
            "图片文件无法识别。请确认上传的是 jpg、png 或 webp 图片。",
            code="IMAGE_UNIDENTIFIED",
            stage="image_preprocess",
            details={"image_bytes": len(image_bytes or b"")},
        ) from exc
    except Exception as exc:
        raise GradingError(
            f"图片预处理失败：{exc}",
            code="IMAGE_PREPROCESS_ERROR",
            stage="image_preprocess",
            details={"image_bytes": len(image_bytes or b"")},
        ) from exc

    width, height = image.size
    largest = max(width, height)

    if largest > max_side:
        scale = max_side / largest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(new_size)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _ollama_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep stable diagnostics from the Ollama response without storing huge fields."""
    return {
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_count": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
        "response_keys": list(data.keys()),
        "has_thinking": bool(data.get("thinking")),
        "thinking_preview": preview_text(data.get("thinking", ""), 1200),
        "response_preview": preview_text(data.get("response", ""), 1200),
    }


def call_ollama_generate(
    *,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT_GRADING,
    timeout: int = 600,
    stage: str = "ollama_request",
    json_mode: bool = True,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Call Ollama /api/generate and return response text plus diagnostics.

    Important fixes for Qwen3-VL:
    - think=False must be a top-level request field, not an options field.
    - format="json" nudges Ollama to return valid JSON.
    - num_predict makes output length explicit.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    if json_mode:
        payload["format"] = "json"
    if image_bytes is not None:
        payload["images"] = [image_bytes_to_base64(image_bytes)]

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise GradingError(
            "无法连接本地 Ollama 服务。请确认 Ollama 已安装并正在运行。",
            code="OLLAMA_CONNECTION_ERROR",
            stage=stage,
            details={"url": OLLAMA_URL, "model": model, "num_predict": num_predict},
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise GradingError(
            "模型响应超时。可尝试减少单批图片数量、降低图片分辨率，或关闭其他占用内存的软件后重试。",
            code="OLLAMA_TIMEOUT",
            stage=stage,
            details={"url": OLLAMA_URL, "model": model, "timeout_seconds": timeout, "num_predict": num_predict},
        ) from exc
    except Exception as exc:
        raise GradingError(
            f"调用 Ollama 请求失败：{exc}",
            code="OLLAMA_REQUEST_ERROR",
            stage=stage,
            details={"url": OLLAMA_URL, "model": model, "num_predict": num_predict},
        ) from exc

    response_text = response.text or ""
    if response.status_code >= 400:
        lower_response = response_text.lower()
        resource_tokens = [
            "insufficient buffer",
            "queue was full",
            "tokenize",
            "runner",
            "out of memory",
            "no space left",
        ]
        if response.status_code == 500 and any(token in lower_response for token in resource_tokens):
            message = (
                "Ollama 内部资源耗尽或队列已满。通常是本地内存/显存不足、Ollama runner 异常或提示词过长导致。"
                "请重启 Ollama，并尝试降低 num_ctx 或减少单批图片数量。"
            )
            code = "OLLAMA_RESOURCE_EXHAUSTED"
        else:
            message = f"Ollama 返回 HTTP {response.status_code}。"
            code = "OLLAMA_HTTP_ERROR"
        raise GradingError(
            message,
            code=code,
            stage=stage,
            details={
                "status_code": response.status_code,
                "response_preview": preview_text(response_text),
                "model": model,
                "num_predict": num_predict,
                "num_ctx": int(num_ctx),
            },
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise GradingError(
            "Ollama 返回内容不是合法 JSON。",
            code="OLLAMA_RESPONSE_JSON_ERROR",
            stage=stage,
            details={"response_preview": preview_text(response_text), "model": model, "num_predict": num_predict},
        ) from exc

    metadata = _ollama_metadata(data)
    if data.get("error"):
        raise GradingError(
            f"Ollama 模型返回错误：{data.get('error')}",
            code="OLLAMA_MODEL_ERROR",
            stage=stage,
            details={"ollama_error": data.get("error"), "model": model, "metadata": metadata},
        )

    response_result = remove_think_tags(data.get("response", ""))
    thinking_result = remove_think_tags(data.get("thinking", ""))

    # Some local Ollama + qwen3-vl environments ignore/partially ignore top-level
    # think=False and place the final JSON in the `thinking` field while leaving
    # `response` empty. Treat a JSON-looking thinking field as a compatible result
    # instead of failing the whole image.
    used_output_field = ""
    fallback_reason = ""
    result = ""

    if json_mode:
        response_has_json = bool(extract_json_object(response_result))
        thinking_has_json = bool(extract_json_object(thinking_result))
        if response_result.strip() and response_has_json:
            result = response_result
            used_output_field = "response"
        elif thinking_result.strip() and thinking_has_json:
            result = thinking_result
            used_output_field = "thinking"
            fallback_reason = "response 为空或不可解析，但 thinking 字段包含可解析 JSON，已作为兼容结果使用。"
        elif response_result.strip():
            # Keep response for downstream parsers/logging even if JSON mode was imperfect.
            result = response_result
            used_output_field = "response_non_json"
            fallback_reason = "response 非空但不是标准 JSON，已交给兼容解析器处理。"
        elif thinking_result.strip() and not response_result.strip():
            done_reason = data.get("done_reason")
            if done_reason == "length":
                message = "模型输出耗尽在 thinking 字段，最终 response 为空，且 thinking 中未找到可解析 JSON。"
                code = "OLLAMA_THINKING_LENGTH_EMPTY"
            else:
                message = "模型只返回了 thinking 字段，但其中未找到可解析 JSON。请查看错误日志中的 thinking_preview。"
                code = "OLLAMA_THINKING_NON_JSON"
            metadata.update(
                {
                    "used_output_field": "none",
                    "fallback_reason": "thinking 字段存在但未包含可解析 JSON。",
                }
            )
            raise GradingError(
                message,
                code=code,
                stage=stage,
                details={
                    "model": model,
                    "metadata": metadata,
                    "response_preview": preview_text(response_text),
                    "num_predict": num_predict,
                    "json_mode": json_mode,
                    "think_sent_as_top_level_false": True,
                },
            )
    else:
        if response_result.strip():
            result = response_result
            used_output_field = "response"
        elif thinking_result.strip():
            result = thinking_result
            used_output_field = "thinking"
            fallback_reason = "response 为空，但 thinking 字段包含文本内容，已作为兼容结果使用。"

    if not result.strip():
        message = "模型返回为空。可能是模型未加载成功、显存/内存不足、输出被中断，或图片请求未被模型处理。"
        code = "OLLAMA_EMPTY_RESPONSE"
        metadata.update({"used_output_field": "none", "fallback_reason": "response 与 thinking 均为空。"})
        raise GradingError(
            message,
            code=code,
            stage=stage,
            details={
                "model": model,
                "metadata": metadata,
                "response_preview": preview_text(response_text),
                "num_predict": num_predict,
                "json_mode": json_mode,
                "think_sent_as_top_level_false": True,
            },
        )

    metadata.update(
        {
            "used_output_field": used_output_field,
            "fallback_reason": fallback_reason,
        }
    )

    return {
        "response": result,
        "used_output_field": used_output_field,
        "fallback_reason": fallback_reason,
        "raw_response_text": response_text,
        "data": data,
        "metadata": metadata,
        "request": {
            "model": model,
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "json_mode": json_mode,
            "think": False,
            "temperature": float(temperature),
            "has_image": image_bytes is not None,
        },
    }


# Retained for compatibility with older imports/tests.
def call_ollama_vision(
    prompt: str,
    image_bytes: bytes,
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: int = 600,
) -> str:
    return call_ollama_generate(
        prompt=prompt,
        image_bytes=image_bytes,
        model=model,
        num_ctx=num_ctx,
        num_predict=DEFAULT_NUM_PREDICT_GRADING,
        timeout=timeout,
        stage="vision_grading",
        json_mode=True,
    )["response"]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def extract_section(text: str, start_label: str, end_labels: Optional[list[str]] = None) -> str:
    if not text:
        return ""
    end_labels = end_labels or []
    pattern = re.escape(start_label) + r"\s*[:：]?\s*"
    match = re.search(pattern, text)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for label in end_labels:
        label_pattern = re.escape(label) + r"\s*[:：]?"
        next_match = re.search(label_pattern, text[start:])
        if next_match:
            end = min(end, start + next_match.start())
    return text[start:end].strip()


DIMENSION_NAMES = [
    "内容创作与逻辑",
    "语言表达准确性",
    "篇章结构与衔接",
    "情境契合度",
    "写作规范性",
]

DIMENSION_ALIASES = {
    "content_logic": "内容创作与逻辑",
    "language_accuracy": "语言表达准确性",
    "structure_cohesion": "篇章结构与衔接",
    "context_fit": "情境契合度",
    "writing_norms": "写作规范性",
    "content": "内容创作与逻辑",
    "language": "语言表达准确性",
    "structure": "篇章结构与衔接",
    "coherence": "篇章结构与衔接",
    "fit": "情境契合度",
    "norms": "写作规范性",
}


def format_score_value(score: float) -> str:
    score = float(score)
    return str(int(score)) if score.is_integer() else f"{score:.1f}".rstrip("0").rstrip(".")


def clamp_half_score(value: Any, max_score: float = 5.0) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0 or score > max_score:
        return None
    # 本项目默认采用 0.5 分步长，避免模型输出 3.7、4.2 等难以解释的分数。
    score = round(score * 2) / 2
    return max(0.0, min(max_score, score))


def normalize_dimension_name(raw_key: Any) -> Optional[str]:
    """Normalize a model-output dimension key to the canonical Chinese name."""
    key = str(raw_key).strip()
    dim_name = DIMENSION_ALIASES.get(key, key)
    if dim_name not in DIMENSION_NAMES:
        # Some models may output keys with suffix like “内容创作与逻辑（5分）”.
        matched = next((name for name in DIMENSION_NAMES if name in dim_name), None)
        if matched:
            dim_name = matched
    return dim_name if dim_name in DIMENSION_NAMES else None


def subscore_value_to_score(raw_value: Any) -> Optional[float]:
    """Extract a numeric sub-score from either a plain value or a nested score object."""
    if isinstance(raw_value, dict):
        for key in ["score", "得分", "分数", "value", "point", "points"]:
            if key in raw_value:
                return clamp_half_score(raw_value.get(key), 5.0)
        # Fallback: if the model used an unexpected key but exactly one numeric-like value, accept it.
        numeric_candidates = []
        for value in raw_value.values():
            score = clamp_half_score(value, 5.0)
            if score is not None:
                numeric_candidates.append(score)
        if len(numeric_candidates) == 1:
            return numeric_candidates[0]
        return None
    return clamp_half_score(raw_value, 5.0)


def normalize_sub_scores(sub_scores: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Normalize five dimension scores to Chinese dimension names.

    0625a supports two model-output shapes:
    1. {"内容创作与逻辑": 3.5}
    2. {"内容创作与逻辑": {"score": 3.5, "basis": "..."}}
    """
    if not isinstance(sub_scores, dict):
        return {}
    normalized: Dict[str, float] = {}
    for raw_key, raw_value in sub_scores.items():
        dim_name = normalize_dimension_name(raw_key)
        if not dim_name:
            continue
        score = subscore_value_to_score(raw_value)
        if score is not None:
            normalized[dim_name] = score
    return {name: normalized[name] for name in DIMENSION_NAMES if name in normalized}


def extract_subscore_bases(sub_scores: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Extract per-dimension evidence/basis text from nested sub_scores."""
    if not isinstance(sub_scores, dict):
        return {}
    bases: Dict[str, str] = {}
    for raw_key, raw_value in sub_scores.items():
        dim_name = normalize_dimension_name(raw_key)
        if not dim_name or not isinstance(raw_value, dict):
            continue
        basis = ""
        for key in ["basis", "reason", "依据", "扣分依据", "评价依据", "comment"]:
            if raw_value.get(key):
                basis = str(raw_value.get(key)).strip()
                break
        if basis:
            bases[dim_name] = basis
    return {name: bases[name] for name in DIMENSION_NAMES if name in bases}


def extract_sub_scores_from_text(text: str) -> Dict[str, float]:
    """Extract five dimension scores from Chinese grading comments."""
    if not text:
        return {}
    found: Dict[str, float] = {}
    for name in DIMENSION_NAMES:
        # 支持：内容创作与逻辑：4分 / 内容创作与逻辑：4/5 / 内容创作与逻辑 4.0 分
        pattern = re.escape(name) + r"(?:（?5分）?)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:分|/\s*5)?"
        match = re.search(pattern, text)
        if match:
            score = clamp_half_score(match.group(1), 5.0)
            if score is not None:
                found[name] = score
    return {name: found[name] for name in DIMENSION_NAMES if name in found}


def is_complete_sub_scores(sub_scores: Dict[str, float]) -> bool:
    return all(name in sub_scores for name in DIMENSION_NAMES)


def sum_sub_scores(sub_scores: Dict[str, float]) -> Optional[float]:
    if not is_complete_sub_scores(sub_scores):
        return None
    return round(sum(float(sub_scores[name]) for name in DIMENSION_NAMES), 1)


TEMPLATE_COPY_PATTERNS = [
    {
        "内容创作与逻辑": 3.5,
        "语言表达准确性": 3.0,
        "篇章结构与衔接": 3.5,
        "情境契合度": 4.0,
        "写作规范性": 3.5,
    },
]

TEMPLATE_COPY_MARKERS = [
    "主要优点：...",
    "主要问题：...",
    "修改建议：...",
    "给学生的反馈：...",
    "结合五个维度说明扣分原因",
    "不要笼统套话",
]


def is_same_subscore_pattern(left: Dict[str, float], right: Dict[str, float]) -> bool:
    return is_complete_sub_scores(left) and all(abs(float(left[name]) - float(right[name])) < 0.01 for name in DIMENSION_NAMES)


def looks_like_template_copy(sub_scores: Dict[str, float], raw_text: str, grading_comment: str) -> bool:
    """Detect obvious model copying of an old prompt template instead of actual scoring."""
    if not any(is_same_subscore_pattern(sub_scores, pattern) for pattern in TEMPLATE_COPY_PATTERNS):
        return False
    check_text = f"{raw_text}\n{grading_comment}"
    return any(marker in check_text for marker in TEMPLATE_COPY_MARKERS)


def format_subscore_line(sub_scores: Dict[str, float]) -> str:
    if not is_complete_sub_scores(sub_scores):
        return ""
    parts = [f"{name}：{format_score_value(sub_scores[name])}/5" for name in DIMENSION_NAMES]
    total = sum_sub_scores(sub_scores)
    return "；".join(parts) + f"。五项合计：{format_score_value(total)}/25。"


def extract_score(text: str) -> Optional[float]:
    """Extract total score. If no explicit total exists, fall back to five-dimension sum."""
    if not text:
        return None

    # 优先提取明确总分，不把分项 4/5 误当作总分。
    priority_patterns = [
        r'"score"\s*:\s*"?(\d+(?:\.\d+)?)"?',
        r'"作文总分"\s*:\s*"?(\d+(?:\.\d+)?)"?',
        r"作文总分\s*[:：]\s*(\d+(?:\.\d+)?)\s*/\s*(?:25|二十五)",
        r"总分\s*[:：]\s*(\d+(?:\.\d+)?)\s*/\s*(?:25|二十五)",
        r"作文得分\s*[:：]\s*(\d+(?:\.\d+)?)\s*/\s*(?:25|二十五)",
        r"得分\s*[:：]\s*(\d+(?:\.\d+)?)\s*/\s*(?:25|二十五)",
    ]

    for pattern in priority_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 25:
                    return round(score * 2) / 2
            except ValueError:
                continue

    sub_total = sum_sub_scores(extract_sub_scores_from_text(text))
    if sub_total is not None:
        return sub_total

    # 最后才接受裸 18.5/25，避免误读分项。
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*/\s*25(?!\d)", text)
    if match:
        try:
            score = float(match.group(1))
            if 0 <= score <= 25:
                return round(score * 2) / 2
        except ValueError:
            pass
    return None


def normalize_grading_comment(
    comment: str,
    score: Optional[float],
    sub_scores: Optional[Dict[str, float]] = None,
) -> str:
    """Normalize score display and force total to equal five dimension scores when available."""
    comment = (comment or "").strip()
    if score is None:
        return comment
    score_text = format_score_value(float(score))
    sub_scores = sub_scores or {}
    subscore_line = format_subscore_line(sub_scores)

    # 先修正所有“作文总分：x/y”的分母和分子，避免出现 18.5/27。
    if comment:
        comment = re.sub(
            r"作文总分\s*[:：]\s*\d+(?:\.\d+)?\s*/\s*\d+",
            f"作文总分：{score_text}/25",
            comment,
        )

    if not comment:
        base = f"一、总分\n作文总分：{score_text}/25"
        if subscore_line:
            base += f"\n{subscore_line}"
        return base

    # 去掉模型原有开头的“总分”小节，由程序统一生成，保证总分与分项一致。
    remaining = comment
    if re.match(r"^一[、.．]\s*总分", remaining):
        match = re.search(r"\n\s*二[、.．]\s*分项评价", remaining)
        if match:
            remaining = remaining[match.start() + 1 :].strip()
        else:
            remaining = re.sub(
                r"^一[、.．]\s*总分[\s\S]*?(?=\n\s*[二三四五六][、.．])",
                "",
                remaining,
                count=1,
            ).strip()

    header = f"一、总分\n作文总分：{score_text}/25"
    if subscore_line:
        header += f"\n{subscore_line}"

    if extract_score(remaining) is None and not remaining.startswith("二、分项评价"):
        return header + "\n\n" + remaining
    return header + "\n\n" + remaining


def build_ocr_prompt() -> str:
    return """
/no_think
不要输出思考过程。只输出最终 JSON，不要使用 Markdown 代码块，不要在 JSON 前后添加任何说明。

你是一名英文手写作文识别助手。请只识别图片中的学生英文作文正文，不要评分，不要解释。

要求：
1. 按原有行文顺序转写学生作文。
2. 不要擅自改写、润色或纠正学生原文。
3. 无法确认的单词用 [?] 标记。
4. 排除答题卡模板文字、页眉页脚、扫描水印、无关提示语。
5. 如果图片无法看清，也要说明原因。

请输出严格 JSON，字段如下：
{
  "recognized_text": "学生手写作文正文",
  "recognition_note": "无法识别或不确定之处；没有则写无"
}
""".strip()


def build_scoring_prompt(
    *,
    original_text: str,
    paragraph1_start: str,
    paragraph2_start: str,
    scoring_rubric: str,
    scoring_standard_name: str,
    student_essay: str,
    recognition_note: str,
) -> str:
    return f"""
/no_think
不要输出思考过程。只输出最终 JSON，不要使用 Markdown 代码块，不要在 JSON 前后添加任何说明。

你是一名熟悉中国高考英语读后续写评分要求的高中英语教师。
你的任务是根据“原文阅读材料”“两段续写开头句”“评分标准”和“学生续写作文识别文本”进行评分。

必须遵守：
- 这是“读后续写”评分，不是普通英语作文评分。
- 必须结合原文阅读材料、两段续写开头句、人物、情节、冲突、情感走向和主题方向评分。
- 必须判断学生作文是否自然承接两段给定开头句。
- 必须判断续写情节是否符合原文逻辑、人物性格和主题发展。
- 不要替学生润色后再评分，不要把作文改好后再给高分。
- 识别文本中的 [?] 或识别不确定信息，不应直接视为学生语言错误。

评分校准要求：
- 优秀档：情节完整自然，语言准确流畅，错误很少，主题升华自然。
- 良好档：整体较好，但仍有少量语言、衔接或情节细节问题。
- 合格档：基本完成续写任务，但语言错误较多、情节或人物细节有明显不足。
- 较弱档：内容基本相关，但语言错误多，逻辑、衔接或情境契合存在较大问题。
- 低档：偏离原文、续写不完整、语言严重影响理解或无法形成完整故事。
- 若作文存在多处明显拼写、词形、搭配或句法错误，语言表达准确性应明显扣分；若错误影响理解，语言分不能偏高。
- 若作文语言错误密集且多处影响理解，即使情节完整，也不宜进入良好档。
- 若情节套路化但基本合理，应根据细节充分性和人物契合度区分内容分与情境分，不要机械固定给同一档位。
- 不要把明显不同质量的作文评为同一组分项分数；不要使用固定中间分作为默认答案。
- 每个维度必须先写出具体依据 basis，再给该维度 score；最后用五个维度 score 相加得到总分。
- 五个分项均为 0-5 分，采用 0.5 分步长；总分范围为 0-25。
- score 必须等于五个分项 score 之和，不得独立估计。
- grading_comment 控制在 350-600 字，必须包含“作文总分：__/25”和五个分项得分；分项分数必须与 JSON 中的 sub_scores 完全一致。

【原文阅读材料】
{compact_text(original_text, 2600) if original_text.strip() else "未提取到原文阅读材料。"}

【第一段续写开头句】
{paragraph1_start if paragraph1_start.strip() else "未提取到第一段续写开头句。"}

【第二段续写开头句】
{paragraph2_start if paragraph2_start.strip() else "未提取到第二段续写开头句。"}

【评分标准名称】
{scoring_standard_name if scoring_standard_name.strip() else "默认评分标准"}

【评分标准】
{compact_rubric_for_prompt(scoring_rubric)}

【学生续写作文识别文本】
{compact_text(student_essay, 1800) if student_essay.strip() else "未识别到学生作文文本。"}

【识别备注】
{recognition_note if recognition_note.strip() else "无"}

输出要求：
- 只输出一个 JSON 对象。
- 字段名必须完全一致：sub_scores、score、score_text、grading_comment、review_required、parse_note。
- sub_scores 必须包含五个维度，且五个键名必须完全一致：内容创作与逻辑、语言表达准确性、篇章结构与衔接、情境契合度、写作规范性。
- sub_scores 中每个维度的值必须是一个对象，包含 basis 和 score 两个字段；basis 写该维度的具体评分依据，score 写该维度得分。
- 不要输出任何示例分数，不要复制本提示中的字段说明文字作为评分内容。
- 不要先确定总分再反推分项；必须先判断五个分项，再计算总分。
- grading_comment 必须用中文，按“总分、分项评价、主要优点、主要问题、修改建议、给学生的反馈”的顺序组织。
""".strip()


def _parse_ocr_result(raw_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    obj = extract_json_object(raw_text)
    if obj:
        recognized_text = str(obj.get("recognized_text", "") or "").strip()
        recognition_note = str(obj.get("recognition_note", "") or "").strip() or "无"
    else:
        recognized_text = extract_section(raw_text, "学生手写文本识别结果", ["识别备注"])
        recognition_note = extract_section(raw_text, "识别备注", []) or "无"
        if not recognized_text:
            recognized_text = raw_text.strip()

    if not recognized_text.strip():
        raise GradingError(
            "模型未能识别出学生作文文本。请检查图片是否清晰、是否为英文作文答题区域。",
            code="OCR_MISSING_TEXT",
            stage="ocr_parse",
            details={
                "raw_output_preview": preview_text(raw_text),
                "ollama_metadata": metadata or {},
            },
        )

    return {
        "raw_output": raw_text,
        "recognized_text": recognized_text,
        "recognition_note": recognition_note,
        "parse_warnings": [],
    }


def _parse_scoring_result(
    raw_text: str,
    *,
    recognized_text: str,
    recognition_note: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    parse_warnings = []
    obj = extract_json_object(raw_text)
    sub_scores: Dict[str, float] = {}
    subscore_bases: Dict[str, str] = {}

    if obj:
        # 1. 优先解析模型显式输出的五个分项。
        sub_scores = normalize_sub_scores(obj.get("sub_scores"))
        subscore_bases = extract_subscore_bases(obj.get("sub_scores"))
        if not is_complete_sub_scores(sub_scores):
            # 兼容模型把分项写进 grading_comment 或其他文本字段。
            merged_text = "\n".join(
                str(obj.get(key, "") or "")
                for key in ["grading_comment", "score_text", "score_check", "parse_note"]
            )
            text_scores = extract_sub_scores_from_text(merged_text)
            if is_complete_sub_scores(text_scores):
                sub_scores = text_scores

        score_value = obj.get("score")
        score = None
        try:
            if score_value is not None and str(score_value).strip() != "":
                score = float(score_value)
        except Exception:
            score = None
        grading_comment = str(obj.get("grading_comment", "") or "").strip()
        score_text = str(obj.get("score_text", "") or "").strip()
        parse_note = str(obj.get("parse_note", "") or "").strip()
        review_required = bool(obj.get("review_required", False))

        # 2. 五项齐全时，强制总分等于五项之和。
        sub_total = sum_sub_scores(sub_scores)
        if sub_total is not None:
            if score is not None and abs(float(score) - float(sub_total)) > 0.01:
                parse_warnings.append(
                    f"模型返回总分 {format_score_value(float(score))} 与五项合计 {format_score_value(sub_total)} 不一致，已按五项合计修正。"
                )
            score = sub_total
        elif score is None:
            score = extract_score(score_text) or extract_score(grading_comment) or extract_score(raw_text)
            # 若 JSON 没有 sub_scores，但正文中有五项，也强制用五项合计。
            text_scores = extract_sub_scores_from_text(grading_comment or raw_text)
            text_total = sum_sub_scores(text_scores)
            if text_total is not None:
                sub_scores = text_scores
                if score is not None and abs(float(score) - float(text_total)) > 0.01:
                    parse_warnings.append(
                        f"模型返回总分 {format_score_value(float(score))} 与五项合计 {format_score_value(text_total)} 不一致，已按五项合计修正。"
                    )
                score = text_total

        if not grading_comment and subscore_bases:
            basis_lines = [
                f"{name}：{format_score_value(sub_scores.get(name, 0.0))}/5。依据：{subscore_bases.get(name, '')}"
                for name in DIMENSION_NAMES
                if name in sub_scores and subscore_bases.get(name)
            ]
            if basis_lines:
                grading_comment = "二、分项评价\n" + "\n".join(basis_lines)
        if not grading_comment and score_text:
            grading_comment = score_text
        if parse_note and parse_note != "无":
            parse_warnings.append(parse_note)
        if review_required:
            parse_warnings.append("模型标记该结果需要人工核对。")
    else:
        grading_comment = extract_section(raw_text, "教师评分意见", []) or raw_text
        sub_scores = extract_sub_scores_from_text(grading_comment)
        sub_total = sum_sub_scores(sub_scores)
        score = sub_total if sub_total is not None else (extract_score(grading_comment) or extract_score(raw_text))

    if score is not None:
        score = round(float(score) * 2) / 2
    if score is not None and not (0 <= float(score) <= 25):
        parse_warnings.append(f"模型返回的总分超出 0-25 范围：{score}。")
        score = None

    if looks_like_template_copy(sub_scores, raw_text, grading_comment):
        raise GradingError(
            "模型输出疑似照抄旧评分模板，未根据作文实际质量生成分项得分。系统将尝试重新评分；如仍出现，请查看原始输出。",
            code="MODEL_TEMPLATE_COPY",
            stage="score_parse",
            details={
                "parsed_result": {
                    "raw_output": raw_text,
                    "recognized_text": recognized_text,
                    "recognition_note": recognition_note,
                    "grading_comment": grading_comment,
                    "score": score,
                    "sub_scores": sub_scores,
                    "parse_warnings": parse_warnings + ["疑似模板照抄。"],
                },
                "raw_output_preview": preview_text(raw_text),
                "ollama_metadata": metadata or {},
            },
        )

    grading_comment = normalize_grading_comment(grading_comment, score, sub_scores=sub_scores)
    if not grading_comment:
        parse_warnings.append("模型未按要求返回教师评分意见。")
    if score is not None and not is_complete_sub_scores(sub_scores):
        parse_warnings.append("模型未完整返回五个分项得分；当前总分来自显式总分或文本兜底解析。")

    if score is None:
        raise GradingError(
            "模型返回了评分内容，但未能提取作文总分。请查看原始输出或错误日志，确认是否缺少 JSON score 字段、sub_scores 或“作文总分：__/25”。",
            code="PARSE_MISSING_SCORE",
            stage="score_parse",
            details={
                "parsed_result": {
                    "raw_output": raw_text,
                    "recognized_text": recognized_text,
                    "recognition_note": recognition_note,
                    "grading_comment": grading_comment,
                    "score": None,
                    "sub_scores": sub_scores,
                    "parse_warnings": parse_warnings,
                },
                "raw_output_preview": preview_text(raw_text),
                "ollama_metadata": metadata or {},
            },
        )

    if not grading_comment.strip():
        raise GradingError(
            "模型返回了内容，但未能提取教师评分意见。",
            code="PARSE_MISSING_GRADING_COMMENT",
            stage="score_parse",
            details={
                "raw_output_preview": preview_text(raw_text),
                "recognized_text_preview": preview_text(recognized_text, 1200),
                "ollama_metadata": metadata or {},
            },
        )

    return {
        "raw_output": raw_text,
        "recognized_text": recognized_text,
        "recognition_note": recognition_note,
        "grading_comment": grading_comment,
        "score": score,
        "sub_scores": sub_scores,
        "parse_warnings": parse_warnings,
    }


def parse_vision_grading_result(raw_text: str) -> Dict[str, Any]:
    """Compatibility parser for the old single-step output format."""
    raw_text = (raw_text or "").strip()
    parse_warnings = []

    json_obj = extract_json_object(raw_text)
    if json_obj:
        recognized_text = str(json_obj.get("recognized_text", "") or "").strip()
        recognition_note = str(json_obj.get("recognition_note", "") or "").strip() or "无"
        grading_comment = str(json_obj.get("grading_comment", "") or "").strip()
        score_value = json_obj.get("score")
        score = None
        try:
            if score_value is not None and str(score_value).strip() != "":
                score = float(score_value)
        except Exception:
            score = None
        if score is None:
            score = extract_score(grading_comment) or extract_score(raw_text)
        grading_comment = normalize_grading_comment(grading_comment, score)
    else:
        recognized_text = extract_section(raw_text, "学生手写文本识别结果", ["识别备注", "教师评分意见"])
        recognition_note = extract_section(raw_text, "识别备注", ["教师评分意见"]) or "无"
        grading_comment = extract_section(raw_text, "教师评分意见", []) or raw_text
        score = extract_score(grading_comment) or extract_score(raw_text)

    if score is not None and not (0 <= float(score) <= 25):
        parse_warnings.append(f"模型返回的总分超出 0-25 范围：{score}。")
        score = None
    if not recognized_text:
        parse_warnings.append("模型未按要求返回学生手写文本识别结果，需要人工核对。")
    if not grading_comment:
        parse_warnings.append("模型未按要求返回教师评分意见。")

    return {
        "raw_output": raw_text,
        "recognized_text": recognized_text,
        "recognition_note": recognition_note,
        "grading_comment": grading_comment,
        "score": score,
        "parse_warnings": parse_warnings,
    }


def recognize_and_grade_image(
    image_bytes: bytes,
    original_text: str,
    paragraph1_start: str,
    paragraph2_start: str,
    scoring_rubric: str,
    scoring_standard_name: str = "默认评分标准",
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Dict[str, Any]:
    """
    0620c two-stage pipeline:
    1. Vision OCR: image -> recognized_text.
    2. Text grading: recognized_text + prompt/rubric -> score + grading_comment.
    """
    ocr_prompt = build_ocr_prompt()
    ocr_call = call_ollama_generate(
        prompt=ocr_prompt,
        image_bytes=image_bytes,
        model=model,
        num_ctx=num_ctx,
        num_predict=DEFAULT_NUM_PREDICT_OCR,
        stage="ocr_request",
        json_mode=True,
    )
    ocr_result = _parse_ocr_result(ocr_call["response"], metadata=ocr_call.get("metadata"))

    scoring_prompt = build_scoring_prompt(
        original_text=original_text,
        paragraph1_start=paragraph1_start,
        paragraph2_start=paragraph2_start,
        scoring_rubric=scoring_rubric,
        scoring_standard_name=scoring_standard_name,
        student_essay=ocr_result["recognized_text"],
        recognition_note=ocr_result.get("recognition_note", "无"),
    )
    time.sleep(0.4)
    try:
        try:
            scoring_call = call_ollama_generate(
                prompt=scoring_prompt,
                image_bytes=None,
                model=model,
                num_ctx=num_ctx,
                num_predict=DEFAULT_NUM_PREDICT_GRADING,
                stage="score_request",
                json_mode=True,
                temperature=0.15,
            )
        except GradingError as first_exc:
            if not is_retryable_ollama_error(first_exc):
                raise
            # Local qwen3-vl/Ollama on Windows may return runner/tokenize HTTP 500
            # when the scoring prompt is too heavy. Retry once with a smaller context
            # and shorter output before marking the image failed.
            time.sleep(3.0)
            retry_prompt = build_scoring_prompt(
                original_text=compact_text(original_text, 1600),
                paragraph1_start=paragraph1_start,
                paragraph2_start=paragraph2_start,
                scoring_rubric=compact_rubric_for_prompt(scoring_rubric),
                scoring_standard_name=scoring_standard_name + "（稳定重试）",
                student_essay=compact_text(ocr_result["recognized_text"], 1500),
                recognition_note=ocr_result.get("recognition_note", "无"),
            )
            try:
                scoring_call = call_ollama_generate(
                    prompt=retry_prompt,
                    image_bytes=None,
                    model=model,
                    num_ctx=min(int(num_ctx), FALLBACK_NUM_CTX),
                    num_predict=DEFAULT_NUM_PREDICT_GRADING_FALLBACK,
                    stage="score_request_retry",
                    json_mode=True,
                    temperature=0.1,
                )
                scoring_call.setdefault("metadata", {})["retry_after_error"] = {
                    "first_error_code": first_exc.code,
                    "first_error_message": str(first_exc),
                    "fallback_num_ctx": min(int(num_ctx), FALLBACK_NUM_CTX),
                    "fallback_num_predict": DEFAULT_NUM_PREDICT_GRADING_FALLBACK,
                }
            except GradingError as retry_exc:
                retry_exc.details.setdefault("first_attempt_error", {
                    "code": first_exc.code,
                    "message": str(first_exc),
                    "details": first_exc.details,
                })
                raise retry_exc
    except GradingError as exc:
        exc.details.setdefault("parsed_result", {})
        exc.details["parsed_result"].update(
            {
                "recognized_text": ocr_result.get("recognized_text", ""),
                "recognition_note": ocr_result.get("recognition_note", "无"),
                "raw_output": "【OCR阶段原始输出】\n" + ocr_result.get("raw_output", ""),
            }
        )
        exc.details.setdefault("ocr_metadata", ocr_call.get("metadata", {}))
        raise

    try:
        parsed = _parse_scoring_result(
            scoring_call["response"],
            recognized_text=ocr_result["recognized_text"],
            recognition_note=ocr_result.get("recognition_note", "无"),
            metadata=scoring_call.get("metadata"),
        )
    except GradingError as parse_exc:
        if parse_exc.code != "MODEL_TEMPLATE_COPY":
            raise
        # Retry once if the model copied an old score template. The retry prompt
        # contains no concrete example score and explicitly asks for evidence-first scoring.
        template_retry_prompt = scoring_prompt + (
            "\n\n【重新评分要求】\n"
            "上一轮输出疑似照抄评分模板。请重新阅读学生作文，先为五个维度各写一条具体依据，"
            "再给分项分数，最后把五个分项相加得到总分。不得沿用上一轮分数组合。"
        )
        scoring_call = call_ollama_generate(
            prompt=template_retry_prompt,
            image_bytes=None,
            model=model,
            num_ctx=num_ctx,
            num_predict=DEFAULT_NUM_PREDICT_GRADING,
            stage="score_request_template_retry",
            json_mode=True,
            temperature=0.25,
        )
        parsed = _parse_scoring_result(
            scoring_call["response"],
            recognized_text=ocr_result["recognized_text"],
            recognition_note=ocr_result.get("recognition_note", "无"),
            metadata=scoring_call.get("metadata"),
        )
        parsed.setdefault("parse_warnings", []).append("首次评分疑似模板照抄，已自动重新评分。")

    combined_raw = (
        "【OCR阶段原始输出】\n"
        + ocr_result.get("raw_output", "")
        + "\n\n【评分阶段原始输出】\n"
        + scoring_call.get("response", "")
    )
    parsed["raw_output"] = combined_raw
    parsed["ocr_raw_output"] = ocr_result.get("raw_output", "")
    parsed["grading_raw_output"] = scoring_call.get("response", "")
    parsed["ollama_metadata"] = {
        "ocr": ocr_call.get("metadata", {}),
        "grading": scoring_call.get("metadata", {}),
        "ocr_request": ocr_call.get("request", {}),
        "grading_request": scoring_call.get("request", {}),
    }
    return parsed


# Backward-compatible wrapper retained for older imports/tests.
def grade_continuation_writing(
    original_text: str,
    paragraph1_start: str,
    paragraph2_start: str,
    scoring_rubric: str,
    student_essay: str,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_scoring_prompt(
        original_text=original_text,
        paragraph1_start=paragraph1_start,
        paragraph2_start=paragraph2_start,
        scoring_rubric=scoring_rubric,
        scoring_standard_name="默认评分标准",
        student_essay=student_essay,
        recognition_note="纯文本评分，无图片识别备注。",
    )
    try:
        result = call_ollama_generate(
            prompt=prompt,
            image_bytes=None,
            model=model,
            num_ctx=DEFAULT_NUM_CTX,
            num_predict=DEFAULT_NUM_PREDICT_GRADING,
            timeout=240,
            stage="text_grading",
            json_mode=True,
            temperature=0.15,
        )["response"]
        parsed = _parse_scoring_result(result, recognized_text=student_essay, recognition_note="纯文本评分。")
        return parsed.get("grading_comment", "") or result
    except Exception as exc:
        return f"评分过程出现错误：{exc}"
