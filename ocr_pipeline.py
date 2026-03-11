"""
ocr_pipeline.py
─────────────────────────────────────────────────────────────────────────────
YOLO + EasyOCR pipeline cho ingredient label detection.

Flow:
    PIL Image
        ↓
    YOLO detect bbox        ← mock mode khi chưa có best.pt
        ↓
    Crop + preprocess       ← expand bbox, CLAHE, upscale
        ↓
    EasyOCR                 ← extract text
        ↓
    Post-process            ← clean, split, normalize
        ↓
    OCRResult               ← ingredient_list + metadata

Usage:
    from ocr_pipeline import OCRPipeline
    from PIL import Image

    pipeline = OCRPipeline()                        # mock mode
    pipeline = OCRPipeline("models/best.pt")        # real YOLO

    result = pipeline.run(image)
    print(result.ingredient_list)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ── Config ────────────────────────────────────────────────────────────────────

YOLO_CONF_THRESHOLD = 0.5
BBOX_PADDING        = 20     # px padding quanh bbox
MIN_CROP_HEIGHT     = 100    # upscale nếu crop height < này
MIN_INGREDIENT_LEN  = 3      # bỏ token ngắn hơn này
OCR_LANGUAGES       = ["en"]


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    ingredient_list:  list[str]             # cleaned ingredient names → feed vào classifier
    raw_text:         str                   # raw OCR text (debug)
    bbox_detected:    bool                  # YOLO có detect được không
    bbox:             Optional[tuple]       # (x1, y1, x2, y2) hoặc None
    confidence:       float                 # YOLO confidence (1.0 nếu mock)
    preview_image:    Optional[Image.Image] # ảnh với bbox drawn
    latency_ms:       int
    mode:             str                   # "yolo" | "mock"
    error:            Optional[str] = None


# ── Preprocessor ──────────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Preprocess cropped ingredient region trước khi đưa vào EasyOCR.

    Steps:
    1. Expand bbox với padding
    2. Upscale nếu quá nhỏ
    3. Grayscale
    4. CLAHE contrast enhancement
    """

    @staticmethod
    def expand_bbox(
        x1: int, y1: int, x2: int, y2: int,
        img_w: int, img_h: int,
        pad: int = BBOX_PADDING,
    ) -> tuple[int, int, int, int]:
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(img_w, x2 + pad),
            min(img_h, y2 + pad),
        )

    @staticmethod
    def pil_to_cv2(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def process(self, image: Image.Image) -> Image.Image:
        """
        Full preprocess pipeline.
        Input: cropped PIL image (vùng ingredient)
        Output: enhanced PIL image sẵn sàng cho OCR
        """
        cv_img = self.pil_to_cv2(image)

        # Upscale nếu quá nhỏ
        h, w = cv_img.shape[:2]
        if h < MIN_CROP_HEIGHT:
            scale = MIN_CROP_HEIGHT / h
            cv_img = cv2.resize(
                cv_img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # CLAHE — contrast enhancement cho text trên nền phức tạp
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Convert lại RGB cho EasyOCR
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return self.cv2_to_pil(rgb)


# ── OCR Post-processor ────────────────────────────────────────────────────────

class OCRPostProcessor:
    """
    Clean raw OCR text → ingredient list.

    Handles:
    - Header stripping: "Ingredients:", "INCI:", "Contains:"
    - Hyphenated line breaks: "Niacin-\namide" → "Niacinamide"
    - Parenthetical inline: "Water (Aqua)" → giữ nguyên (classifier handle)
    - Noise filtering: số, ký tự đặc biệt, tokens quá ngắn
    """

    # Header patterns thường xuất hiện trên nhãn
    HEADER_PATTERNS = [
        r"^ingredients?\s*:?\s*",
        r"^inci\s*:?\s*",
        r"^contains?\s*:?\s*",
        r"^composition\s*:?\s*",
        r"^ingrédients?\s*:?\s*",   # French
        r"^inhaltsstoffe\s*:?\s*",  # German
    ]

    # Noise patterns cần xóa
    NOISE_PATTERNS = [
        r"\d+(\.\d+)?\s*%",    # percentages: "2%", "0.5%"
        r"^\d+$",               # pure numbers
        r"^[^\w]+$",            # pure punctuation
    ]

    def clean(self, raw_text: str, kb_names: list[str] | None = None) -> list[str]:
        """
        Input : raw OCR text, optional KB names for greedy matching
        Output: cleaned ingredient name list

        Handles 2 separator formats:
        - Comma-separated : "Water, Glycerin, Niacinamide"
        - Space-separated : "Water Mineral Oil Glycerin" (Eucerin style)
        """
        text = raw_text

        # Fix hyphenated line breaks: "Niacin-\namide" → "Niacinamide"
        text = re.sub(r"-\s*\n\s*", "", text)

        # Strip header
        text_lower = text.lower()
        for pattern in self.HEADER_PATTERNS:
            match = re.match(pattern, text_lower, re.MULTILINE)
            if match:
                text = text[match.end():]
                break

        # Detect separator format
        has_commas = text.count(",") >= 2

        if has_commas:
            # Comma-separated — split bình thường
            raw_tokens = re.split(r"[,\n]+", text)
        else:
            # Space/newline separated — dùng KB greedy longest-match nếu có
            joined = " ".join(line.strip() for line in text.splitlines() if line.strip())
            joined = re.sub(r"\s+", " ", joined).strip()

            if kb_names:
                raw_tokens = self._greedy_match(joined, kb_names)
            else:
                # Fallback: split theo newline
                raw_tokens = re.split(r"\n+", text)

        ingredients = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue

            # Skip noise patterns
            skip = False
            for noise in self.NOISE_PATTERNS:
                if re.match(noise, token, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            # Skip quá ngắn
            clean_token = re.sub(r"[^\w]", "", token)
            if len(clean_token) < MIN_INGREDIENT_LEN:
                continue

            # Strip trailing/leading punctuation
            token = re.sub(r"^[^a-zA-Z0-9(]+", "", token)
            token = re.sub(r"[^a-zA-Z0-9)]+$", "", token)

            if token:
                ingredients.append(token)

        return ingredients

    @staticmethod
    def _greedy_match(text: str, kb_names: list[str]) -> list[str]:
        """
        Greedy longest-match tokenizer dùng KB names.

        "Water Mineral Oil Glycerin" + KB có "Mineral Oil"
        → ["Water", "Mineral Oil", "Glycerin"]
        """
        # Build normalized lookup: lower → original name
        # Sort by length descending để ưu tiên match dài nhất
        normalized = sorted(
            [(name.lower(), name) for name in kb_names],
            key=lambda x: len(x[0]),
            reverse=True,
        )

        text_lower = text.lower()
        results = []
        pos = 0
        n = len(text_lower)

        while pos < n:
            # Skip leading spaces
            while pos < n and text_lower[pos] == " ":
                pos += 1
            if pos >= n:
                break

            matched = False
            for norm_name, orig_name in normalized:
                end = pos + len(norm_name)
                if text_lower[pos:end] == norm_name:
                    # Check word boundary — không match giữa chừng
                    if end < n and text_lower[end].isalpha():
                        continue
                    results.append(orig_name)
                    pos = end
                    matched = True
                    break

            if not matched:
                # Lấy word tiếp theo (fallback)
                end = pos
                while end < n and text_lower[end] != " ":
                    end += 1
                word = text[pos:end].strip()
                if word:
                    results.append(word)
                pos = end

        return results


# ── OCR Engine ────────────────────────────────────────────────────────────────

class EasyOCREngine:
    """
    Wrapper cho EasyOCR với lazy loading.
    Model download ~100MB lần đầu, cache sau.
    """

    def __init__(self, languages: list[str] = OCR_LANGUAGES):
        self.languages = languages
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr
        print(f"[EasyOCR] Loading model (languages: {self.languages})...")
        # gpu=False để stable trên Apple Silicon
        # EasyOCR MPS support còn experimental
        self._reader = easyocr.Reader(self.languages, gpu=False)
        print("[EasyOCR] Ready ✓")

    def read(self, image: Image.Image) -> str:
        """
        Extract text từ PIL image.
        Returns: raw text string (joined từ all detections)
        """
        self._load()

        img_array = np.array(image)

        results = self._reader.readtext(
            img_array,
            paragraph   = True,     # group text thành đoạn
            min_size    = 10,
            contrast_ths= 0.1,
            text_threshold = 0.6,
        )

        # Join tất cả text detections
        lines = []
        for detection in results:
            if len(detection) == 2:
                _, text = detection
            else:
                _, text, conf = detection
                if conf < 0.3:
                    continue
            lines.append(text.strip())

        return "\n".join(lines)


# ── LLM Cleaner ───────────────────────────────────────────────────────────────

class LLMCleaner:
    """
    Dùng Qwen2.5:7b (Ollama) để clean raw OCR text → ingredient list.

    Chỉ chạy khi rule-based postprocessor ra kết quả kém:
    - Có noise tokens (numbers, foreign text)
    - Hyphenated words chưa được fix
    - Semicolons/periods thay vì commas

    Prompt rất ngắn gọn, yêu cầu output JSON array.
    """

    SYSTEM_PROMPT = """You are an ingredient list extractor.
Given raw OCR text from a cosmetic product label, extract ONLY the ingredient names.
Return a JSON array of strings. No explanation, no markdown, no extra text.
Example output: ["Water", "Glycerin", "Niacinamide", "Phenoxyethanol"]"""

    USER_PROMPT_TEMPLATE = """Extract ingredients from this OCR text:
{raw_text}

Return only a JSON array."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
    ):
        self.ollama_url = ollama_url
        self.model = model

    def clean(self, raw_text: str) -> list[str] | None:
        """
        Returns cleaned ingredient list, hoặc None nếu LLM fail.
        Caller fallback về rule-based nếu None.
        """
        import json
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": self.USER_PROMPT_TEMPLATE.format(raw_text=raw_text)},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic
                "num_predict": 512,
            },
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            content = data["message"]["content"].strip()

            # Strip markdown fences nếu có
            content = content.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]

        except Exception as e:
            print(f"[LLMCleaner] Failed: {e}")

        return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class OCRPipeline:
    """
    Full pipeline: Image → ingredient list.

    Args:
        yolo_model_path: Path to best.pt
                         None → mock mode (full image, no detection)
        use_llm_cleaner: Dùng LLM để clean OCR text (default: True)
        ollama_url:      Ollama server URL

    Mock mode:
        Không cần best.pt, dùng full image cho OCR.
        Swap sang real mode bằng cách pass yolo_model_path.
    """

    def __init__(
        self,
        yolo_model_path: Optional[str | Path] = None,
        use_llm_cleaner: bool = True,
        ollama_url: str = "http://localhost:11434",
    ):
        self._yolo = None
        self._use_yolo = False
        self._preprocessor = ImagePreprocessor()
        self._ocr = EasyOCREngine()
        self._postprocessor = OCRPostProcessor()
        self._llm_cleaner = LLMCleaner(ollama_url=ollama_url) if use_llm_cleaner else None

        if yolo_model_path and Path(yolo_model_path).exists():
            self._load_yolo(yolo_model_path)
        else:
            if yolo_model_path:
                print(f"[OCRPipeline] YOLO model not found: {yolo_model_path}")
            print("[OCRPipeline] Running in MOCK mode (no YOLO detection)")

    def _load_yolo(self, model_path: str | Path) -> None:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(str(model_path))
            self._use_yolo = True
            print(f"[OCRPipeline] YOLO loaded: {model_path}")
        except Exception as e:
            print(f"[OCRPipeline] Failed to load YOLO: {e} → fallback to mock mode")

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect(self, image: Image.Image) -> tuple[bool, Optional[tuple], float]:
        """
        Returns: (detected, bbox, confidence)
        bbox = (x1, y1, x2, y2) in pixel coords
        """
        if not self._use_yolo:
            # Mock: full image
            w, h = image.size
            return False, (0, 0, w, h), 1.0

        results = self._yolo(image, conf=YOLO_CONF_THRESHOLD, verbose=False)

        if not results or len(results[0].boxes) == 0:
            # Fallback: full image
            w, h = image.size
            return False, (0, 0, w, h), 0.0

        # Lấy bbox confidence cao nhất
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        conf     = float(boxes.conf[best_idx])
        xyxy     = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)

        return True, (x1, y1, x2, y2), conf

    # ── Preview image ─────────────────────────────────────────────────────────

    def _draw_bbox(
        self,
        image: Image.Image,
        bbox: tuple,
        detected: bool,
    ) -> Image.Image:
        """Draw bbox lên ảnh gốc để hiện trong UI."""
        preview = image.copy()
        draw    = ImageDraw.Draw(preview)
        x1, y1, x2, y2 = bbox
        color = "#22c55e" if detected else "#94a3b8"  # green if detected, gray if mock
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        return preview

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, image: Image.Image, kb_names: list[str] | None = None) -> OCRResult:
        """
        Full pipeline run.

        Args:
            image: PIL Image từ st.camera_input hoặc st.file_uploader

        Returns:
            OCRResult với ingredient_list + metadata
        """
        t0 = time.time()

        # 1. Detect bbox
        detected, bbox, conf = self._detect(image)

        # 2. Crop + expand bbox
        x1, y1, x2, y2 = bbox
        x1e, y1e, x2e, y2e = self._preprocessor.expand_bbox(
            x1, y1, x2, y2,
            image.width, image.height,
        )
        cropped = image.crop((x1e, y1e, x2e, y2e))

        # 3. Preprocess
        processed = self._preprocessor.process(cropped)

        # 4. OCR
        try:
            raw_text = self._ocr.read(processed)
        except Exception as e:
            return OCRResult(
                ingredient_list = [],
                raw_text        = "",
                bbox_detected   = detected,
                bbox            = bbox,
                confidence      = conf,
                preview_image   = self._draw_bbox(image, bbox, detected),
                latency_ms      = int((time.time() - t0) * 1000),
                mode            = "yolo" if self._use_yolo else "mock",
                error           = str(e),
            )

        # 5. Post-process — LLM cleaner → fallback rule-based
        ingredient_list = None

        if self._llm_cleaner:
            ingredient_list = self._llm_cleaner.clean(raw_text)
            if ingredient_list:
                print(f"[OCRPipeline] LLM cleaner: {len(ingredient_list)} ingredients")
            else:
                print("[OCRPipeline] LLM cleaner failed → fallback rule-based")

        if not ingredient_list:
            ingredient_list = self._postprocessor.clean(raw_text, kb_names=kb_names)

        # 6. Preview
        preview = self._draw_bbox(image, (x1, y1, x2, y2), detected)

        return OCRResult(
            ingredient_list = ingredient_list,
            raw_text        = raw_text,
            bbox_detected   = detected,
            bbox            = bbox,
            confidence      = conf,
            preview_image   = preview,
            latency_ms      = int((time.time() - t0) * 1000),
            mode            = "yolo" if self._use_yolo else "mock",
        )

    @property
    def mode(self) -> str:
        return "yolo" if self._use_yolo else "mock"


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test với mock text image
    from PIL import ImageFont, ImageDraw, Image as PILImage

    # Tạo test image với ingredient text
    img = PILImage.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)
    test_text = (
        "Ingredients: Water, Glycerin, Niacinamide,\n"
        "Sodium Hyaluronate, Phenoxyethanol,\n"
        "Parfum (Fragrance), Panthenol (Vitamin B5)"
    )
    draw.text((20, 20), test_text, fill="black")

    # Run pipeline
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline   = OCRPipeline(model_path)

    print(f"\n[Test] Mode: {pipeline.mode}")
    result = pipeline.run(img)

    print(f"\n[Result]")
    print(f"  Mode          : {result.mode}")
    print(f"  YOLO detected : {result.bbox_detected}")
    print(f"  Confidence    : {result.confidence:.2f}")
    print(f"  Latency       : {result.latency_ms}ms")
    print(f"  Raw text      :\n{result.raw_text}")
    print(f"\n  Ingredients ({len(result.ingredient_list)}):")
    for ing in result.ingredient_list:
        print(f"    - {ing}")

    if result.error:
        print(f"\n  Error: {result.error}")