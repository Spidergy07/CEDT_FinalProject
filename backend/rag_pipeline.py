import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # defer error until use

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # defer error until use

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
except Exception as e:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None  # defer error until use
    Qwen2VLProcessor = None

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:  # pragma: no cover
    process_vision_info = None


# ----------------------------
# Configuration & Setup
# ----------------------------
@dataclass
class PipelineConfig:
    index_path: str = "image.index"
    mapping_path: str = "mapping.txt"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    clip_model_name: str = "clip-ViT-B-32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 400
    final_max_tokens: int = 600
    min_pixels: int = 224 * 224
    max_pixels: int = 512 * 512
    normalize_embeddings: bool = True  # cosine/IP search expects normalized vectors


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------
# Enhanced Retriever Class
# ----------------------------
class ImageRetriever:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.clip_model = None
        self.faiss_index = None
        self.mapping: Dict[int, Dict[str, str]] = {}
        self._load_components()

    def _require(self, cond: bool, msg: str) -> None:
        if not cond:
            raise RuntimeError(msg)

    def _load_components(self) -> None:
        """Load CLIP model, FAISS index, and mapping with error handling"""
        try:
            self._require(SentenceTransformer is not None, "sentence-transformers is not installed")
            self._require(faiss is not None, "faiss is not installed")

            logger.info("Loading CLIP model…")
            # SentenceTransformer chooses device internally; pass cpu for stability if CUDA absent
            st_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = SentenceTransformer(self.config.clip_model_name, device=st_device)

            logger.info("Loading FAISS index…")
            index_path = Path(self.config.index_path)
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            self.faiss_index = faiss.read_index(str(index_path))

            logger.info("Loading mapping…")
            self._load_mapping()

        except Exception as e:
            logger.error(f"Failed to load retriever components: {e}")
            raise

    def _load_mapping(self) -> None:
        """Load image path mapping with validation"""
        mapping_path = Path(self.config.mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        with open(mapping_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        # Support either: idx \t topic \t path  OR  idx \t path
                        if len(parts) == 3:
                            topic, path = parts[1], parts[2]
                        else:
                            topic, path = "Unknown", parts[-1]
                        self.mapping[idx] = {"topic": topic, "path": path}
                    else:
                        logger.warning(f"Invalid mapping format at line {line_num}: {line.strip()}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing mapping line {line_num}: {e}")

    def _encode_query(self, query: str) -> np.ndarray:
        self._require(self.clip_model is not None, "CLIP model not loaded")
        q_emb = self.clip_model.encode(query.strip(), convert_to_numpy=True).astype("float32")
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
        return q_emb

    def search_images(self, query: str, k: int = 4) -> List[Dict]:
        """Search top-k images using CLIP + FAISS with validation"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        self._require(self.faiss_index is not None, "FAISS index not loaded")

        try:
            q_emb = self._encode_query(query)
            D, I = self.faiss_index.search(np.array([q_emb]), k)

            results: List[Dict] = []
            for rank, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx == -1:
                    continue
                meta = self.mapping.get(int(idx))
                if meta:
                    results.append({
                        "rank": rank + 1,
                        "path": meta["path"],
                        "topic": meta.get("topic", "Unknown"),
                        "score": float(score),
                    })
                else:
                    logger.warning(f"Index {idx} not found in mapping")

            return results

        except Exception as e:
            logger.error(f"Error during image search: {e}")
            raise


# ----------------------------
# Enhanced Generator Class
# ----------------------------
class ContentGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vl_model = None
        self.vl_processor = None
        self._setup_model()

    def _require(self, cond: bool, msg: str) -> None:
        if not cond:
            raise RuntimeError(msg)

    def _setup_model(self) -> None:
        """Initialize Qwen2-VL model with safe CPU/CUDA fallback"""
        try:
            self._require(Qwen2VLForConditionalGeneration is not None, "Qwen2VL model class not available")
            self._require(Qwen2VLProcessor is not None, "Qwen2VL processor class not available")
            self._require(process_vision_info is not None, "qwen_vl_utils.process_vision_info not available")

            # Help avoid CUDA OOM fragmentation
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            logger.info("Loading Qwen2-VL model…")
            use_cuda = torch.cuda.is_available() and self.config.device.startswith("cuda")

            model_kwargs = {}
            if use_cuda:
                # Prefer 4-bit on GPUs that support it; gracefully fall back.
                model_kwargs.update(dict(torch_dtype=torch.float16, device_map="auto"))
                try:
                    model_kwargs.update(dict(load_in_4bit=True))
                except Exception:
                    pass
            else:
                model_kwargs.update(dict(torch_dtype=torch.float32))

            self.vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            ).eval()

            if not use_cuda:
                self.vl_model.to(self.config.device)

            # Processor: do not pass unknown kwargs; tweak attributes if present
            self.vl_processor = Qwen2VLProcessor.from_pretrained(self.config.model_id)
            # Optionally adjust min/max pixels if supported
            try:
                ip = getattr(self.vl_processor, "image_processor", None)
                if ip is not None:
                    if hasattr(ip, "min_pixels"):
                        setattr(ip, "min_pixels", self.config.min_pixels)
                    if hasattr(ip, "max_pixels"):
                        setattr(ip, "max_pixels", self.config.max_pixels)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            raise

    def analyze_single_page(self, query: str, image_path: str) -> str:
        """Extract content from a single page with error handling"""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(str(p)).convert("RGB")

            chat_template = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f"Extract all important details relevant to: {query}",
                        },
                    ],
                }
            ]

            text = self.vl_processor.apply_chat_template(
                chat_template, tokenize=False, add_generation_prompt=True
            )

            image_inputs, _ = process_vision_info(chat_template)

            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Send tensors to the right device; model may be sharded when device_map="auto"
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = self.vl_model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)

            # Trim prompt tokens
            gen_trim = [out[len(in_ids) :] for in_ids, out in zip(inputs["input_ids"], gen_ids)]
            return self.vl_processor.batch_decode(gen_trim, skip_special_tokens=True)[0]

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return f"[Error extracting content: {str(e)}]"

    def synthesize_content(self, query: str, page_summaries: List[str]) -> str:
        """Create final synthesis from multiple page summaries"""
        if not page_summaries:
            return "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม"

        combined_text = "\n\n".join(page_summaries)
        final_prompt = f"""
คุณคือ TA Tohtoh ผู้ช่วยสอนที่เข้มงวดแต่ใจดี
นักศึกษาถามว่า: "{query}"

นี่คือสรุปที่ดึงจากหลายหน้าเอกสาร:
{combined_text}

ตอนนี้ กรุณาสรุปเป็นเนื้อหาแบบมีโครงสร้าง:
- รวมประเด็นสำคัญทั้งหมดจากทุกหน้า
- ใช้ bullet point หรือหัวข้อย่อย
- น้ำเสียงเข้มงวดแต่ใส่ใจนักเรียน
- กระชับแต่ครอบคลุม
- ตอบเป็นภาษาไทยเท่านั้น
"""

        try:
            chat_template = [
                {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
            ]
            text = self.vl_processor.apply_chat_template(
                chat_template, tokenize=False, add_generation_prompt=True
            )

            inputs = self.vl_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = self.vl_model.generate(**inputs, max_new_tokens=self.config.final_max_tokens)

            gen_trim = [out[len(in_ids) :] for in_ids, out in zip(inputs["input_ids"], gen_ids)]
            return self.vl_processor.batch_decode(gen_trim, skip_special_tokens=True)[0]

        except Exception as e:
            logger.error(f"Error in content synthesis: {e}")
            return f"เกิดข้อผิดพลาดในการสรุปข้อมูล: {str(e)}"


# ----------------------------
# Main Pipeline Class
# ----------------------------
class RAGPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.retriever = ImageRetriever(self.config)
        self.generator = ContentGenerator(self.config)
        logger.info("RAG Pipeline initialized successfully")

    def process_pages(self, query: str, retrieved_paths: List[str]) -> List[str]:
        """Process multiple pages and return summaries"""
        page_summaries: List[str] = []
        for path in retrieved_paths:
            try:
                summary = self.generator.analyze_single_page(query, path)
                page_summaries.append(f"📄 {Path(path).name}:\n{summary}")
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                page_summaries.append(f"📄 {Path(path).name}: [Error extracting: {e}]")
        return page_summaries

    def run(self, query: str, top_k: int = 4) -> Dict:
        """Run complete retriever + generator pipeline"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: {query}")

        try:
            retrieved = self.retriever.search_images(query, k=top_k)
            retrieved_paths = [r["path"] for r in retrieved]

            if not retrieved_paths:
                return {
                    "question": query,
                    "answer": "ไม่พบเอกสารที่เกี่ยวข้องกับคำถาม",
                    "sources": [],
                    "details": [],
                }

            page_summaries = self.process_pages(query, retrieved_paths)
            final_answer = self.generator.synthesize_content(query, page_summaries)

            result = {
                "question": query,
                "answer": final_answer,
                "sources": retrieved_paths,
                "details": retrieved,
            }

            logger.info("Pipeline completed successfully")
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise


# ----------------------------
# Convenience Functions
# ----------------------------
def create_pipeline(custom_config: Optional[Dict] = None) -> RAGPipeline:
    """Factory function to create pipeline with custom config"""
    config = PipelineConfig()
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return RAGPipeline(config)


def run_pipeline(query: str, top_k: int = 4, custom_config: Optional[Dict] = None) -> Dict:
    """Quick function to run pipeline with minimal setup"""
    pipeline = create_pipeline(custom_config)
    return pipeline.run(query, top_k)

