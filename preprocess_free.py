#!/usr/bin/env python3
"""
Free PDF Image Embedding Preprocessor using HuggingFace Transformers
No API keys required - runs completely offline
"""

import os
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False

class FreeEmbeddingProcessor:
    def __init__(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        logger.info("Loading multilingual embedding model...")
        
        # Best free multilingual model for Thai + English
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        logger.info("Model loaded successfully!")
        
    def extract_text_from_image_path(self, image_path: str) -> str:
        """Extract meaningful text from image path for embedding"""
        path_obj = Path(image_path)
        
        # Extract information from filename structure
        parts = path_obj.stem.split('_')
        
        # Common patterns in your file structure
        text_parts = []
        
        for part in parts:
            # Skip page numbers
            if part.startswith('page'):
                continue
            # Add meaningful parts
            if len(part) > 2:  # Skip very short parts
                # Replace common separators
                clean_part = part.replace('-', ' ').replace('.', ' ')
                text_parts.append(clean_part)
        
        # Combine with parent directory info
        parent_info = path_obj.parent.name
        if parent_info != 'pdf_images':
            text_parts.insert(0, parent_info.replace('_', ' '))
        
        # Create searchable text
        full_text = ' '.join(text_parts)
        
        # Add Thai translations for common terms
        translations = {
            'Data Science': 'วิทยาการข้อมูล Data Science',
            'Machine Learning': 'การเรียนรู้ของเครื่อง Machine Learning',
            'Statistics': 'สถิติ Statistics',
            'Probability': 'ความน่าจะเป็น Probability',
            'Algorithm': 'อัลกอริทึม Algorithm',
            'Programming': 'การเขียนโปรแกรม Programming',
            'Computer': 'คอมพิวเตอร์ Computer',
            'Logic': 'ตรรกศาสตร์ Logic',
            'Calculus': 'แคลคูลัส Calculus',
            'Activity': 'กิจกรรม Activity แบบฝึกหัด',
            'Slide': 'สไลด์ Slide เอกสาร',
            'Problem Solving': 'การแก้ปัญหา Problem Solving',
            'Counting': 'การนับ Counting จำนวน',
            'Conditional': 'เงื่อนไข Conditional',
            'Digital': 'ดิจิทัล Digital',
            'Combinational': 'การจัดหมู่ Combinational'
        }
        
        # Add relevant translations
        for en_term, combined_term in translations.items():
            if en_term.lower() in full_text.lower():
                full_text = full_text.replace(en_term, combined_term)
        
        return full_text
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Get embeddings for this batch
                batch_embeddings = self.model.encode(batch, show_progress_bar=True)
                all_embeddings.extend(batch_embeddings.tolist())
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                zero_embedding = [0.0] * 384  # Model dimension
                all_embeddings.extend([zero_embedding] * len(batch))
        
        return all_embeddings
    
    def process_images(self, image_paths_file: str = 'processed_image_paths.txt') -> Dict[str, Any]:
        """Process all images and generate embeddings"""
        
        if not os.path.exists(image_paths_file):
            logger.error(f"Image paths file {image_paths_file} not found")
            return None
        
        # Read image paths
        with open(image_paths_file, 'r', encoding='utf-8') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        # Extract text from all paths
        texts = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                text = self.extract_text_from_image_path(image_path)
                texts.append(text)
                valid_paths.append(image_path)
            except Exception as e:
                logger.error(f"Failed to extract text from {image_path}: {e}")
        
        logger.info(f"Extracted text from {len(texts)} images")
        
        # Get embeddings in batches
        embeddings = self.get_embeddings_batch(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return {
            'embeddings': embeddings,
            'image_paths': valid_paths,
            'model': 'paraphrase-multilingual-MiniLM-L12-v2',
            'dimensions': len(embeddings[0]) if embeddings else 384
        }
    
    def save_embeddings(self, results: Dict[str, Any], 
                       embeddings_file: str = 'pdf_image_embeddings.json',
                       paths_file: str = 'processed_image_paths.txt'):
        """Save embeddings and paths to files"""
        
        # Save embeddings
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(results['embeddings'], f)
        
        # Update paths file with only valid paths
        with open(paths_file, 'w', encoding='utf-8') as f:
            for path in results['image_paths']:
                f.write(f"{path}\n")
        
        # Save metadata
        metadata = {
            'model': results['model'],
            'dimensions': results['dimensions'],
            'total_embeddings': len(results['embeddings']),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cost': 'FREE',
            'provider': 'HuggingFace Sentence Transformers'
        }
        
        with open('embedding_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved {len(results['embeddings'])} embeddings to {embeddings_file}")
        logger.info(f"✅ Model: {results['model']}, Dimensions: {results['dimensions']}")
        logger.info("✅ Cost: 100% FREE!")

def install_requirements():
    """Install required packages if not available"""
    try:
        import subprocess
        import sys
        
        packages = [
            'sentence-transformers',
            'torch',
            'transformers'
        ]
        
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    except Exception as e:
        logger.error(f"Failed to install requirements: {e}")
        logger.info("Please run manually: pip install sentence-transformers torch transformers")

def main():
    """Main processing function"""
    logger.info("🚀 Starting FREE embedding preprocessing...")
    logger.info("💰 Cost: $0.00 (100% Free!)")
    
    # Try to install requirements if needed
    if not HAS_SENTENCE_TRANSFORMERS:
        install_requirements()
        
        # Re-import after installation
        try:
            global SentenceTransformer
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("Failed to import sentence-transformers after installation")
            logger.info("Please install manually: pip install sentence-transformers")
            return
    
    processor = FreeEmbeddingProcessor()
    
    # Process images
    results = processor.process_images()
    
    if results:
        # Save results
        processor.save_embeddings(results)
        logger.info("🎉 Preprocessing completed successfully!")
        logger.info("💰 Total cost: $0.00")
    else:
        logger.error("❌ Preprocessing failed!")

if __name__ == "__main__":
    main()