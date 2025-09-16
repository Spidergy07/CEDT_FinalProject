#!/usr/bin/env python3
"""
Enhanced PDF Image Embedding Preprocessor using Voyage AI
Supports Thai language and multilingual content with superior accuracy
"""

import os
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VoyageEmbeddingProcessor:
    def __init__(self):
        self.api_key = os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        self.model = "voyage-multilingual-2"  # Best for Thai + multilingual content
        self.base_url = "https://api.voyageai.com/v1/embeddings"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Rate limiting
        self.requests_per_minute = 600  # Voyage AI limit
        self.min_delay = 60 / self.requests_per_minute
        self.last_request_time = 0
        
    def get_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Get embedding from Voyage AI with retry logic"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        for attempt in range(retries):
            try:
                payload = {
                    "input": [text],
                    "model": self.model
                }
                
                response = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    return data['data'][0]['embedding']
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise Exception(f"Failed to get embedding after {retries} attempts")
    
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
            'Data Science': 'วิทยาการข้อมูล',
            'Machine Learning': 'การเรียนรู้ของเครื่อง',
            'Statistics': 'สถิติ',
            'Probability': 'ความน่าจะเป็น',
            'Algorithm': 'อัลกอริทึม',
            'Programming': 'การเขียนโปรแกรม',
            'Computer': 'คอมพิวเตอร์',
            'Logic': 'ตรรกศาสตร์',
            'Calculus': 'แคลคูลัส',
            'Activity': 'กิจกรรม',
            'Slide': 'สไลด์',
            'Problem Solving': 'การแก้ปัญหา'
        }
        
        for en_term, th_term in translations.items():
            if en_term.lower() in full_text.lower():
                full_text += f' {th_term}'
        
        return full_text
    
    def process_images(self, image_paths_file: str = 'processed_image_paths.txt') -> Dict[str, Any]:
        """Process all images and generate embeddings"""
        
        if not os.path.exists(image_paths_file):
            logger.error(f"Image paths file {image_paths_file} not found")
            return None
        
        # Read image paths
        with open(image_paths_file, 'r', encoding='utf-8') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        embeddings = []
        valid_paths = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Extract text from path
                text = self.extract_text_from_image_path(image_path)
                
                # Get embedding
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                valid_paths.append(image_path)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images...")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(embeddings)} out of {len(image_paths)} images")
        
        return {
            'embeddings': embeddings,
            'image_paths': valid_paths,
            'model': self.model,
            'dimensions': len(embeddings[0]) if embeddings else 0
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
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('embedding_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(results['embeddings'])} embeddings to {embeddings_file}")
        logger.info(f"Model: {results['model']}, Dimensions: {results['dimensions']}")

def main():
    """Main processing function"""
    logger.info("Starting Voyage AI embedding preprocessing...")
    
    processor = VoyageEmbeddingProcessor()
    
    # Process images
    results = processor.process_images()
    
    if results:
        # Save results
        processor.save_embeddings(results)
        logger.info("Preprocessing completed successfully!")
    else:
        logger.error("Preprocessing failed!")

if __name__ == "__main__":
    main()