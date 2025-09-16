require('dotenv').config()
const express = require('express')
const axios = require('axios');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');

const app = express()
const port = process.env.PORT || 3000

// Enhanced middleware
app.use(express.json({ limit: '10mb' }));
app.use((req, res, next) => {
  res.setTimeout(300000); // 5 minutes timeout
  next();
  print("hrllo")

});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Global error:', err);
  res.status(500).json({ error: 'Internal server error', details: process.env.NODE_ENV === 'development' ? err.message : undefined });
});

// Input validation middleware
const validateRequest = (requiredFields) => (req, res, next) => {
  const missing = requiredFields.filter(field => !req.body[field]);
  if (missing.length > 0) {
    return res.status(400).json({ error: `Missing required fields: ${missing.join(', ')}` });
  }
  next();
};

// Enhanced logging
const log = {
  info: (msg, data = {}) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`, data),
  error: (msg, error = {}) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`, error),
  warn: (msg, data = {}) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`, data)
};

// Free Local Embedding Configuration
const USE_FREE_EMBEDDINGS = process.env.USE_FREE_EMBEDDINGS === 'true';
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
const VOYAGE_MODEL = "voyage-multilingual-2";

async function getEmbedding(text) {
  if (USE_FREE_EMBEDDINGS || !VOYAGE_API_KEY) {
    log.warn('Using free embedding fallback - similarity search will use text matching');
    // For free version, return a simple text-based hash as embedding
    return generateTextHash(text);
  }
  
  // Use Voyage AI if API key is available
  try {
    const response = await axios.post('https://api.voyageai.com/v1/embeddings', {
      input: [text],
      model: VOYAGE_MODEL
    }, {
      headers: {
        'Authorization': `Bearer ${VOYAGE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });
    
    return response.data.data[0].embedding;
  } catch (error) {
    log.error('Voyage AI embedding error, falling back to free method', { error: error.message });
    return generateTextHash(text);
  }
}

function generateTextHash(text) {
  // Simple text-based embedding for free version
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Array(384).fill(0); // Match free model dimension
  
  words.forEach((word, index) => {
    const hash = simpleHash(word);
    embedding[hash % embedding.length] += 1;
  });
  
  // Normalize
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  if (norm > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }
  
  return embedding;
}

function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// Enhanced data storage with caching
class EmbeddingCache {
  constructor() {
    this.docEmbeddings = null;
    this.imgPaths = [];
    this.queryCache = new Map();
    this.maxCacheSize = 1000;
  }

  async loadEmbeddingsData() {
    try {
      const embeddingsPath = path.join(__dirname, 'pdf_image_embeddings.json');
      const pathsFile = path.join(__dirname, 'processed_image_paths.txt');
      
      // Load image paths
      if (fsSync.existsSync(pathsFile)) {
        const pathsData = await fs.readFile(pathsFile, 'utf8');
        this.imgPaths = pathsData.split('\n')
          .map(p => p.trim())
          .filter(p => p && fsSync.existsSync(p)); // Only keep existing files
        log.info(`Loaded ${this.imgPaths.length} valid image paths`);
      }
      
      // Load embeddings with validation
      if (fsSync.existsSync(embeddingsPath)) {
        const embeddingsData = await fs.readFile(embeddingsPath, 'utf8');
        this.docEmbeddings = JSON.parse(embeddingsData);
        
        // Validate embeddings structure
        if (!Array.isArray(this.docEmbeddings) || this.docEmbeddings.length === 0) {
          throw new Error('Invalid embeddings format');
        }
        
        log.info(`Loaded embeddings: ${this.docEmbeddings.length} x ${this.docEmbeddings[0]?.length || 0}`);
        
        // Ensure embeddings match image paths
        if (this.docEmbeddings.length !== this.imgPaths.length) {
          log.warn('Embeddings count mismatch with image paths', {
            embeddings: this.docEmbeddings.length,
            images: this.imgPaths.length
          });
        }
      } else {
        throw new Error('Embeddings file not found');
      }
    } catch (error) {
      log.error('Error loading embeddings data', error);
      throw error;
    }
  }

  getCachedQuery(question) {
    return this.queryCache.get(question);
  }

  setCachedQuery(question, result) {
    if (this.queryCache.size >= this.maxCacheSize) {
      const firstKey = this.queryCache.keys().next().value;
      this.queryCache.delete(firstKey);
    }
    this.queryCache.set(question, result);
  }

  isReady() {
    return this.docEmbeddings && this.imgPaths.length > 0;
  }
}

const embeddingCache = new EmbeddingCache();

// Initialize embedding cache on startup
(async () => {
  try {
    await embeddingCache.loadEmbeddingsData();
    log.info('Embedding cache initialized successfully');
  } catch (error) {
    log.error('Failed to initialize embedding cache', error);
    process.exit(1);
  }
})();

// Tree of Thoughts (ToT) Implementation
class TreeOfThoughts {
  constructor(model, maxDepth = 3, branchingFactor = 3) {
    this.model = model;
    this.maxDepth = maxDepth;
    this.branchingFactor = branchingFactor;
    this.thoughtTree = [];
  }

  // Generate multiple thought branches for a given problem
  async generateThoughts(problem, context = "", depth = 0) {
    const prompt = `คุณคือ TA Tohtoh ที่ใช้ Tree of Thoughts ในการแก้ปัญหา

ปัญหา: ${problem}
บริบท: ${context}
ระดับความลึก: ${depth}/${this.maxDepth}

สร้าง ${this.branchingFactor} แนวทางการคิดที่แตกต่างกันสำหรับปัญหานี้:

1. แนวทางที่ 1: [อธิบายแนวคิดและเหตุผล]
2. แนวทางที่ 2: [อธิบายแนวคิดและเหตุผลที่แตกต่าง]
3. แนวทางที่ 3: [อธิบายแนวคิดและเหตุผลที่แตกต่างอีก]

แต่ละแนวทางควรมีมุมมองที่แตกต่างกันและสามารถนำไปสู่การแก้ปัญหาได้
ตอบเป็นภาษาไทยเท่านั้น`;

    try {
      const result = await this.model.generateContent([prompt]);
      const response = await result.response;
      const thoughts = this.parseThoughts(response.text());
      
      return thoughts.map((thought, index) => ({
        id: `${depth}-${index}`,
        depth: depth,
        content: thought,
        score: 0,
        children: []
      }));
    } catch (error) {
      console.error('Error generating thoughts:', error);
      return [];
    }
  }

  // Parse thoughts from AI response
  parseThoughts(response) {
    const lines = response.split('\n');
    const thoughts = [];
    
    for (const line of lines) {
      const match = line.match(/^\d+\. แนวทางที่ \d+: (.+)/);
      if (match) {
        thoughts.push(match[1].trim());
      }
    }
    
    return thoughts.length > 0 ? thoughts : [response]; // Fallback if parsing fails
  }

  // Evaluate thoughts based on relevance, feasibility, and clarity
  async evaluateThoughts(thoughts, problem, context) {
    for (const thought of thoughts) {
      const evaluationPrompt = `คุณคือ TA Tohtoh ที่ประเมินคุณภาพของแนวคิด

ปัญหา: ${problem}
บริบท: ${context}
แนวคิด: ${thought.content}

ประเมินแนวคิดนี้ตามเกณฑ์:
1. ความเกี่ยวข้อง (0-10)
2. ความเป็นไปได้ (0-10)
3. ความชัดเจน (0-10)

ให้คะแนนรวม (0-30) และอธิบายเหตุผลสั้นๆ
รูปแบบ: คะแนน: X/30 - เหตุผล: [อธิบาย]
ตอบเป็นภาษาไทยเท่านั้น`;

      try {
        const result = await this.model.generateContent([evaluationPrompt]);
        const response = await result.response;
        const scoreMatch = response.text().match(/คะแนน: (\d+)\/30/);
        thought.score = scoreMatch ? parseInt(scoreMatch[1]) : 15; // Default score
        thought.evaluation = response.text();
      } catch (error) {
        console.error('Error evaluating thought:', error);
        thought.score = 15; // Default score on error
      }
    }
    
    return thoughts.sort((a, b) => b.score - a.score);
  }

  // Build the complete thought tree
  async buildTree(problem, context = "") {
    console.log('Building Tree of Thoughts for:', problem);
    
    // Generate initial thoughts
    const initialThoughts = await this.generateThoughts(problem, context, 0);
    const evaluatedThoughts = await this.evaluateThoughts(initialThoughts, problem, context);
    
    this.thoughtTree = evaluatedThoughts;
    
    // Expand the best thoughts to deeper levels
    for (let depth = 1; depth < this.maxDepth; depth++) {
      const currentLevelThoughts = this.thoughtTree.filter(t => t.depth === depth - 1);
      const bestThoughts = currentLevelThoughts.slice(0, 2); // Expand top 2 thoughts
      
      for (const parentThought of bestThoughts) {
        const childThoughts = await this.generateThoughts(
          problem, 
          `${context}\n\nแนวคิดก่อนหน้า: ${parentThought.content}`, 
          depth
        );
        
        const evaluatedChildren = await this.evaluateThoughts(childThoughts, problem, context);
        parentThought.children = evaluatedChildren;
        this.thoughtTree.push(...evaluatedChildren);
      }
    }
    
    return this.thoughtTree;
  }

  // Get the best reasoning path
  getBestPath() {
    const path = [];
    let currentThoughts = this.thoughtTree.filter(t => t.depth === 0);
    
    for (let depth = 0; depth < this.maxDepth; depth++) {
      if (currentThoughts.length === 0) break;
      
      const bestThought = currentThoughts.reduce((best, current) => 
        current.score > best.score ? current : best
      );
      
      path.push(bestThought);
      currentThoughts = bestThought.children || [];
    }
    
    return path;
  }

  // Generate final answer based on the best path
  async generateFinalAnswer(problem, context, bestPath) {
    const pathSummary = bestPath.map((thought, index) => 
      `ขั้นที่ ${index + 1}: ${thought.content} (คะแนน: ${thought.score}/30)`
    ).join('\n');

    const finalPrompt = `คุณคือ TA Tohtoh ที่ใช้ Tree of Thoughts ในการแก้ปัญหา

ปัญหา: ${problem}
บริบท: ${context}

เส้นทางการคิดที่ดีที่สุด:
${pathSummary}

จากเส้นทางการคิดข้างต้น ให้สรุปคำตอบสุดท้ายที่:
1. รวมแนวคิดจากทุกขั้นตอน
2. ให้คำตอบที่ชัดเจนและปฏิบัติได้
3. อธิบายเหตุผลสำคัญ
4. แสดงบุคลิก TA Tohtoh ที่เข้มงวดแต่ใส่ใจ

ตอบเป็นภาษาไทยเท่านั้น`;

    try {
      const result = await this.model.generateContent([finalPrompt]);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error generating final answer:', error);
      return 'ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบสุดท้าย';
    }
  }
}

// Optimized similarity calculation with vectorization
function calculateCosineSimilarity(queryEmb, docEmb) {
  let dotProduct = 0;
  let queryNorm = 0;
  let docNorm = 0;
  
  for (let i = 0; i < queryEmb.length; i++) {
    dotProduct += queryEmb[i] * docEmb[i];
    queryNorm += queryEmb[i] * queryEmb[i];
    docNorm += docEmb[i] * docEmb[i];
  }
  
  return dotProduct / (Math.sqrt(queryNorm) * Math.sqrt(docNorm));
}

// Enhanced search function with caching and optimization
async function search(question, maxImgSize = 800, topK = 4) {
  const startTime = Date.now();
  
  try {
    if (!embeddingCache.isReady()) {
      throw new Error('Embeddings not ready. Please ensure system is initialized.');
    }

    // Check cache first
    const cacheKey = `${question}_${topK}`;
    const cached = embeddingCache.getCachedQuery(cacheKey);
    if (cached) {
      log.info('Cache hit for query', { question: question.substring(0, 50), timeMs: Date.now() - startTime });
      return cached;
    }

    // Enhanced query embedding with retry logic using Voyage AI
    let queryEmb;
    let retries = 3;
    while (retries > 0) {
      try {
        queryEmb = await getEmbedding(question);
        break;
      } catch (embeddingError) {
        retries--;
        if (retries === 0) throw embeddingError;
        log.warn(`Voyage AI embedding retry ${3 - retries}/3`, { error: embeddingError.message });
        await new Promise(resolve => setTimeout(resolve, 1000 * (4 - retries))); // Exponential backoff
      }
    }

    // Optimized similarity computation with parallel processing
    const batchSize = 1000; // Process in batches to manage memory
    const similarities = [];
    const { docEmbeddings, imgPaths } = embeddingCache;
    
    for (let batchStart = 0; batchStart < docEmbeddings.length; batchStart += batchSize) {
      const batchEnd = Math.min(batchStart + batchSize, docEmbeddings.length);
      const batchSimilarities = [];
      
      for (let i = batchStart; i < batchEnd; i++) {
        const similarity = calculateCosineSimilarity(queryEmb, docEmbeddings[i]);
        batchSimilarities.push({
          index: i,
          similarity,
          imagePath: imgPaths[i]
        });
      }
      
      similarities.push(...batchSimilarities);
    }

    // Efficient top-K selection using partial sort
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topResults = similarities.slice(0, Math.min(topK, similarities.length));
    
    // Filter out results with very low similarity (< 0.1)
    const filteredResults = topResults.filter(r => r.similarity > 0.1);
    const finalResults = filteredResults.length > 0 ? filteredResults : topResults.slice(0, 1);
    
    const result = {
      imagePath: finalResults[0].imagePath,
      similarity: finalResults[0].similarity,
      question: question,
      topResults: finalResults,
      processingTimeMs: Date.now() - startTime
    };
    
    // Cache the result
    embeddingCache.setCachedQuery(cacheKey, result);
    
    log.info('Search completed', {
      question: question.substring(0, 50),
      resultsCount: finalResults.length,
      topSimilarity: finalResults[0].similarity.toFixed(4),
      timeMs: result.processingTimeMs
    });
    
    return result;
  } catch (error) {
    log.error('Search function error', { question: question.substring(0, 50), error: error.message });
    throw error;
  }
}
// Enhanced answer function with improved prompts and error handling
async function answer(question, imgPath) {
  const startTime = Date.now();
  
  try {
    log.info('Generating answer', { question: question.substring(0, 50), imagePath: path.basename(imgPath) });
    
    // Validate file exists and is readable
    if (!fsSync.existsSync(imgPath)) {
      throw new Error(`Image file not found: ${imgPath}`);
    }
    
    // Read image with async/await
    const imageData = await fs.readFile(imgPath);
    const base64Image = imageData.toString('base64');
    
    // Enhanced prompt with better instruction structure
    const prompt = `🎓 TA Tohtoh ระบบการเรียนรู้อัจฉริยะ

=== บทบาทและบุคลิกภาพ ===
คุณคือ TA Tohtoh ผู้ช่วยสอนผู้เชี่ยวชาญ ที่มีความรู้ลึกซึ้งและประสบการณ์การสอนมากมาย
• บุคลิก: เข้มงวดแต่เป็นมิตร, ตรงไปตรงมา, ใส่ใจนักศึกษา
• มีความอดทนแต่จะแสดงความผิดหวดเมื่อนักศึกษาไม่ตั้งใจเรียน
• มุ่งเน้นให้นักศึกษาเข้าใจอย่างแท้จริง ไม่ใช่แค่จำ

=== บริบทการทำงาน ===
รูปภาพนี้: เอกสารการเรียนการสอนจากฐานข้อมูลของเรา
คำถามนักศึกษา: "${question}"

=== วิธีการตอบที่ดีเยี่ยม ===
1. 📖 วิเคราะห์เนื้อหา: อ่านและเข้าใจรูปภาพอย่างละเอียด
2. 🎯 ตอบตรงประเด็น: ให้คำตอบที่เกี่ยวข้องโดยตรงกับคำถาม
3. 📊 แยกย่อยเนื้อหา: สำหรับเนื้อหาซับซ้อน ให้แบ่งเป็นขั้นตอนชัดเจน
4. 🔗 เชื่อมโยงความรู้: ربطเนื้อหาเข้ากับแนวคิดใหญ่ของวิชา
5. 💡 เสริมความเข้าใจ: ให้ตัวอย่างหรือการอธิบายเพิ่มเติมเมื่อจำเป็น
6. 🎓 แนะนำการศึกษาต่อ: บอกว่าควรศึกษาอะไรต่อ หรือเชื่อมโยงกับเนื้อหาอื่น

=== รูปแบบการตอบ ===
• ใช้ภาษาไทยที่เป็นธรรมชาติ ไม่เป็นทางการจัด
• แสดงบุคลิกของ TA ที่เข้าใจและใส่ใจ
• ถ้าเนื้อหาพื้นฐาน: อาจแสดงความคาดหวงให้นักศึกษารู้
• ถ้าเนื้อหาซับซ้อน: อธิบายอย่างละเอียดและอดทน

=== ข้อควรระวัง ===
❌ อย่าเริ่มด้วยการทักทายยาวๆ
❌ อย่าเพิกเฉยกับคำถาม
❌ อย่าให้คำตอบสั้นเกินไปโดยไม่มีการอธิบาย

✨ เริ่มตอบเลย พร้อมแสดงความเชี่ยวชาญและความใส่ใจ`;
    
    // Enhanced image preparation with validation
    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: imgPath.toLowerCase().endsWith('.jpg') || imgPath.toLowerCase().endsWith('.jpeg') 
          ? "image/jpeg" : "image/png"
      }
    };
    
    // Generate content with enhanced error handling
    let result, response, answerText;
    try {
      result = await model.generateContent([prompt, imagePart]);
      response = await result.response;
      answerText = response.text();
    } catch (genError) {
      log.warn('Gemini generation failed, retrying with simplified prompt', { error: genError.message });
      
      // Fallback with simpler prompt
      const fallbackPrompt = `TA Tohtoh: วิเคราะห์รูปภาพและตอบ "${question}" ให้ละเอียดและชัดเจน ตอบเป็นภาษาไทย`;
      result = await model.generateContent([fallbackPrompt, imagePart]);
      response = await result.response;
      answerText = response.text();
    }
    
    const processingTime = Date.now() - startTime;
    log.info('Answer generated successfully', { 
      question: question.substring(0, 50), 
      timeMs: processingTime,
      answerLength: answerText.length 
    });
    
    return {
      question: question,
      imagePath: imgPath,
      answer: answerText,
      processingTimeMs: processingTime,
      metadata: {
        imageSize: imageData.length,
        timestamp: new Date().toISOString()
      }
    };
  } catch (error) {
    log.error('Answer function error', { 
      question: question.substring(0, 50), 
      imagePath: path.basename(imgPath),
      error: error.message 
    });
    
    return {
      question: question,
      imagePath: imgPath,
      answer: `ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: ${error.message}\n\nกรุณาตรวจสอบ:\n- ไฟล์รูปภาพสามารถเข้าถึงได้\n- รูปภาพมีคุณภาพดีและอ่านได้\n- ลองใหม่อีกครั้งในอีกสักครู่`,
      processingTimeMs: Date.now() - startTime,
      error: true
    };
  }
}

// Enhanced multi-image analysis with parallel processing
async function answerMultipleImages(question, topResults) {
  const startTime = Date.now();
  const maxImages = Math.min(topResults.length, 4);
  
  try {
    log.info('Starting multi-image analysis', {
      question: question.substring(0, 50),
      imageCount: maxImages
    });
    
    // Parallel image analysis for better performance
    const analysisPromises = topResults.slice(0, maxImages).map(async (result, index) => {
      try {
        const imageData = await fs.readFile(result.imagePath);
        const base64Image = imageData.toString('base64');
        
        // Enhanced extraction prompt
        const prompt = `📝 เครื่องมือดึงข้อมูล - ระบบสมาร์ท AI

=== ภารกิจการดึงข้อมูล ===
วิเคราะห์และดึงข้อมูลทั้งหมดจากเอกสารการเรียนนี้ (อย่ากำหนดว่าเป็นหน้าปก)

ข้อมูลที่ต้องดึง:
📝 ข้อความทั้งหมด: หัวข้อ, เนื้อหา, จุดสำคัญ, คำนิยาม
📈 องค์ประกอบภาพ: แผนภูมิ, กราฟ, ตาราง, โค้ด + คำอธิบาย
💡 แนวคิดหลัก: ทฤษฎี, หลักการ, การปฏิบัติ
🔧 ตัวอย่างปฏิบัติ: โค้ด, ขั้นตอน, การคำนวณ
📁 ข้อมูลเมตา: หมายเลขหน้า, หัวข้อบท, ชื่อวิชา

รูปแบบผลลัพธ์:
• จัดรูปแบบหัวข้อ คำอธิบาย และจุดสำคัญ
• เนื้อหาที่สำคัญให้แบ่งหมวดหมู่อย่างชัดเจน
• บันทึกตัวเลข สูตร ชื่อพิเศษ ที่สำคัญ

ให้ละเอียดและครบถ้วน - เนื้อหานี้จะรวมกับหน้าอื่นเพื่อคำตอบที่สมบูรณ์

ตอบเป็นภาษาไทยเท่านั้น`;
        
        const imagePart = {
          inlineData: {
            data: base64Image,
            mimeType: result.imagePath.toLowerCase().match(/\.(jpg|jpeg)$/i) 
              ? "image/jpeg" : "image/png"
          }
        };
        
        const analysisResult = await model.generateContent([prompt, imagePart]);
        const analysisResponse = await analysisResult.response;
        const analysis = analysisResponse.text();
        
        log.info(`Image ${index + 1} analyzed`, {
          path: path.basename(result.imagePath),
          similarity: result.similarity.toFixed(3),
          contentLength: analysis.length
        });
        
        return {
          imagePath: result.imagePath,
          similarity: result.similarity,
          analysis: analysis,
          order: index + 1
        };
      } catch (imageError) {
        log.error(`Failed to analyze image ${index + 1}`, { 
          path: result.imagePath, 
          error: imageError.message 
        });
        
        return {
          imagePath: result.imagePath,
          similarity: result.similarity,
          analysis: `เกิดข้อผิดพลาดในการวิเคราะห์ภาพนี้: ${imageError.message}`,
          order: index + 1,
          error: true
        };
      }
    });
    
    // Wait for all analyses to complete
    const imageAnalyses = await Promise.all(analysisPromises);
    const successfulAnalyses = imageAnalyses.filter(a => !a.error);
    
    if (successfulAnalyses.length === 0) {
      throw new Error('ไม่สามารถวิเคราะห์ภาพได้เลย');
    }
    
    // Enhanced synthesis with better organization
    const combinedAnalysis = successfulAnalyses
      .sort((a, b) => b.similarity - a.similarity) // Sort by relevance
      .map((item, index) => {
        const filename = path.basename(item.imagePath);
        const relevance = (item.similarity * 100).toFixed(1);
        return `เอกสารที่ ${index + 1} - ${filename} (ความเกี่ยวข้อง: ${relevance}%)\n${'='.repeat(50)}\n${item.analysis}`;
      }).join('\n\n' + '='.repeat(80) + '\n\n');
    
    // Enhanced final synthesis prompt
    const finalPrompt = `🎓 TA Tohtoh ระบบการสังเคราะห์ข้อมูลหลายแหล่ง

=== ข้อมูลการทำงาน ===
คำถามนักศึกษา: "${question}"
จำนวนเอกสารที่วิเคราะห์: ${successfulAnalyses.length} เอกสาร

=== เนื้อหาจากเอกสาร ===
${combinedAnalysis}

=== คำสั่งการตอบ ===
จากเนื้อหาทั้งหมดข้างต้น ให้คุณตอบคำถามอย่าง:

🎯 ตรงตอจุด: ตอบตรงตามที่ถาม - ถ้าถามสรุปให้สรุป
📈 ครบถ้วน: รวมข้อมูลจากทุกแหล่งอย่างมีตรรกะ
📝 ชัดเจน: ใช้โครงสร้างและจุดย่อยสำหรับความชัดเจน
🔗 เชื่อมโยง: แสดงความเชื่อมโยงของแนวคิดต่างๆ
⚠️ ระบุข้อจำกัด: หากข้อมูลไม่เพียงพอ บอกตรงๆ

ความคาดหวัง: ถ้านักศึกษาไม่ตั้งใจเรียน อาจแสดงความผิดหวังเล็กน้อย
ภาษา: ภาษาไทยเท่านั้น น้ำเสียง TA ที่เข้มงวดแต่ใส่ใจ`;
    
    const finalResult = await model.generateContent([finalPrompt]);
    const finalResponse = await finalResult.response;
    const finalAnswer = finalResponse.text();
    
    const processingTime = Date.now() - startTime;
    log.info('Multi-image analysis completed', {
      question: question.substring(0, 50),
      imagesProcessed: successfulAnalyses.length,
      totalTime: processingTime,
      avgTimePerImage: Math.round(processingTime / maxImages)
    });
    
    return {
      question: question,
      imagesAnalyzed: successfulAnalyses.length,
      imagePaths: successfulAnalyses.map(item => item.imagePath),
      answer: finalAnswer,
      processingTimeMs: processingTime,
      individualAnalyses: imageAnalyses,
      metadata: {
        averageSimilarity: (successfulAnalyses.reduce((sum, a) => sum + a.similarity, 0) / successfulAnalyses.length).toFixed(3),
        timestamp: new Date().toISOString(),
        failedAnalyses: imageAnalyses.filter(a => a.error).length
      }
    };
  } catch (error) {
    log.error('Multi-image analysis failed', {
      question: question.substring(0, 50),
      error: error.message
    });
    
    return {
      question: question,
      imagesAnalyzed: 0,
      imagePaths: [],
      answer: `ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์หลายเอกสาร: ${error.message}\n\nกรุณาลองอีกครั้ง หรือลดความซับซ้อนของคำถาม`,
      processingTimeMs: Date.now() - startTime,
      error: true
    };
  }
}

// Enhanced API endpoints with better validation and error handling
app.post("/search", validateRequest(['question']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { question, maxImgSize = 800, topK = 4 } = req.body;
    
    // Input validation
    if (typeof question !== 'string' || question.trim().length === 0) {
      return res.status(400).json({ error: "Valid question string is required" });
    }
    
    if (topK > 10) {
      return res.status(400).json({ error: "topK cannot exceed 10" });
    }
    
    const result = await search(question.trim(), maxImgSize, topK);
    
    res.json({
      ...result,
      metadata: {
        ...result.metadata,
        endpoint: '/search',
        requestTime: Date.now() - startTime
      }
    });
  } catch (error) {
    log.error('Search endpoint error', { error: error.message, question: req.body.question?.substring(0, 50) });
    res.status(500).json({ 
      error: 'Search failed', 
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      requestTime: Date.now() - startTime
    });
  }
});

app.post("/answer", validateRequest(['question', 'imagePath']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { question, imagePath } = req.body;
    
    // Enhanced input validation
    if (typeof question !== 'string' || question.trim().length === 0) {
      return res.status(400).json({ error: "Valid question string is required" });
    }
    
    if (typeof imagePath !== 'string' || !fsSync.existsSync(imagePath)) {
      return res.status(400).json({ error: "Valid image path is required" });
    }
    
    const result = await answer(question.trim(), imagePath);
    
    res.json({
      ...result,
      metadata: {
        ...result.metadata,
        endpoint: '/answer',
        requestTime: Date.now() - startTime
      }
    });
  } catch (error) {
    log.error('Answer endpoint error', { 
      error: error.message, 
      question: req.body.question?.substring(0, 50),
      imagePath: req.body.imagePath 
    });
    res.status(500).json({ 
      error: 'Answer generation failed', 
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      requestTime: Date.now() - startTime
    });
  }
});

// Smart Universal API - Automatically chooses the best method
app.post("/smart-search", validateRequest(['question']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { question, forceMethod = 'auto' } = req.body;
    
    if (typeof question !== 'string' || question.trim().length === 0) {
      return res.status(400).json({ error: "Valid question string is required" });
    }
    
    const cleanQuestion = question.trim();
    log.info('Smart search initiated', { question: cleanQuestion.substring(0, 50) });
    
    // Intelligent analysis of the question
    const questionAnalysis = analyzeQuestion(cleanQuestion);
    log.info('Question analysis', questionAnalysis);
    
    let searchResult, answerResult, method, topK;
    
    // Smart method selection based on question complexity
    switch (forceMethod === 'auto' ? questionAnalysis.bestMethod : forceMethod) {
      case 'simple':
        method = 'simple';
        topK = 3;
        searchResult = await search(cleanQuestion, 800, topK);
        answerResult = await answer(cleanQuestion, searchResult.imagePath);
        break;
        
      case 'comprehensive':
        method = 'comprehensive';
        topK = Math.min(questionAnalysis.estimatedDocuments * 2, 15);
        searchResult = await search(cleanQuestion, 800, topK);
        answerResult = await answerMultipleImages(cleanQuestion, searchResult.topResults);
        break;
        
      case 'tree_of_thoughts':
        method = 'tree_of_thoughts';
        topK = Math.min(questionAnalysis.estimatedDocuments, 10);
        searchResult = await search(cleanQuestion, 800, topK);
        
        // Initialize Tree of Thoughts
        const tot = new TreeOfThoughts(model, 3, 3);
        const imageContext = searchResult.topResults.map((result, index) => {
          const filename = path.basename(result.imagePath);
          return `📄 เอกสาร ${index + 1}: ${filename}`;
        }).join('\n');
        
        const thoughtTree = await tot.buildTree(cleanQuestion, `ข้อมูลอ้างอิง:\n${imageContext}`);
        const bestPath = tot.getBestPath();
        const finalAnswer = await tot.generateFinalAnswer(cleanQuestion, imageContext, bestPath);
        
        answerResult = {
          question: cleanQuestion,
          answer: finalAnswer,
          thoughtProcess: bestPath,
          imagesUsed: searchResult.topResults.length,
          processingTimeMs: Date.now() - startTime - searchResult.processingTimeMs
        };
        break;
        
      default:
        method = 'enhanced_multi';
        topK = questionAnalysis.estimatedDocuments;
        
        // Enhanced search with query expansion
        const expandedQueries = expandQuery(cleanQuestion);
        const searchPromises = expandedQueries.map(query => 
          search(query, 800, Math.ceil(topK / expandedQueries.length))
        );
        
        const searchResults = await Promise.all(searchPromises);
        const combinedResults = combineAndRankResults(searchResults, topK);
        
        searchResult = {
          question: cleanQuestion,
          topResults: combinedResults,
          processingTimeMs: searchResults.reduce((sum, r) => sum + r.processingTimeMs, 0)
        };
        
        answerResult = await answerMultipleImages(cleanQuestion, combinedResults);
    }
    
    const totalTime = Date.now() - startTime;
    
    res.json({
      success: true,
      method: method,
      questionAnalysis: questionAnalysis,
      search: {
        question: searchResult.question,
        topResults: searchResult.topResults || [{ imagePath: searchResult.imagePath, similarity: searchResult.similarity }],
        searchTime: searchResult.processingTimeMs,
        topK: topK
      },
      answer: answerResult,
      performance: {
        totalProcessingTime: totalTime,
        searchTime: searchResult.processingTimeMs,
        analysisTime: answerResult.processingTimeMs || 0,
        efficiency: totalTime < 30000 ? 'excellent' : totalTime < 60000 ? 'good' : 'needs_optimization'
      },
      metadata: {
        endpoint: '/smart-search',
        timestamp: new Date().toISOString(),
        version: '3.0',
        aiSelected: forceMethod === 'auto'
      }
    });
  } catch (error) {
    log.error('Smart search error', { 
      error: error.message, 
      question: req.body.question?.substring(0, 50) 
    });
    res.status(500).json({ 
      success: false,
      error: 'Smart search failed', 
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: Date.now() - startTime
    });
  }
});

// Question analysis function
function analyzeQuestion(question) {
  const words = question.toLowerCase().split(/\s+/);
  const questionWords = ['อะไร', 'ไง', 'คือ', 'what', 'how', 'why', 'explain'];
  const summaryWords = ['สรุป', 'ทั้งหมด', 'รวม', 'ทุก', 'summary', 'all', 'entire'];
  const complexWords = ['เปรียบเทียบ', 'วิเคราะห์', 'อธิบาย', 'ยกตัวอย่าง', 'compare', 'analyze', 'detail'];
  const multiDocWords = ['activity', 'กิจกรรม', 'หลาย', 'ต่าง', 'แต่ละ', 'multiple'];
  
  const hasQuestionWords = words.some(w => questionWords.includes(w));
  const hasSummaryWords = words.some(w => summaryWords.includes(w));
  const hasComplexWords = words.some(w => complexWords.includes(w));
  const hasMultiDocWords = words.some(w => multiDocWords.includes(w));
  
  let complexity = 'simple';
  let estimatedDocuments = 3;
  let bestMethod = 'simple';
  
  if (hasSummaryWords && hasMultiDocWords) {
    complexity = 'comprehensive';
    estimatedDocuments = 8;
    bestMethod = 'comprehensive';
  } else if (hasComplexWords) {
    complexity = 'analytical';
    estimatedDocuments = 6;
    bestMethod = 'tree_of_thoughts';
  } else if (hasMultiDocWords) {
    complexity = 'multi_document';
    estimatedDocuments = 5;
    bestMethod = 'enhanced_multi';
  }
  
  return {
    complexity,
    estimatedDocuments,
    bestMethod,
    hasQuestionWords,
    hasSummaryWords,
    hasComplexWords,
    hasMultiDocWords,
    wordCount: words.length
  };
}

// Query expansion function
function expandQuery(question) {
  const baseQuery = question;
  const queries = [baseQuery];
  
  // Add keyword variations
  if (question.includes('activity')) {
    queries.push(question.replace('activity', 'กิจกรรม'));
    queries.push(question.replace('activity', 'แบบฝึกหัด'));
  }
  
  if (question.includes('สอง')) {
    queries.push(question.replace('สอง', '2'));
    queries.push(question.replace('สอง', 'two'));
  }
  
  return queries.slice(0, 3); // Limit to 3 queries
}

// Combine and rank results from multiple searches
function combineAndRankResults(searchResults, maxResults) {
  const allResults = [];
  const seen = new Set();
  
  for (const searchResult of searchResults) {
    if (searchResult.topResults) {
      for (const result of searchResult.topResults) {
        if (!seen.has(result.imagePath)) {
          seen.add(result.imagePath);
          allResults.push(result);
        }
      }
    }
  }
  
  // Sort by similarity and take top results
  allResults.sort((a, b) => b.similarity - a.similarity);
  return allResults.slice(0, maxResults);
}

// Comprehensive search endpoint for complex questions requiring multiple documents
app.post("/comprehensive-search", validateRequest(['question']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { question, topK = 12 } = req.body;
    
    if (typeof question !== 'string' || question.trim().length === 0) {
      return res.status(400).json({ error: "Valid question string is required" });
    }
    
    if (topK > 20) {
      return res.status(400).json({ error: "topK for comprehensive search cannot exceed 20" });
    }
    
    const cleanQuestion = question.trim();
    log.info('Starting comprehensive search', { question: cleanQuestion.substring(0, 50), topK });
    
    // Multi-stage search for comprehensive coverage
    const searchResult = await search(cleanQuestion, 800, topK);
    
    // Enhanced multi-document analysis
    const answerResult = await answerMultipleImages(cleanQuestion, searchResult.topResults);
    
    const totalTime = Date.now() - startTime;
    
    res.json({
      success: true,
      search: {
        question: searchResult.question,
        topResults: searchResult.topResults,
        totalImagesFound: searchResult.topResults.length,
        searchTime: searchResult.processingTimeMs,
        comprehensiveness: searchResult.topResults.length >= 10 ? 'high' : 'medium'
      },
      answer: answerResult,
      performance: {
        totalProcessingTime: totalTime,
        searchTime: searchResult.processingTimeMs,
        analysisTime: answerResult.processingTimeMs,
        averageTimePerImage: answerResult.imagesAnalyzed > 0 
          ? Math.round(answerResult.processingTimeMs / answerResult.imagesAnalyzed) 
          : 0
      },
      metadata: {
        endpoint: '/comprehensive-search',
        timestamp: new Date().toISOString(),
        version: '2.1',
        recommendedForComplexQuestions: true
      }
    });
  } catch (error) {
    log.error('Comprehensive search error', { 
      error: error.message, 
      question: req.body.question?.substring(0, 50) 
    });
    res.status(500).json({ 
      success: false,
      error: 'Comprehensive search failed', 
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: Date.now() - startTime
    });
  }
});

// Enhanced combined endpoint with optimized processing
app.post("/search-and-answer", validateRequest(['question']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { question, maxImgSize = 800, topK = 4 } = req.body;
    
    // Enhanced validation
    if (typeof question !== 'string' || question.trim().length === 0) {
      return res.status(400).json({ error: "Valid question string is required" });
    }
    
    if (topK > 15) {
      return res.status(400).json({ error: "topK for search-and-answer cannot exceed 15 for performance reasons" });
    }
    
    // Parallel processing preparation
    const cleanQuestion = question.trim();
    
    // Search for relevant images
    log.info('Starting search-and-answer', { question: cleanQuestion.substring(0, 50), topK });
    const searchResult = await search(cleanQuestion, maxImgSize, topK);
    
    // Multi-image analysis
    const answerResult = await answerMultipleImages(cleanQuestion, searchResult.topResults);
    
    const totalTime = Date.now() - startTime;
    
    res.json({
      success: true,
      search: {
        question: searchResult.question,
        topResults: searchResult.topResults,
        totalImagesFound: searchResult.topResults.length,
        searchTime: searchResult.processingTimeMs
      },
      answer: answerResult,
      performance: {
        totalProcessingTime: totalTime,
        searchTime: searchResult.processingTimeMs,
        analysisTime: answerResult.processingTimeMs,
        averageTimePerImage: answerResult.imagesAnalyzed > 0 
          ? Math.round(answerResult.processingTimeMs / answerResult.imagesAnalyzed) 
          : 0
      },
      metadata: {
        endpoint: '/search-and-answer',
        timestamp: new Date().toISOString(),
        version: '2.0'
      }
    });
  } catch (error) {
    log.error('Search and answer endpoint error', { 
      error: error.message, 
      question: req.body.question?.substring(0, 50) 
    });
    res.status(500).json({ 
      success: false,
      error: 'Search and answer failed', 
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: Date.now() - startTime
    });
  }
});

// Deprecated: Use /search-and-answer instead
app.post("/search-and-answer-multi", validateRequest(['question']), async (req, res) => {
  log.warn('Deprecated endpoint accessed', { endpoint: '/search-and-answer-multi' });
  
  // Redirect to the main endpoint
  req.url = '/search-and-answer';
  return app.handle(req, res);
});

// Enhanced Tree of Thoughts API endpoint
app.post('/tree-of-thoughts', validateRequest(['problem']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { problem, context = '', maxDepth = 3, branchingFactor = 3 } = req.body;
    
    // Enhanced validation
    if (typeof problem !== 'string' || problem.trim().length === 0) {
      return res.status(400).json({ error: 'Valid problem statement is required' });
    }
    
    if (maxDepth > 5) {
      return res.status(400).json({ error: 'maxDepth cannot exceed 5' });
    }
    
    if (branchingFactor > 5) {
      return res.status(400).json({ error: 'branchingFactor cannot exceed 5' });
    }

    log.info('Tree of Thoughts request', { 
      problem: problem.substring(0, 50), 
      maxDepth, 
      branchingFactor 
    });
    
    // Initialize Tree of Thoughts with Gemini model
    const tot = new TreeOfThoughts(model, maxDepth, branchingFactor);
    
    // Build the thought tree
    const thoughtTree = await tot.buildTree(problem.trim(), context);
    
    // Get the best reasoning path
    const bestPath = tot.getBestPath();
    
    // Generate final answer
    const finalAnswer = await tot.generateFinalAnswer(problem.trim(), context, bestPath);
    
    const processingTime = Date.now() - startTime;
    
    res.json({
      success: true,
      problem: problem.trim(),
      context,
      thoughtTree,
      bestPath,
      finalAnswer,
      metadata: {
        maxDepth,
        branchingFactor,
        totalThoughts: thoughtTree.length,
        processingTime,
        timestamp: new Date().toISOString(),
        endpoint: '/tree-of-thoughts'
      }
    });
  } catch (error) {
    log.error('Tree of Thoughts error', { 
      error: error.message, 
      problem: req.body.problem?.substring(0, 50) 
    });
    res.status(500).json({ 
      success: false,
      error: 'Tree of Thoughts processing failed',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: Date.now() - startTime
    });
  }
});

// Enhanced Tree of Thoughts with image search integration
app.post('/tot-with-images', validateRequest(['problem']), async (req, res) => {
  const startTime = Date.now();
  try {
    const { 
      problem, 
      context = '', 
      maxImgSize = 800, 
      topK = 4, 
      maxDepth = 3, 
      branchingFactor = 3 
    } = req.body;
    
    // Enhanced validation
    if (typeof problem !== 'string' || problem.trim().length === 0) {
      return res.status(400).json({ error: 'Valid problem statement is required' });
    }
    
    if (maxDepth > 4) {
      return res.status(400).json({ error: 'maxDepth for image-integrated ToT cannot exceed 4' });
    }
    
    if (topK > 12) {
      return res.status(400).json({ error: 'topK for image-integrated ToT cannot exceed 12 for performance reasons' });
    }

    log.info('ToT with images request', {
      problem: problem.substring(0, 50),
      topK,
      maxDepth,
      branchingFactor
    });
    
    // Search for relevant images
    const searchResult = await search(problem.trim(), maxImgSize, topK);
    
    // Create enhanced context with image information
    const imageContext = searchResult.topResults.map((result, index) => {
      const filename = path.basename(result.imagePath);
      const relevance = (result.similarity * 100).toFixed(1);
      return `🖼️ เอกสารที่ ${index + 1}: ${filename} (ความเกี่ยวข้อง: ${relevance}%)`;
    }).join('\n');
    
    const enhancedContext = `${context}${context ? '\n\n' : ''}📁 ข้อมูลอ้างอิงจากเอกสาร:\n${imageContext}`;
    
    // Initialize Tree of Thoughts
    const tot = new TreeOfThoughts(model, maxDepth, branchingFactor);
    
    // Build the thought tree with enhanced context
    const thoughtTree = await tot.buildTree(problem.trim(), enhancedContext);
    
    // Get the best reasoning path
    const bestPath = tot.getBestPath();
    
    // Generate final answer with image context
    const finalAnswer = await tot.generateFinalAnswer(problem.trim(), enhancedContext, bestPath);
    
    const processingTime = Date.now() - startTime;
    
    res.json({
      success: true,
      problem: problem.trim(),
      context: enhancedContext,
      searchResults: {
        totalFound: searchResult.topResults.length,
        averageSimilarity: searchResult.topResults.length > 0 
          ? (searchResult.topResults.reduce((sum, r) => sum + r.similarity, 0) / searchResult.topResults.length).toFixed(3)
          : 0,
        images: searchResult.topResults.map(r => ({
          path: path.basename(r.imagePath),
          similarity: r.similarity.toFixed(3)
        }))
      },
      thoughtTree,
      bestPath,
      finalAnswer,
      performance: {
        totalProcessingTime: processingTime,
        searchTime: searchResult.processingTimeMs,
        thoughtProcessingTime: processingTime - searchResult.processingTimeMs
      },
      metadata: {
        maxDepth,
        branchingFactor,
        totalThoughts: thoughtTree.length,
        imagesFound: searchResult.topResults.length,
        timestamp: new Date().toISOString(),
        endpoint: '/tot-with-images'
      }
    });
  } catch (error) {
    log.error('ToT with images error', {
      error: error.message,
      problem: req.body.problem?.substring(0, 50)
    });
    res.status(500).json({
      success: false,
      error: 'Tree of Thoughts with images failed',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: Date.now() - startTime
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  const isReady = embeddingCache.isReady();
  const cacheSize = embeddingCache.queryCache.size;
  
  res.status(isReady ? 200 : 503).json({
    status: isReady ? 'healthy' : 'not ready',
    embedding_cache: {
      ready: isReady,
      image_count: embeddingCache.imgPaths.length,
      embedding_count: embeddingCache.docEmbeddings?.length || 0,
      cache_size: cacheSize,
      max_cache_size: embeddingCache.maxCacheSize
    },
    uptime: process.uptime(),
    memory: {
      used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
      total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      external: Math.round(process.memoryUsage().external / 1024 / 1024)
    },
    node_version: process.version,
    timestamp: new Date().toISOString()
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  log.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  log.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

process.on('uncaughtException', (error) => {
  log.error('Uncaught exception', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  log.error('Unhandled promise rejection', { reason, promise });
  process.exit(1);
});

app.listen(port, () => {
  log.info(`Server running at http://localhost:${port}`, {
    port,
    environment: process.env.NODE_ENV || 'development',
    nodeVersion: process.version
  });
});