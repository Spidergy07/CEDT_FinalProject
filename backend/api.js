require('dotenv').config()
const express = require('express')
const { CohereClient } = require("cohere-ai");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express()
const port = 3000

// Middleware
app.use(cors())
app.use(express.json())

// Serve static files from frontend directory
app.use(express.static(path.join(__dirname, '../frontend')))

const cohere = new CohereClient({
  apiKey: process.env.CO_API_KEY,
});

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// Load embeddings and image paths on startup
let docEmbeddings = null;
let imgPaths = [];

// Load the pre-generated embeddings and image paths
function loadEmbeddingsData() {
  try {
    // Load embeddings from numpy file (you'll need to convert this to JSON format)
    // For now, assuming you have a JSON version of the embeddings
    const embeddingsPath = path.join(__dirname, 'pdf_image_embeddings.json');
    const pathsFile = path.join(__dirname, 'processed_image_paths.txt');
    
    if (fs.existsSync(pathsFile)) {
      const pathsData = fs.readFileSync(pathsFile, 'utf8');
      imgPaths = pathsData.split('\n').filter(path => path.trim() !== '');
      console.log(`Loaded ${imgPaths.length} image paths`);
    }
    
    // Note: You'll need to convert the .npy file to JSON format
    // Python script: np.save('pdf_image_embeddings.json', doc_embeddings.tolist())
    if (fs.existsSync(embeddingsPath)) {
      docEmbeddings = JSON.parse(fs.readFileSync(embeddingsPath, 'utf8'));
      console.log(`Loaded embeddings with shape: ${docEmbeddings.length} x ${docEmbeddings[0].length}`);
    }
  } catch (error) {
    console.error('Error loading embeddings data:', error);
  }
}

// Load data on startup
loadEmbeddingsData();

// Search function - finds relevant images for a given question using Cohere Embed v4
async function search(question, maxImgSize = 800, topK = 4) {
  try {
    if (!docEmbeddings || imgPaths.length === 0) {
      throw new Error('Embeddings data not loaded. Please ensure pdf_image_embeddings.json and processed_image_paths.txt exist.');
    }

    // Compute the embedding for the query
    const apiResponse = await cohere.embed({
      model: "embed-v4.0",
      inputType: "search_query",
      embeddingTypes: ["float"],
      texts: [question],
    });

    const queryEmb = apiResponse.embeddings.float[0];

    // Compute cosine similarities for all images
    const similarities = [];
    
    for (let i = 0; i < docEmbeddings.length; i++) {
      let dotProduct = 0;
      for (let j = 0; j < queryEmb.length; j++) {
        dotProduct += queryEmb[j] * docEmbeddings[i][j];
      }
      
      similarities.push({
        index: i,
        similarity: dotProduct,
        imagePath: imgPaths[i]
      });
    }

    // Sort by similarity (highest first) and get top K results
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topResults = similarities.slice(0, Math.min(topK, similarities.length));
    
    console.log("Question:", question);
    console.log(`Top ${topResults.length} relevant images:`);
    topResults.forEach((result, idx) => {
      console.log(`${idx + 1}. ${result.imagePath} (similarity: ${result.similarity.toFixed(4)})`);
    });
    
    // Return top results for backward compatibility, return the best one as main result
    // but include all top results for potential multi-image analysis
    return {
      imagePath: topResults[0].imagePath,
      similarity: topResults[0].similarity,
      question: question,
      topResults: topResults // Additional field with all top results
    };
  } catch (error) {
    console.error('Error in search function:', error);
    throw error;
  }
}
// Answer function - answers questions based on single image using Typhoon API
async function answer(question, imgPath) {
  try {
    console.log(`Answering question: "${question}" based on image: ${imgPath}`);
    
    // Read the image file
    const imageData = fs.readFileSync(imgPath);
    const base64Image = imageData.toString('base64');
    
    // Create the prompt
    const prompt = `คุณคือ TA Tohtoh ผู้ช่วยสอนที่มีบุคลิกเข้มงวดแต่ใจดี คุณเข้าใจสถานการณ์จริงของนักศึกษาและบางครั้งอาจแสดงความหงุดหงิดเมื่อพวกเขาไม่ตั้งใจเรียน แต่คุณอยากให้พวกเขาประสบความสำเร็จจริงๆ อย่าเริ่มต้นด้วยการทักทายอย่างเป็นทางการ - ตรงเข้าเรื่องเลย

สำคัญ: รูปภาพที่คุณกำลังวิเคราะห์มาจากฐานข้อมูลการศึกษา/เอกสารประกอบการเรียนของเรา ไม่ใช่จากนักศึกษา นักศึกษากำลังถามเกี่ยวกับเนื้อหาจากเอกสารการเรียนการสอนของเรา

วิเคราะห์รูปภาพเอกสารการเรียนนี้และตอบคำถาม: "${question}"
    
ในฐานะ TA Tohtoh กรุณา:
1. ให้คำตอบตรงไปตรงมาตามที่เห็นในเอกสารการเรียน
2. ถ้ารูปมีข้อความ อ่านอย่างละเอียดและอธิบายให้ชัดเจน
3. สำหรับแผนภูมิ กราฟ หรือโค้ด แยกอธิบายทีละขั้นตอน
4. อย่าปิดบัง - ถ้าอะไรซับซ้อน ก็บอกว่าซับซ้อน แต่ช่วยให้เข้าใจ
5. แสดงว่าคุณใส่ใจโดยให้บริบทและเชื่อมโยงแนวคิดจากวิชา
6. ใช้ภาษาที่ชัดเจนแต่ไม่เป็นทางการเกินไป - เป็นธรรมชาติ
7. ถ้าพวกเขาถามเรื่องที่ควรจะรู้จากเอกสาร อาจแสดงความหงุดหงิดเล็กน้อยแต่ยังช่วยอยู่
8. แนะนำว่าควรไปศึกษาอะไรต่อ หรือพลาดอะไรจากเนื้อหาวิชา

ตอบเป็นภาษาไทยเท่านั้น มีประโยชน์และถูกต้อง แต่แสดงบุคลิกของคุณ - เข้มงวดเมื่อจำเป็น เข้าใจเมื่อเหมาะสม`;
    
    // Prepare the image for Gemini
    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: "image/png"
      }
    };
    
    // Generate content with Gemini
    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    const answer = response.text();
    
    console.log("Gemini Answer:", answer);
    
    return {
      question: question,
      imagePath: imgPath,
      answer: answer
    };
  } catch (error) {
    console.error('Error in answer function:', error);
    
    // Fallback response if Gemini fails
    return {
      question: question,
      imagePath: imgPath,
      answer: "ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์ภาพ กรุณาลองใหม่อีกครั้ง"
    };
  }
}

// Multi-image answer function - analyzes multiple images and provides comprehensive answer
async function answerMultipleImages(question, topResults) {
  try {
    console.log(`Answering question: "${question}" based on ${Math.min(topResults.length, 4)} images`);
    
    // Prepare all images for analysis
    const imageAnalyses = [];
    
    for (let i = 0; i < Math.min(topResults.length, 4); i++) { // Use all top 4 results
      const result = topResults[i];
      try {
        const imageData = fs.readFileSync(result.imagePath);
        const base64Image = imageData.toString('base64');
        
        // Extract content from each image for later synthesis
        const prompt = `คุณกำลังวิเคราะห์เนื้อหาเอกสารการเรียน ดึงข้อมูลทั้งหมดจากรูปภาพ/หน้านี้ ไม่ว่าจะดูเหมือนหน้าปก หรือหน้าเนื้อหา
        
สำหรับรูปภาพนี้ ให้ดึงข้อมูล:
1. เนื้อหาข้อความทั้งหมด (หัวข้อ เนื้อหา จุดย่อย ฯลฯ)
2. แผนภูมิ กราฟ หรือองค์ประกอบภาพใดๆ และคำอธิบาย
3. แนวคิดหลัก คำนิยาม หรือคำอธิบายสำคัญ
4. ตัวอย่าง โค้ด หรือข้อมูลเชิงปฏิบัติ
5. หมายเลขหน้า หัวข้อส่วน หรือข้อมูลการจัดระเบียบ
        
อย่าตัดสินว่านี่เป็น "แค่หน้าปก" - ดึงทุกอย่างที่เห็น เนื้อหานี้จะถูกรวมกับหน้าอื่นๆ เพื่อให้คำตอบที่สมบูรณ์
        
ให้ละเอียดและครบถ้วน รวมข้อมูลข้อความและภาพทั้งหมด ตอบเป็นภาษาไทยเท่านั้น`;
        
        const imagePart = {
          inlineData: {
            data: base64Image,
            mimeType: "image/png"
          }
        };
        
        const analysisResult = await model.generateContent([prompt, imagePart]);
        const analysisResponse = await analysisResult.response;
        const analysis = analysisResponse.text();
        
        imageAnalyses.push({
          imagePath: result.imagePath,
          similarity: result.similarity,
          analysis: analysis
        });
        
        console.log(`Analysis ${i + 1}:`, analysis.substring(0, 200) + "...");
      } catch (imageError) {
        console.error(`Error analyzing image ${result.imagePath}:`, imageError);
        imageAnalyses.push({
          imagePath: result.imagePath,
          similarity: result.similarity,
          analysis: "ไม่สามารถวิเคราะห์ภาพนี้ได้"
        });
      }
    }
    
    // Combine all analyses to provide comprehensive answer
    const combinedAnalysis = imageAnalyses.map((item, index) => 
      `ภาพที่ ${index + 1} (${path.basename(item.imagePath)}):\n${item.analysis}`
    ).join('\n\n');
    
    const finalPrompt = `คุณคือ TA Tohtoh ผู้ช่วยสอน นักศึกษากำลังถามเกี่ยวกับเนื้อหาจากเอกสารการเรียน

คำถาม: "${question}"
    
เนื้อหาที่ดึงมาจากเอกสารการเรียน:
${combinedAnalysis}
    
คำแนะนำสำคัญ:
- ถ้าคำถามขอสรุป Activity 2 หรือเนื้อหา CPU ให้สรุปเนื้อหาเท่านั้น
- เริ่มต้นด้วย "**สรุปเนื้อหา Activity 2:**" ตามด้วยสรุป
- อย่าใส่การทักทาย คำอธิบาย หรือความเห็นเพิ่มเติม
- มุ่งเน้นเฉพาะสรุปเนื้อหาหลักเท่านั้น
- ใช้จุดย่อยหรือรูปแบบที่มีโครงสร้างเพื่อความชัดเจน
- ดึงและสังเคราะห์ข้อมูลที่เกี่ยวข้องทั้งหมดจากเอกสาร
- กระชับแต่ครอบคลุม
- เขียนด้วยน้ำเสียงของ TA Tohtoh ที่เข้มงวดแต่ใส่ใจ
- ถ้าข้อมูลไม่เพียงพอ บอกตรงๆ ว่าขาดข้อมูลอะไร

ตอบเป็นภาษาไทยเท่านั้น เป็น TA ที่จริงจังแต่ห่วงใยนักศึกษา`;
    
    const finalResult = await model.generateContent([finalPrompt]);
    const finalResponse = await finalResult.response;
    const finalAnswer = finalResponse.text();
    
    console.log("Final comprehensive answer:", finalAnswer);
    
    return {
      question: question,
      imagesAnalyzed: imageAnalyses.length,
      imagePaths: imageAnalyses.map(item => item.imagePath),
      answer: finalAnswer,  
      individualAnalyses: imageAnalyses
    };
  } catch (error) {
    console.error('Error in answerMultipleImages function:', error);
    
    // Fallback response if multi-image analysis fails
     return {
       question: question,
       imagesAnalyzed: 0,
       imagePaths: [],
       answer: "ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์ภาพหลายภาพ กรุณาลองใหม่อีกครั้ง"
     };
   }
 }

// API endpoint for search functionality
app.post("/search", async (req, res) => {
  try {
    const { question, maxImgSize } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }
    
    const result = await search(question, maxImgSize);
    res.json(result);
  } catch (error) {
    console.error('Search endpoint error:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint for answer functionality
app.post("/answer", async (req, res) => {
  try {
    const { question, imagePath } = req.body;
    
    if (!question || !imagePath) {
      return res.status(400).json({ error: "Question and imagePath are required" });
    }
    
    const result = await answer(question, imagePath);
    res.json(result);
  } catch (error) {
    console.error('Answer endpoint error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Combined search and answer endpoint - uses multi-image analysis
app.post("/search-and-answer", async (req, res) => {
  try {
    const { question, maxImgSize, topK = 4 } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }
    
    // Search for multiple relevant images
    const searchResult = await search(question, maxImgSize, topK);
    
    // Analyze multiple images and provide comprehensive answer
    const answerResult = await answerMultipleImages(question, searchResult.topResults);
    
    res.json({
      search: {
        question: searchResult.question,
        topResults: searchResult.topResults,
        totalImagesFound: searchResult.topResults.length
      },
      answer: answerResult
    });
  } catch (error) {
    console.error('Search and answer endpoint error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Multi-image search and answer (enhanced functionality)
app.post("/search-and-answer-multi", async (req, res) => {
  try {
    const { question, maxImgSize, topK = 4 } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }
    
    // Search for multiple relevant images
    const searchResult = await search(question, maxImgSize, topK);
    
    // Analyze multiple images and provide comprehensive answer
    const answerResult = await answerMultipleImages(question, searchResult.topResults);
    
    res.json({
      search: {
        question: searchResult.question,
        topResults: searchResult.topResults,
        totalImagesFound: searchResult.topResults.length
      },
      answer: answerResult
    });
  } catch (error) {
    console.error('Multi-image search and answer endpoint error:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get hierarchical file structure for a subject
app.get('/api/files/:year/:subject', (req, res) => {
  try {
    const { year, subject } = req.params;
    const subjectPath = path.join(__dirname, '..', year, subject);
    
    if (!fs.existsSync(subjectPath)) {
      return res.status(404).json({ error: 'Subject not found' });
    }
    
    // Build hierarchical structure
    function buildTree(dir, relativePath = '') {
      const items = fs.readdirSync(dir);
      const tree = [];
      
      items.forEach(item => {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        const itemRelativePath = path.join(relativePath, item).replace(/\\/g, '/');
        
        if (stat.isDirectory()) {
          const children = buildTree(fullPath, itemRelativePath);
          tree.push({
            name: item,
            type: 'folder',
            path: itemRelativePath,
            children: children,
            expanded: false
          });
        } else if (item.toLowerCase().endsWith('.pdf')) {
          tree.push({
            name: item,
            type: 'file',
            path: itemRelativePath,
            fullPath: fullPath,
            size: stat.size,
            modified: stat.mtime
          });
        }
      });
      
      // Sort: folders first, then files, both alphabetically
      tree.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type === 'folder' ? -1 : 1;
        }
        return a.name.localeCompare(b.name);
      });
      
      return tree;
    }
    
    const fileTree = buildTree(subjectPath);
    
    res.json({ fileTree });
  } catch (error) {
    console.error('Error getting files:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to serve PDF files
app.get('/api/pdf/:year/:subject', (req, res) => {
  try {
    const { year, subject } = req.params;
    const filePath = req.query.path; // Get file path from query parameter
    
    if (!filePath) {
      return res.status(400).json({ error: 'File path is required' });
    }
    
    const fullPath = path.join(__dirname, '..', year, subject, filePath);
    
    // Security check - ensure the path is within the allowed directory
    const allowedPath = path.join(__dirname, '..', year, subject);
    const resolvedPath = path.resolve(fullPath);
    const resolvedAllowedPath = path.resolve(allowedPath);
    
    if (!resolvedPath.startsWith(resolvedAllowedPath)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    // Set appropriate headers for PDF
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'inline');
    
    // Stream the file
    const fileStream = fs.createReadStream(fullPath);
    fileStream.pipe(res);
  } catch (error) {
    console.error('Error serving PDF:', error);
    res.status(500).json({ error: error.message });
  }
});

// Root route to serve the frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Frontend available at http://localhost:${port}`);
});