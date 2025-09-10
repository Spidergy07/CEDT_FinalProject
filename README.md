# RAG API with TA Tohtoh - ระบบตอบคำถามด้วย AI

ระบบ RAG (Retrieval-Augmented Generation) ที่ใช้ Gemini AI และ Cohere Embeddings สำหรับการตอบคำถามเกี่ยวกับเนื้อหาการเรียนการสอน โดยมี TA Tohtoh เป็นผู้ช่วยสอนที่มีบุคลิกเข้มงวดแต่ใส่ใจนักศึกษา

## 🚀 ฟีเจอร์หลัก

- **Multi-Image Analysis** - วิเคราะห์และสรุปจาก 4 รูปภาพที่เกี่ยวข้องมากที่สุด
- **TA Tohtoh Personality** - ผู้ช่วยสอนที่ตอบเป็นภาษาไทยเท่านั้น มีบุคลิกเข้มงวดแต่ใส่ใจ
- **Semantic Search** - ค้นหาเนื้อหาที่เกี่ยวข้องด้วย Cohere Embeddings
- **Vision AI** - วิเคราะห์รูปภาพด้วย Gemini 2.0 Flash
- **Educational Focus** - เน้นการให้คำตอบเชิงการศึกษา

## 🛠️ เทคโนโลยีที่ใช้

- **Node.js & Express** - Backend API
- **Gemini 2.0 Flash** - Multi-modal AI สำหรับวิเคราะห์รูปภาพ
- **Cohere Embed v4** - Text embeddings สำหรับ semantic search
- **Cosine Similarity** - การคำนวณความเหมือนสำหรับการค้นหา

## 📋 การติดตั้ง

1. **Clone repository**
```bash
git clone https://github.com/Spidergy07/CEDT_FinalProject.git
cd CEDT_FinalProject
git checkout LLM_api
```

2. **ติดตั้ง dependencies**
```bash
npm install
```

3. **ตั้งค่า Environment Variables**
สร้างไฟล์ `.env` และใส่ API keys:
```
CO_API_KEY=your_cohere_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

4. **รัน server**
```bash
node api.js
```

Server จะรันที่ `http://localhost:3000`

## 📡 API Endpoints

### 1. ค้นหารูปภาพที่เกี่ยวข้อง
```http
POST /search
Content-Type: application/json

{
  "question": "สรุปเนื้อหา activity 2"
}
```

### 2. ตอบคำถามจากรูปภาพเดียว
```http
POST /answer
Content-Type: application/json

{
  "question": "อธิบายเนื้อหานี้หน่อย",
  "imagePath": "/path/to/image.png"
}
```

### 3. ค้นหาและตอบคำถาม (แนะนำ)
```http
POST /search-and-answer
Content-Type: application/json

{
  "question": "สรุปเนื้อหา activity 2 ให้หน่อยครับ",
  "topK": 4
}
```

### 4. วิเคราะห์หลายรูปภาพ (ครอบคลุมที่สุด)
```http
POST /search-and-answer-multi
Content-Type: application/json

{
  "question": "สรุปเนื้อหา activity 2 ให้หน่อยครับ",
  "topK": 4
}
```

## 🎭 บุคลิก TA Tohtoh

TA Tohtoh เป็นผู้ช่วยสอนที่:
- **ตอบเป็นภาษาไทยเท่านั้น**
- **เข้มงวดแต่ใส่ใจ** - จะดุเมื่อจำเป็นแต่ช่วยเหลือจริงจัง
- **ตรงไปตรงมา** - ไม่อ้อมค้อม เข้าเรื่องทันที
- **เป็นมิตร** - ใช้ภาษาธรรมชาติ ไม่เป็นทางการเกินไป
- **เชี่ยวชาญ** - ให้คำแนะนำที่เป็นประโยชน์และแม่นยำ

## 📊 ตัวอย่างการใช้งาน

```javascript
// ตัวอย่างการเรียกใช้ API
fetch('http://localhost:3000/search-and-answer', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'สรุปเนื้อหา activity 2 ให้หน่อยครับ'
  })
})
.then(response => response.json())
.then(data => {
  console.log('คำตอบจาก TA Tohtoh:', data.answer.answer);
});
```

## 📁 โครงสร้างโปรเจค

```
final/
├── api.js                          # Main API server
├── package.json                    # Node.js dependencies
├── pdf_image_embeddings.json       # Embeddings data (JSON format)
├── pdf_image_embeddings.npy        # Embeddings data (NumPy format)
├── processed_image_paths.txt       # Image paths mapping
├── pdf_images/                     # Directory containing all PDF images
│   ├── Activity_2_Briefing-*/     # Activity 2 images
│   ├── Central_Processing_Unit-*/  # CPU content images
│   └── ...                        # Other course materials
└── .env                           # Environment variables (not in repo)
```

## ⚡ การทำงานของระบบ

1. **รับคำถาม** - ระบบรับคำถามจากผู้ใช้
2. **สร้าง Embedding** - แปลงคำถามเป็น vector ด้วย Cohere Embed v4
3. **ค้นหาความคล้าย** - คำนวณ cosine similarity กับ embeddings ทั้งหมด
4. **เลือกรูปภาพ** - เลือก 4 รูปที่เกี่ยวข้องมากที่สุด
5. **วิเคราะห์รูปภาพ** - ใช้ Gemini วิเคราะห์เนื้อหาจากแต่ละรูป
6. **สังเคราะห์คำตอบ** - รวมข้อมูลและตอบในลีลา TA Tohtoh

## 🔧 การปรับแต่ง

### เปลี่ยนจำนวนรูปภาพที่วิเคราะห์
```javascript
// ในไฟล์ api.js
async function search(question, maxImgSize = 800, topK = 4) {
  // เปลี่ยน topK เป็นค่าที่ต้องการ (1-10 แนะนำ)
}
```

### ปรับบุคลิก TA Tohtoh
แก้ไข prompts ในไฟล์ `api.js` ส่วน:
- `answer()` function - สำหรับคำตอบจากรูปเดียว
- `answerMultipleImages()` function - สำหรับคำตอบจากหลายรูป

## 🚨 ข้อควรระวัง

- **API Keys** - ต้องมี Cohere API key และ Gemini API key
- **ขนาดไฟล์** - รูปภาพจะถูกแปลงเป็น base64 ซึ่งอาจใหญ่มาก
- **Rate Limits** - Gemini API อาจมีข้อจำกัดการเรียกใช้
- **Memory** - การโหลด embeddings ใช้ RAM ประมาณ 100-200MB

## 📝 License

โปรเจคนี้อยู่ภายใต้ MIT License - ดูรายละเอียดในไฟล์ [LICENSE](LICENSE)

## 👨‍💻 ผู้พัฒนา

- **hotaq (Chinnaphat Khuncharoen)**
- **Repository**: https://github.com/Spidergy07/CEDT_FinalProject

---

💡 **หมายเหตุ**: ระบบนี้พัฒนาขึ้นเพื่อการศึกษาและช่วยเหลือนักศึกษาในการทำความเข้าใจเนื้อหา ไม่ควรใช้แทนการศึกษาจากแหล่งข้อมูลหลัก