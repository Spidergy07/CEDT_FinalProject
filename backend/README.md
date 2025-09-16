# RAG API with TA Tohtoh (Backend)

ระบบ RAG (Retrieval-Augmented Generation) ใช้ Cohere Embeddings สำหรับค้นหาเนื้อหา และ Gemini สำหรับวิเคราะห์ภาพ พร้อมบุคลิก TA Tohtoh (ตอบเป็นภาษาไทย เข้มงวดแต่ใจดี)

## ฟีเจอร์หลัก

- Multi‑Image Analysis (สูงสุด 4 รูป)
- Semantic Search ด้วย Cohere Embed v4
- Vision Analysis ด้วย Gemini 2.0 Flash
- เสิร์ฟ Frontend สถิตจากโฟลเดอร์ `../frontend`

## ความต้องการ

- Node.js 18+ และ npm
- API keys: `CO_API_KEY`, `GEMINI_API_KEY`
- ข้อมูลค้นหา: `backend/pdf_image_embeddings.json` และ `backend/processed_image_paths.txt`

## การติดตั้งและรัน

วิธีที่ 1 (แนะนำ): ใช้สคริปต์จากโฟลเดอร์รากของโปรเจกต์

- `./deploy.sh start` — ติดตั้ง dependencies, ตรวจ `.env`, แล้วรันเซิร์ฟเวอร์เป็น background
- `./deploy.sh logs` — ดู log
- `./deploy.sh stop` — หยุดเซิร์ฟเวอร์

วิธีที่ 2 (Manual) ในโฟลเดอร์ `backend`

```bash
cd backend
cp env.example .env   # จากนั้นแก้ค่า API keys ให้ถูกต้อง
npm ci                 # หรือ npm install
npm start              # หรือ node api.js
```

ค่าเริ่มต้นรันที่ `http://localhost:3000`. สามารถกำหนดพอร์ตได้ด้วยตัวแปร `PORT` ใน `.env` (ไฟล์นี้อยู่ใน `backend/.env`).

ตัวอย่าง `.env`:
```
CO_API_KEY=your_cohere_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
PORT=3000
```

หมายเหตุเกี่ยวกับข้อมูลคำค้นหา (จำเป็นต่อ endpoint การค้นหา/ตอบ):
- `backend/pdf_image_embeddings.json` — เวกเตอร์ embedding ของทุกหน้า/ภาพ
- `backend/processed_image_paths.txt` — รายการ path ของรูปที่สัมพันธ์กับแต่ละ embedding

## API Endpoints

- POST `/search`
  - Body: `{ "question": string, "topK?": number }`
  - ผลลัพธ์: ภาพที่เกี่ยวข้องสูงสุดตาม similarity

- POST `/answer`
  - Body: `{ "question": string, "imagePath": string }`
  - ผลลัพธ์: คำตอบวิเคราะห์จากรูปเดียว

- POST `/search-and-answer`
  - Body: `{ "question": string, "topK?": number }`
  - ผลลัพธ์: ค้นหา + ตอบจากรูปที่เกี่ยวข้องที่สุด

- POST `/search-and-answer-multi`
  - Body: `{ "question": string, "topK?": number }`
  - ผลลัพธ์: ค้นหา + วิเคราะห์หลายรูป แล้วสรุปคำตอบรวม

- GET `/api/files/:year/:subject`
  - Query: ไม่มี
  - ผลลัพธ์: โครงสร้างไฟล์ (tree) ของวิชาที่ระบุ เพื่อใช้แสดงไฟล์ PDF

- GET `/api/pdf/:year/:subject?path=<relative_path>`
  - Stream ไฟล์ PDF ตาม path ที่ปลอดภัยในโฟลเดอร์ของวิชานั้น

- GET `/`
  - ส่งคืนหน้า `frontend/index.html` (เสิร์ฟสถิตจาก `../frontend`)

## โครงสร้าง (ย่อ)

```
CEDT_FinalProject/
├── backend/
│   ├── api.js
│   ├── package.json
│   ├── .env (ไม่เข้าระบบ git)
│   ├── pdf_image_embeddings.json
│   ├── processed_image_paths.txt
│   └── pdf_images/
├── frontend/
│   ├── index.html
│   └── styles.css
└── deploy.sh
```

## หมายเหตุและคำเตือน

- เปิด CORS แล้ว (เพื่อให้ frontend เรียก API ได้)
- ข้อมูล embedding มีขนาดใหญ่ ควรตรวจ RAM เพียงพอ (100–200MB+)
- หากไม่มีไฟล์ embeddings/paths บาง endpoint จะตอบไม่ได้

## ผู้พัฒนา

- hotaq (Chinnaphat Khuncharoen)
- Repository: https://github.com/Spidergy07/CEDT_FinalProject
