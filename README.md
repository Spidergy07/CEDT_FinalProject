# Study PDF Platform - RAG-ready

A comprehensive PDF study platform with AI-powered question answering using Cohere embeddings and Google Gemini for image analysis.

## Features

- **AI-Powered Q&A**: Ask questions about PDF documents and get intelligent answers
- **Multi-Image Analysis**: Analyzes multiple relevant pages for comprehensive answers
- **Thai Language Support**: Optimized for Thai educational content with TA Tohtoh personality
- **Modern UI**: Responsive design with dark theme
- **Real-time Chat**: Interactive chat interface for document queries

## Tech Stack

### Backend
- Node.js with Express
- Cohere AI for embeddings
- Google Gemini for image analysis
- CORS enabled for frontend communication

### Frontend
- Vanilla HTML, CSS, JavaScript
- Responsive design
- Real-time chat interface

## Setup Instructions

### 1. Backend Setup

```bash
cd backend
npm install
```

### 2. Environment Variables

Create a `.env` file in the backend directory:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```
CO_API_KEY=your_cohere_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start the Application

```bash
# Start the backend server
cd backend
npm start

# The application will be available at http://localhost:3000
```

## API Endpoints

- `POST /search-and-answer` - Main endpoint for asking questions
- `POST /search` - Search for relevant images
- `POST /answer` - Answer questions about specific images
- `GET /` - Serves the frontend

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Ask questions about the PDF documents in Thai or English
3. The AI will analyze relevant pages and provide comprehensive answers

## File Structure

```
├── backend/
│   ├── api.js                 # Main server file
│   ├── package.json           # Backend dependencies
│   ├── pdf_images/            # Processed PDF images
│   ├── pdf_image_embeddings.json  # Pre-computed embeddings
│   └── processed_image_paths.txt  # Image paths mapping
├── frontend/
│   ├── index.html             # Main frontend file
│   └── styles.css             # Styling
└── README.md                  # This file
```

## Development

The platform is designed to work with pre-processed PDF images and embeddings. The backend loads these on startup and uses them for semantic search and question answering.

## License

This project is part of a CEDT Final Project.
