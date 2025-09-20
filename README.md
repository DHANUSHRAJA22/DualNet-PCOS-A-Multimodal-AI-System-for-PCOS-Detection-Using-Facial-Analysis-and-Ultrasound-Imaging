# Multimodal PCOS Analyzer

A comprehensive, production-ready frontend for PCOS screening analysis built with Vite + React + TypeScript + Tailwind CSS + shadcn/ui + Framer Motion.

## Features

- **Dual Image Upload**: Support for both facial images and uterus X-rays
- **Camera Capture**: Built-in camera functionality for real-time image capture
- **EXIF Orientation Fix**: Automatic image orientation correction
- **Responsive Design**: Mobile-first approach with WCAG accessibility compliance
- **Real-time Analysis**: Integration with Flask backend for AI-powered screening
- **Smooth Animations**: Framer Motion powered scroll animations and micro-interactions
- **Production Ready**: Type-safe, optimized, and deployment-ready

## Quick Start

### Development

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open http://localhost:8080 in your browser

### Backend Setup

Ensure your Flask backend is running on port 5000:

```bash
# In your backend directory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The frontend will automatically proxy API requests to `http://127.0.0.1:5000`.

## Production Usage

### Build for Production

```bash
npm run build
npm run preview
```

### Deploy with Custom API

Deploy the built application to any static hosting service. Use the `?api=` query parameter to specify your backend URL:

```
https://your-app.com?api=https://your-backend-api.com
```

Example:
```
https://pcos-analyzer.netlify.app?api=https://api.pcos-backend.com
```

## API Integration

### Backend Requirements

Your Flask backend should provide these endpoints:

- `POST /predict` - Image analysis endpoint (multipart form-data)
  - Accepts: `face_img` and/or `xray_img` files
  - Returns: JSON with analysis results
- `GET /health` - Health check endpoint
- `GET /static/*` - Static file serving for result images

### Request Format

```typescript
// FormData with one or both fields
face_img: File  // Face image file
xray_img: File  // X-ray image file
```

### Response Format

```typescript
{
  ok?: boolean;
  face_pred?: string;
  face_scores?: number[];
  face_img?: string;
  xray_pred?: string;
  yolo_vis?: string;
  found_labels?: string[];
  xray_img?: string;
  combined?: string;
}
```

## Testing Checklist

- [ ] Face image upload only
- [ ] X-ray image upload only
- [ ] Both images uploaded simultaneously
- [ ] Camera capture functionality
- [ ] Backend health check (`/health` endpoint)
- [ ] Production deployment with `?api=` parameter
- [ ] Mobile responsiveness
- [ ] Accessibility compliance
- [ ] Error handling (network failures, invalid files)
- [ ] Smooth animations and micro-interactions

## Technology Stack

- **Frontend**: Vite + React 18 + TypeScript
- **Styling**: TailwindCSS + shadcn/ui components
- **Animations**: Framer Motion
- **Routing**: React Router DOM
- **State Management**: React hooks
- **Image Processing**: Custom EXIF orientation correction
- **Notifications**: Sonner for toast messages
- **Icons**: Lucide React
- **Development**: ESLint + TypeScript strict mode

## File Structure

```
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── UploadArea.tsx   # File upload widget
│   │   ├── CameraCapture.tsx # Camera modal
│   │   ├── ResultCard.tsx   # Result display
│   │   └── MedicalDisclaimer.tsx
│   ├── pages/
│   │   ├── Index.tsx        # Upload page
│   │   └── Results.tsx      # Results page
│   ├── lib/
│   │   ├── api.ts          # API utilities
│   │   ├── image.ts        # Image processing
│   │   └── utils.ts        # General utilities
│   └── styles/
│       └── globals.css     # Global styles
├── vite.config.ts          # Vite configuration
└── tailwind.config.ts      # Tailwind configuration
```

## Security & Privacy

- All image processing happens client-side before upload
- No images are stored permanently on the frontend
- EXIF data is processed locally for orientation correction
- Medical disclaimer prominently displayed
- Educational/research use only

## Browser Support

- Modern browsers with ES2020 support
- Camera API requires HTTPS in production
- File drag-and-drop supported
- Responsive design for all viewport sizes

## Contributing

1. Follow TypeScript strict mode
2. Use provided ESLint configuration
3. Ensure accessibility compliance
4. Test on multiple devices and browsers
5. Update documentation for new features

## License

Educational and research use only. Not for medical diagnosis or treatment.

---

## Running Backend

### Local Development

### Local Development (when you have direct access to your machine)

If you're running this locally on your own machine:

1. **Start Flask backend** (in a separate terminal):
```bash
cd /path/to/your/backend/directory
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py  # Runs on http://127.0.0.1:5000
```

2. **Start frontend** (in this terminal):
```bash
npm install
npm run dev  # Runs on http://localhost:8080
```

3. **Visit**: http://localhost:8080

The frontend will automatically proxy API calls to your local Flask server at 127.0.0.1:5000.

### Sandbox Environments (StackBlitz, CodeSandbox, etc.)

**IMPORTANT**: If you're running this in StackBlitz, CodeSandbox, or similar browser-based environments, the proxy to `127.0.0.1:5000` will fail with `ECONNREFUSED` errors. This is expected because these sandboxes cannot access your local machine.

**Solution**: You need to expose your Flask backend publicly:

1. **Expose your backend publicly**:
   - **Option A - Using ngrok** (recommended for testing):
     ```bash
     # Install ngrok from https://ngrok.com/
     # In your backend directory, start Flask:
     python app.py
     
     # In another terminal:
     ngrok http 5000
     # Copy the https://abc123.ngrok.io URL
     ```
   
   - **Option B - Deploy to cloud**:
     Deploy your Flask backend to Render, Railway, Heroku, or similar platforms.

2. **Open the app with the API parameter**:
   ```
   https://stackblitz.com/~/your-project?api=https://YOUR-PUBLIC-BACKEND-URL
   ```
   
   Example:
   ```
   https://stackblitz.com/~/github-abc123?api=https://abc123.ngrok.io
   ```

3. **Verify backend is working**: 
   Open `https://YOUR-PUBLIC-BACKEND-URL/health` in a new browser tab. 
   It should return JSON: `{"status": "ok", "models": {...}}`

### Troubleshooting Connection Issues

**If you see "Backend Status: Unreachable" or proxy errors:**

1. **Check if you're in a sandbox**: Look at your URL. If it contains `stackblitz.com`, `codesandbox.io`, etc., you're in a sandbox and need to use the `?api=` parameter method above.

2. **Verify your backend is running**: 
   - Local: Visit `http://127.0.0.1:5000/health` in your browser
   - Public: Visit `https://your-backend-url/health` in your browser
   - Should return: `{"status": "ok", "models": {"face": true, "yolo": true}}`

3. **Use the "Set API URL" button**: Click the button next to "Backend Status" to manually configure your backend URL.

4. **Common solutions**:
   - **Local development**: Make sure Flask is running on port 5000
   - **Sandbox environments**: Use ngrok or deploy your backend, then add `?api=https://your-backend-url` to the frontend URL
   - **CORS issues**: Ensure your Flask backend has proper CORS headers configured

---

**Project by DHANUSH RAJA (21MIC0158)**