A local web application for batch color correction with intelligent clustering. Process multiple contexts simultaneously, cluster images by lighting conditions, apply corrections per cluster, and export corrected images at multiple resolutions.

## Features

- **Intelligent Clustering**: Automatically groups images by lighting conditions using corner analysis
- **Drag & Drop**: Move images between clusters for fine-tuned control
- **Auto-Correction**: One-click white balance and exposure correction per cluster
- **Manual Controls**: Fine-tune temperature, tint, exposure, contrast, saturation, and RGB gains
- **Batch Export**: Export at 3000px, 1500px, and 450px widths with overwrite or suffix options
- **Live Preview**: See corrections in real-time before exporting

## Architecture

- **Backend**: FastAPI (Python) - handles image processing, clustering, and export
- **Frontend**: React + Vite + Chakra UI - clean, responsive interface
- **State**: In-memory session storage with react-query for data fetching

## Project Structure

```
color_correct/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── routes/           # API endpoints
│   │   └── services/         # Business logic
│   └── requirements.txt
├── frontend/          # React frontend
│   ├── src/
│   │   ├── pages/            # Main pages
│   │   ├── components/       # UI components
│   │   ├── api/              # API client
│   │   └── state/            # State management
│   └── package.json
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to backend directory:
```bash
cd color_correct/backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd color_correct/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Usage

### 1. Session Setup

- Add one or more context directory paths
  - Example: `D:\ararat\data\files\N\38\478020\4419550\1`
- Configure options:
  - **Raw Mode**: Process from RAW files (slower, higher quality)
  - **Overwrite**: Replace existing files or add `-color_correct` suffix
  - **Sensitivity**: Control clustering fineness (0.1-5.0, default 1.0)
- Click "Start Color Correction"

### 2. Clustering

The app automatically:
- Scans all contexts for images
- Clusters images by lighting conditions
- Groups similar images together

### 3. Review & Correct

- View clusters in columns
- **Drag images** between clusters to rearrange
- **Click "Auto-Correct"** on a cluster to estimate white balance
- **Select an image** to open the correction panel
- **Adjust sliders** for fine-tuning:
  - Temperature & Tint
  - Exposure (EV)
  - Contrast & Saturation
  - RGB Gains
- **Apply to Cluster** to save corrections

### 4. Export

- Review the summary in the export bar
- Click "Export All" to process all images
- Track progress in real-time
- View detailed export summary when complete
- Creates .ascota metadata folders

## File Discovery

The app expects this structure:

```
{context_path}/
└── finds/
    └── individual/
        └── {find_number}/
            └── photos/
                ├── 1.CR3           # RAW file (required)
                ├── 1.jpg           # 450px render
                ├── 1-1500.jpg      # 1500px render
                └── 1-3000.jpg      # 3000px render
```

## Export Behavior

### Overwrite Mode (ON)
- Replaces: `1.jpg`, `1-1500.jpg`, `1-3000.jpg`
- Uses original file format (or .jpg default)

### Overwrite Mode (OFF)
- Creates: `1-color_correct.jpg`, `1-1500-color_correct.jpg`, `1-3000-color_correct.jpg`
- Always uses .jpg format

All images preserve aspect ratio and are resized by width.

## Troubleshooting

### Backend won't start
- Ensure Python 3.8+ is installed
- Check that all dependencies installed successfully
- Verify port 8000 is not in use

### Frontend won't start
- Ensure Node.js 16+ is installed
- Run `npm install` again if needed
- Check that port 5173 is available

### Images not loading
- Verify context paths are correct
- Check that images follow the expected directory structure
- Ensure backend can access the file paths

## API Documentation

Once the backend is running, visit:
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

