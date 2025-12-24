# Preprocess Pipeline Application

A local web application for 3-stage preprocessing pipeline: card detection, background segmentation, and scale calculation. Processes images through sequential stages with user feedback, saving all results to `.ascota` metadata folders.

## Features

- **Stage 1: Card Detection** - Automatically detects color reference cards with manual editing capability
- **Stage 2: Background Segmentation** - Generates masks with interactive brush painting
- **Stage 3: Scale Calculation** - Calculates pixels per cm and surface area with manual point adjustment

## Architecture

- **Backend**: FastAPI (Python) - handles image processing, detection, segmentation, and scale calculation
- **Frontend**: React + Vite + Chakra UI - clean, responsive interface
- **State**: In-memory session storage with react-query for data fetching

## Project Structure

```
preprocess/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── routes/           # API endpoints
│   │   │   ├── sessions.py  # Session management
│   │   │   ├── stage1_cards.py  # Card detection endpoints
│   │   │   ├── stage2_mask.py   # Segmentation endpoints
│   │   │   └── stage3_scale.py  # Scale calculation endpoints
│   │   └── services/         # Business logic
│   │       ├── scanner.py    # Image discovery
│   │       ├── card_detection.py  # Card detection service
│   │       ├── segmentation.py    # Mask generation & editing
│   │       ├── scale_calc.py      # Scale calculation service
│   │       ├── metadata.py        # .ascota metadata management
│   │       └── models.py           # Pydantic models
│   └── requirements.txt
├── frontend/          # React frontend
│   ├── src/
│   │   ├── pages/            # Main pages
│   │   │   ├── SessionSetup.tsx
│   │   │   ├── Stage1Cards.tsx
│   │   │   ├── Stage2Mask.tsx
│   │   │   └── Stage3Scale.tsx
│   │   ├── components/       # UI components
│   │   │   ├── CardEditor.tsx    # Interactive card point editor
│   │   │   ├── MaskPainter.tsx   # Brush tool for mask editing
│   │   │   └── ScaleEditor.tsx   # 8-hybrid point editor
│   │   ├── api/              # API client
│   │   └── state/            # State management
│   └── package.json
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to backend directory:
```bash
cd preprocess/backend
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
uvicorn app.main:app --reload --port 8001
```

The API will be available at `http://localhost:8001`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd preprocess/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5174`

## Usage

### 1. Session Setup

- Add one or more context directory paths
  - Example: `D:\ararat\data\files\N\38\478020\4419550\1`
- Click "Start Preprocessing"

### 2. Stage 1: Card Detection

- Click "Detect Cards" to run automatic detection on all images
- View results in grid: each image shows number of detected cards and card types
- Click an image to open the Card Editor:
  - Drag the 4 corner points of each card to adjust
  - Add new cards using the "Add Card" button
  - Delete cards using the "Delete" button
  - Cards are color-coded: Green (24-color), Blue (8-hybrid), Red (checker)
- Click "Next Stage" to save and proceed

### 3. Stage 2: Background Segmentation

- Click "Generate Masks" to create masks using card coordinates from Stage 1
- View results in grid: each image shows mask status
- Click an image to open the Mask Painter:
  - Use "Paint In" to mark foreground areas
  - Use "Paint Out" to mark background areas
  - Adjust brush size with the slider
  - Toggle mask visibility
- Click "Next Stage" to save and proceed

### 4. Stage 3: Scale Calculation

- Click "Calculate Scale" to compute pixels per cm and surface area
- View results in cards showing:
  - Pixels per cm
  - Surface area in cm²
  - Calculation method used
- For 8-hybrid cards: Click the result card to edit circle centers
  - Drag the three circle centers to adjust positions
  - Click "Recalculate & Save" to update scale
- Click "Save Results" to finalize

## File Discovery

The app expects this structure:

```
{context_path}/
└── finds/
    └── individual/
        └── {find_number}/
            └── photos/
                ├── 1.CR3           # RAW file
                ├── 1-3000.jpg      # 3000px render (required)
                ├── 1-1500.jpg      # 1500px render (optional)
                └── 1.jpg           # 450px render (optional)
```

## Metadata Storage

All results are saved to `.ascota` folders in each find directory:

```
{find_path}/.ascota/
├── preprocess.json      # Stage 1, 2, and 3 metadata
└── masks/
    └── {image_id}_mask.png  # Binary mask files
```

### Metadata Format

**preprocess.json**:
```json
{
  "stage1": {
    "images": {
      "1": {
        "image_size": [3000, 2000],
        "cards": [
          {
            "card_id": "card_0",
            "card_type": "24_color_card",
            "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "confidence": 0.95
          }
        ]
      }
    },
    "timestamp": "2024-01-01T00:00:00"
  },
  "stage2": {
    "masks": {
      "1": {
        "mask_path": "masks/1_mask.png"
      }
    },
    "timestamp": "2024-01-01T00:00:00"
  },
  "stage3": {
    "images": {
      "1": {
        "pixels_per_cm": 120,
        "surface_area_cm2": 45.67,
        "method": "checker_card",
        "card_used": "card_0"
      }
    },
    "timestamp": "2024-01-01T00:00:00"
  }
}
```

## Troubleshooting

### Backend won't start
- Ensure Python 3.8+ is installed
- Check that all dependencies installed successfully
- Verify port 8001 is not in use

### Frontend won't start
- Ensure Node.js 16+ is installed
- Run `npm install` again if needed
- Check that port 5174 is available

### Images not loading
- Verify context paths are correct
- Check that images follow the expected directory structure
- Ensure -3000 pixel images exist for all images
- Ensure backend can access the file paths

### Card detection fails
- Check that images contain visible color reference cards
- Try adjusting card points manually in the editor
- Verify image quality and lighting conditions

### Mask generation fails
- Ensure Stage 1 is completed first
- Check that card coordinates are valid
- Verify transformers library is installed for RMBG

### Scale calculation fails
- Ensure a checker_card or 8_hybrid_card was detected in Stage 1
- For 8-hybrid cards, manually adjust circle centers if automatic detection fails
- Verify card crop is clear and well-lit

## API Documentation

Once the backend is running, visit:
- API docs: `http://localhost:8001/docs`
- Health check: `http://localhost:8001/health`

