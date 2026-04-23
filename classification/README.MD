# Classification Pipeline Application

Local web application for running pottery classification on preprocessed
contexts. The app supports **Pottery gate**, **Type**, **Decoration**, **Color**,
and **Texture** workflows with manual review and export.

## Features

- **Context validation**: Session setup checks context preprocess status before run.
- **Multiple classification modes**:
  - Pottery vs non-pottery gate
  - Type (body/base/rim/appendage + optional appendage subtype)
  - Decoration
  - Color clustering/classification workflow
  - Texture clustering/classification workflow
- **Interactive review**: grouped results, confidence sorting, manual relabel,
  and undo for each workflow.
- **Export**: writes classifications into each find's
  `.ascota/classification.json` while preserving other keys.
- **Image serving**: backend serves images by session/item for fast review UI.

## Architecture

- **Backend**: FastAPI on port `8002`.
- **Frontend**: React + Vite + Chakra UI on port `5175`.
- **Core models**: uses `ascota_classification` package from repository `src/`.
- **Session data**: temporary run state under `classification/.ascota/`
  (cleared after export).

## Project Structure

```
classification/
├── backend/
│   └── app/
│       ├── main.py
│       ├── routes/
│       │   ├── sessions.py
│       │   ├── pottery_classification.py
│       │   ├── type_classification.py
│       │   ├── decoration_classification.py
│       │   ├── color_classification.py
│       │   ├── texture_classification.py
│       │   └── images.py
│       └── services/
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── SessionSetup.tsx
│       │   ├── PotteryClassification.tsx
│       │   ├── TypeClassification.tsx
│       │   ├── DecorationClassification.tsx
│       │   ├── ColorClassification.tsx
│       │   └── TextureClassification.tsx
│       ├── api/
│       └── state/
└── README.MD
```

## Setup

### Backend

From the repository root:

```bash
cd classification/backend
uvicorn app.main:app --reload --port 8002
```

API: `http://localhost:8002`

### Frontend

```bash
cd classification/frontend
npm install
npm run dev
```

App: `http://localhost:5175` (frontend proxies `/api` to backend `8002`).

### Optional: Appendage subtype classification

When appendage subtype is enabled, appendage images are sent to Azure OpenAI.
Set these env vars (shell or repo-root `.env`):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

Without them, appendages remain labeled as `"appendage"`.

## Usage

1. Create or load a session in **Session Setup**.
2. Select a classification mode and run inference/clustering.
3. Review grouped results, apply manual label changes as needed.
4. Use **Undo** to revert the latest manual action.
5. Export to persist output into per-find `.ascota/classification.json`.

## Output Format

Export writes/merges into each find's `.ascota/classification.json`.
Example shape:

```json
{
  "type": {
    "1": { "label": "body", "confidence": 1.0 }
  },
  "decoration": {
    "1": { "label": "incised", "confidence": 0.91 }
  }
}
```

Temporary run state is stored as `classification/.ascota/run_{session_id}.json`
and cleaned on successful export.
