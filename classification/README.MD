# Classification Pipeline Application

A local web application for running pottery classification on preprocessed contexts. Supports **Type** classification (body, base, rim, appendage, appendage subtypes). Decoration, Color, and Texture are planned (WIP).

## Features

- **Single context**: Add one context directory; continue only if it has been preprocessed (`.ascota/context_status.json` with `preprocess_status: true`).
- **Type classification**: Run the type pipeline on all finds with masks; optional appendage subtype (Azure OpenAI); resolution 1000 / 1500 / 3000 px.
- **Results by category**: Images grouped by predicted label; sort by confidence; click to change classification; undo; export to each find's `.ascota/classification.json`.

## Architecture

- **Backend**: FastAPI (Python), port 8002. Uses `ascota_classification.type` from repo `src/`.
- **Frontend**: React + Vite + Chakra UI, port 5175.
- **Temp data**: Stored in `classification/.ascota/` (hidden on Windows); deleted on export.

## Project structure

```
classification/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── sessions.py
│   │   │   ├── type_classification.py
│   │   │   └── images.py
│   │   └── services/
│   │       ├── metadata.py
│   │       ├── scanner.py
│   │       ├── image_utils.py
│   │       ├── classification_store.py
│   │       ├── export.py
│   │       └── models.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── SessionSetup.tsx
│   │   │   └── TypeClassification.tsx
│   │   ├── api/
│   │   ├── state/
│   │   └── ...
│   └── package.json
├── .ascota/          # Temp run state (created at runtime, cleared on export)
└── README.MD
```

## Setup

### Backend

From the **repository root** (so `ascota_classification` and dependencies are available):

1. Navigate to backend:
   ```bash
   cd classification/backend
   ```

2. Run the backend (PYTHONPATH must include repo root):
   ```bash
   uvicorn app.main:app --reload --port 8002
   ```

API: `http://localhost:8002`

### Frontend

1. Navigate to frontend:
   ```bash
   cd classification/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start dev server:
   ```bash
   npm run dev
   ```

App: `http://localhost:5175` (proxies `/api` to backend 8002).

### Appendage subtype (Stage 3)

When **Enable Appendage subtype classification** is on, objects classified as *appendage* are sent to Azure OpenAI GPT-4o for subtype labels (lid, rim-handle, spout, rounded, body-decorated, tile). The backend loads `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` from the environment; it also loads a `.env` file from the **repository root** at startup (if `python-dotenv` is installed). Set those variables in your shell or in a `.env` at the repo root so Stage 3 can run. Without them, appendages stay labelled as "appendage".

## Usage

1. **Session setup**: Enter a context directory path and click **Check**. Continue is enabled only when the context is preprocessed. Select classification type (only **Type** is available; others WIP) and click **Continue**.
2. **Type page**: Optionally enable **Appendage subtype classification** (tooltip explains). Choose resolution (1000 / 1500 / 3000 px, default 1500). Click **Run classification**; a loading screen appears while the pipeline runs.
3. **Results**: Images are grouped by label. Use the sort dropdown for confidence order. Click an image to open a panel and change its classification (confidence set to 100%). Use **Undo** to revert the last edit.
4. **Export**: Click **Export** to write results to each find's `.ascota/classification.json` (key `type`). Temp data in `classification/.ascota/` is removed and the session is cleared.

## Data formats

- **classification.json** (per find):  
  `{ "type": { "1": { "label": "body", "confidence": 1.0 }, "2": { "label": "rim", "confidence": 0.87 } } }`  
  Export merges into existing JSON so other classification kinds are not overwritten.

- **Temp run state**: `classification/.ascota/run_{session_id}.json`; deleted on export.
