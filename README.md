# SatQuery AI - ISRO SIH26167

An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis.

## Architecture

* **Backend:** FastAPI (Python) orchestrating an agentic tool registry.
* **Agent Controller:** LLM-based query interpretation and task planning.
* **Tool Registry:** Abstracted tool interface (`RemoteSensingTool`) for models.
* **Frontend:** Next.js + Tailwind CSS providing a unified Analysis Workspace.
* **ML/RS Integration:** Hugging Face `transformers` (BLIP for VQA), `rasterio`/`numpy` for SAR/Optical fusion and bi-temporal change detection.

## Quick Start (Demo Mode)

The application is fully containerized. You do not need to configure any models manually; lightweight real models will be downloaded automatically on the first run.

```bash
# 1. Build and start the system
docker-compose up --build

# 2. Open the UI
# http://localhost:3000
```

*Note: The first backend request involving VQA will download the ~1GB BLIP model to the container. Subsequent requests will be fast.*

## Executing the Demos

1. **Demo A (Single Image VQA):** Upload an optical image, ask "Describe the land-cover".
2. **Demo B (Grounding):** Upload an optical image, ask "Highlight the water body".
3. **Demo C (Change Analysis):** Select "Bi-temporal Pair", upload two aligned images, ask "What changed between these dates?".
4. **Demo D (Optical + SAR):** Select "Optical + SAR Pair", upload both, ask "Analyze built-up areas using both modalities".
5. **Demo E (Agent Routing):** The Execution Trace UI demonstrates how the system routed your natural language query to the exact mathematical/ML tool.

## Offline Execution / Mocks

By default, the agent uses a lightweight heuristic routing engine if `GEMINI_API_KEY` is not set. To use the powerful LLM router:
`export GEMINI_API_KEY=your_key` before running docker-compose.

If you are completely offline and cannot download HF models, set `USE_MOCK_MODELS=true` in `docker-compose.yml` to use purely mock deterministic tools.

## Fine-Tuning Pipeline

A complete, runnable script for PEFT/LoRA adaptation on BigEarthNet is provided:
`python training/scripts/train_lora_bigearthnet.py`

## Benchmark Evaluation

Evaluation adapters for VRSBench, RSVQA, and CDVQA are located in `backend/evaluation/evaluate.py`.
