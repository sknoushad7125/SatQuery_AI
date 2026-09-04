import uuid
import json
import base64
import io
from PIL import Image
import time
from datetime import datetime
from typing import List
from backend.api.schemas.domain import (
    AnalysisInput, AnalysisQuery, ToolResult, ExecutionTrace, ExecutionStep, AnalysisTask
)
from backend.agents.router import QueryRouter
from backend.tools.vqa.real_vqa import RealVQATool
from backend.tools.captioning.real_caption import RealCaptioningTool
from backend.tools.sar.real_sar_classification import RealSARClassificationTool
from backend.tools.change.real_change import RealSemanticChangeTool
from backend.tools.optical_sar.real_fusion import RealDecisionFusionTool
from backend.services.llm_provider import get_llm_provider

class AgentController:
    def __init__(self):
        self.router = QueryRouter()
        self.vqa_tool = RealVQATool()
        self.caption_tool = RealCaptioningTool()
        self.sar_tool = RealSARClassificationTool()
        self.change_tool = RealSemanticChangeTool()
        self.fusion_tool = RealDecisionFusionTool()
        self.llm = get_llm_provider()

    def _truncate_large_fields(self, data):
        if isinstance(data, dict):
            return {k: self._truncate_large_fields(v) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) > 50:
                return f"<List truncated, length={len(data)}>"
            return [self._truncate_large_fields(v) for v in data]
        return data

    def process(self, query: AnalysisQuery, input_data: AnalysisInput, image_paths: List[str]) -> ExecutionTrace:
        trace_id = str(uuid.uuid4())
        input_type = getattr(input_data, 'input_type', None)

        steps = []

        # 1. Input Validation & 2. Intent / Routing
        val_step = ExecutionStep(
            step_name="Input Validation",
            description="Validating input modalities and query compatibility.",
            status="running"
        )
        route, reason = self.router.route(query.text, input_data.images, input_type)

        task_mapping = {
            "vqa": "single_vqa",
            "change_detection": "bi_temporal_change",
            "optical_sar": "optical_sar_analysis",
            "captioning": "captioning",
            "sar_classification": "single_sar_classification",
            "validation_failed": "unsupported",
            "unsupported": "unsupported"
        }
        task_type_str = task_mapping.get(route, "unsupported")
        task = AnalysisTask(task_type=task_type_str)

        if route == "validation_failed":
            val_step.status = "failure"
            val_step.error_message = reason
            steps.append(val_step)
            return ExecutionTrace(
                trace_id=trace_id,
                task=task,
                steps=steps,
                final_result=reason,
                completed_at=datetime.utcnow()
            )

        # Geospatial bounds check for temporal pair
        if route == "change_detection" and len(input_data.images) == 2:
            img1 = input_data.images[0]
            img2 = input_data.images[1]
            if img1.bounds and img2.bounds:
                b1 = img1.bounds
                b2 = img2.bounds
                # check overlap (minx, miny, maxx, maxy)
                # overlap = not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])
                overlap = not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])
                if not overlap:
                    val_step.error_message = "WARNING: The two images do not appear to geographically overlap based on their metadata."

        val_step.status = "success"
        steps.append(val_step)

        routing_step = ExecutionStep(
            step_name="Intent / Routing",
            description=f"Routed to {route} (Reason: {reason})",
            status="success"
        )
        steps.append(routing_step)

        # 3. Model Execution
        exec_step = ExecutionStep(
            step_name="Specialist Model Execution",
            description=f"Executing routed tool: {route}",
            status="running"
        )

        if route == "vqa":
            tool_res = self.vqa_tool.run(image_paths, query.text)
        elif route == "captioning":
            tool_res = self.caption_tool.run(image_paths, query.text)
        elif route == "sar_classification":
            tool_res = self.sar_tool.run(image_paths, query.text)
        elif route == "change_detection":
            tool_res = self.change_tool.run(image_paths, query.text)
        elif route == "optical_sar":
            tool_res = self.fusion_tool.run(image_paths, query.text)
        else:
            tool_res = ToolResult(
                tool_name="QueryRouter",
                model_name="QueryRouter",
                model_type="real",
                status="failure",
                error_message="Unsupported query or incompatible image modalities provided."
            )

        exec_step.tool_results = [tool_res]
        exec_step.status = "success" if tool_res.status == "success" else "failure"
        steps.append(exec_step)

        if tool_res.status == "failure":
            return ExecutionTrace(
                trace_id=trace_id,
                task=task,
                steps=steps,
                final_result=tool_res.error_message,
                completed_at=datetime.utcnow()
            )

        # 4. Evidence Extraction
        ev_step = ExecutionStep(
            step_name="Evidence Extraction",
            description="Extracting structured evidence and metadata from model outputs",
            status="success"
        )
        steps.append(ev_step)

        # 5. Gemini Synthesis Step
        synth_step = ExecutionStep(
            step_name="Gemini Semantic Synthesis",
            description="Synthesizing final detailed answer from model outputs",
            status="running"
        )

        safe_evidence = self._truncate_large_fields(tool_res.evidence) if tool_res.evidence else None
        safe_structured = self._truncate_large_fields(tool_res.structured_data) if tool_res.structured_data else None

        tool_data = {
            "tool_name": tool_res.tool_name,
            "status": tool_res.status,
            "raw_text": tool_res.text,
            "confidence": tool_res.confidence,
            "evidence": safe_evidence,
            "structured_data": safe_structured,
            "error_message": getattr(tool_res, 'error_message', None)
        }

        tool_outputs_json = json.dumps([tool_data], indent=2)

        format_instructions = ""
        if route == "vqa":
            format_instructions = "For VQA, use exactly this structure:\n\nANSWER\n- [Direct answer]\n\nOBSERVATIONS\n- [2-4 sentences explaining the visual evidence]\n\nMODEL EVIDENCE\n- [specialist task]\n- [relevant structured output]\n- [confidence if available]\n\nINTERPRETATION\n- [concise explanation]"
        elif route == "captioning":
            format_instructions = "For captioning, use exactly this structure:\n\nSCENE DESCRIPTION\n- [2-4 sentence remote-sensing description]\n\nKEY OBSERVATIONS\n- [main elements]\n\nMODEL\n- [model name]"
        elif route == "change_detection":
            format_instructions = "For change detection, use exactly this structure:\n\nCHANGE SUMMARY\n- [detailed explanation]\n\nQUANTITATIVE RESULT\n- [change percentage]\n- [confidence]\n- [number of regions]\n\nSPATIAL EVIDENCE\n- [explain detected regions]\n\nBEFORE -> AFTER INTERPRETATION\n- [only if visual evidence is provided]"
        elif route == "optical_sar":
            format_instructions = "For optical-SAR, use exactly this structure:\n\nCROSS-MODAL ANALYSIS\n- [detailed explanation]\n\nOPTICAL EVIDENCE\n- [optical contribution]\n\nSAR EVIDENCE\n- [SAR contribution]\n\nFUSION RESULT\n- [quantitative predicted classes]\n\nUNCERTAINTY\n- [explain limitations]"
        elif route == "sar_classification":
            format_instructions = "For SAR classification, use exactly this structure:\n\nCLASSIFICATION RESULT\n- [predicted classes and probabilities]\n\nINTERPRETATION\n- [short explanation]"

        prompt = f"""You are the synthesis agent for SatQuery AI.
Your job is to read the structured outputs from remote sensing ML tools and formulate a detailed, grounded final answer to the user's query.

User Query: "{query.text}"
Task Classification: {route}

---
Tool Outputs / Evidence (JSON array of executed tool results):
{tool_outputs_json}
---

CRITICAL INSTRUCTIONS:
1. ONLY use the evidence returned by the tools above. Do not invent observations, objects, or numerical values.
2. Examine the "status" field for each tool. If a tool's status is "failure", you MUST NOT treat it as successful. Do NOT invent an observation from a failed tool.
3. Use successful tool evidence (e.g. text, percentage_change, class_percentages, etc.) when available.
4. Distinguish model predictions from absolute certainty. If the confidence is low, mention the uncertainty.
5. Answer the user's actual question directly and concisely.
6. Provide ONLY the final structured answer text. Do not output JSON or wrapper text.

{format_instructions}
"""
        if route == "change_detection" and tool_res.evidence and "change_crops" in tool_res.evidence:
            change_crops = tool_res.evidence["change_crops"]
            if change_crops:
                multimodal_payload = [
                    f"You are the synthesis agent for SatQuery AI.\nUser Query: '{query.text}'\n\n",
                    f"INSTRUCTIONS:\n",
                    f"You are evaluating cropped images of regions where a specialized model detected changes.\n",
                    f"The specialized change detector is trained on LEVIR-CD (building-change scenarios). The spatial regions were explicitly selected by this detector.\n",
                    f"You must visually compare the BEFORE vs AFTER crops for each region to describe what changed.\n",
                    f"For change description queries, produce a concise natural-language description of what is visibly different.\n",
                    f"For change-based VQA queries, directly answer the user's question using the visual evidence.\n",
                    f"CRITICAL: Do NOT describe every detected region as definitely a building merely because the model was trained on LEVIR-CD. Describe only what is visually supported by the crops.\n",
                    f"CRITICAL: If the query asks about unsupported semantic classes (e.g. roads, water, vegetation) that the underlying detector is not validated for, explicitly state this limitation instead of guessing.\n",
                    f"Do not invent changes not visible in the supplied images.\n\n",
                    f"{format_instructions}\n\n",
                    f"Here are the cropped regions:\n"
                ]
                for crop in change_crops:
                    reg_id = crop["region_id"]
                    try:
                        img_before = Image.open(io.BytesIO(base64.b64decode(crop["before"])))
                        img_after = Image.open(io.BytesIO(base64.b64decode(crop["after"])))
                        multimodal_payload.extend([
                            f"Region {reg_id} Before:", img_before,
                            f"Region {reg_id} After:", img_after
                        ])
                    except Exception:
                        pass
                prompt = multimodal_payload
            else:
                prompt += "\nNOTE: The specialized detector found no meaningful changed regions."

        try:
            synthesized_text = self.llm.generate(prompt).strip()
            synth_step.status = "success"
            steps.append(synth_step)
        except Exception as e:
            synth_step.status = "failure"
            synth_step.error_message = "LLM synthesis encountered an error (internal API failure)."
            steps.append(synth_step)

            fallback_step = ExecutionStep(
                step_name="Fallback Synthesis",
                description="Using raw specialist output due to LLM synthesis failure",
                status="success"
            )
            synthesized_text = tool_res.text if tool_res.text else f"LLM synthesis failed. Raw tool error: {tool_res.error_message}"
            steps.append(fallback_step)

        # Optional validation warning appended if any
        if getattr(val_step, 'error_message', None):
            if "WARNING:" in val_step.error_message:
                synthesized_text = val_step.error_message + "\n\n" + synthesized_text

        return ExecutionTrace(
            trace_id=trace_id,
            task=task,
            steps=steps,
            final_result=synthesized_text,
            final_confidence=tool_res.confidence,
            final_evidence=tool_res.evidence,
            completed_at=datetime.utcnow()
        )
