import json
from backend.api.schemas.domain import ExecutionTrace

class ReportService:
    @staticmethod
    def generate_report(trace: ExecutionTrace) -> str:
        """Generates a downloadable markdown report."""
        report = f"# SatQuery AI Analysis Report\n\n"
        report += f"**Trace ID:** {trace.trace_id}\n"
        report += f"**Completed At:** {trace.completed_at}\n\n"
        report += f"## Task\n{trace.task.task_type}\n\n"
        report += f"## Result\n{trace.final_result}\n\n"
        report += f"**Confidence:** {trace.final_confidence}\n\n"
        report += f"## Execution Trace\n"
        
        for step in trace.steps:
            report += f"- **{step.step_name}** ({step.status}): {step.description}\n"
            
        return report
