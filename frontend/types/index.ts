export type ImageMetadata = {
    filename: string;
    format: string;
    width: number;
    height: number;
    bands: number;
    modality: string;
    crs?: string;
    transform?: number[];
    bounds?: number[];
    georeferenced: boolean;
};

export type ToolResult = {
    tool_name: string;
    model_name: string;
    model_type: "real" | "mock";
    status: "success" | "failure";
    text?: string;
    structured_data?: any;
    evidence?: any;
    confidence?: number;
    metadata?: any;
    execution_time_ms?: number;
};

export type ExecutionStep = {
    step_name: string;
    description: string;
    status: "pending" | "running" | "success" | "failure";
    tool_results?: ToolResult[];
    error_message?: string;
    timestamp: string;
};

export type AnalysisTask = {
    task_type: string;
};

export type ExecutionTrace = {
    trace_id: string;
    task: AnalysisTask;
    steps: ExecutionStep[];
    final_result?: string;
    final_confidence?: number;
    final_evidence?: any;
    completed_at?: string;
};
