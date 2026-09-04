import { ExecutionTrace } from "../types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const analyzeQuery = async (query: string, inputType: string, files: File[]): Promise<ExecutionTrace> => {
    const formData = new FormData();
    formData.append("query", query);
    formData.append("input_type", inputType);
    
    files.forEach(file => {
        formData.append("files", file);
    });

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "An error occurred during analysis");
    }

    return response.json();
};
