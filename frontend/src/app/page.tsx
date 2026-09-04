
"use client";

import { useState } from "react";
import { Satellite, Brain, Image as ImageIcon, Upload, MessageSquare, Activity, CheckCircle, AlertTriangle, Layers, Info, Map } from "lucide-react";

export default function Home() {
    const [query, setQuery] = useState("");
    const [inputType, setInputType] = useState("single");
    const [files, setFiles] = useState<File[]>([]);
    const [fileBefore, setFileBefore] = useState<File | null>(null);
    const [fileAfter, setFileAfter] = useState<File | null>(null);
    const [fileOpt, setFileOpt] = useState<File | null>(null);
    const [fileSar, setFileSar] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [trace, setTrace] = useState<any>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
        }
    };

    const handleAnalyze = async () => {
        let submissionFiles: File[] = [];
        if (inputType === "single") {
            submissionFiles = files;
        } else if (inputType === "temporal_pair") {
            if (fileBefore && fileAfter) submissionFiles = [fileBefore, fileAfter];
        } else if (inputType === "optical_sar_pair") {
            if (fileOpt && fileSar) submissionFiles = [fileOpt, fileSar];
        }

        if (submissionFiles.length === 0) {
            setError("Please upload all required satellite imagery first.");
            return;
        }
        if (!query.trim()) {
            setError("Please enter a natural language query.");
            return;
        }
        setFiles(submissionFiles); // Keep files state updated for mask rendering
        setLoading(true);
        setError("");
        setTrace(null);

        const formData = new FormData();
        formData.append("query", query);
        formData.append("input_type", inputType);
        submissionFiles.forEach(f => formData.append("files", f));

        try {
            const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}));
                throw new Error(errorData.detail || "Analysis failed due to a server error.");
            }
            const data = await res.json();
            setTrace(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    let maskDataUrl = null;
    if (trace?.final_evidence?.change_mask_b64) {
        maskDataUrl = `data:image/png;base64,${trace.final_evidence.change_mask_b64}`;
    } else if (trace?.final_evidence?.prediction_mask_b64) {
        maskDataUrl = `data:image/png;base64,${trace.final_evidence.prediction_mask_b64}`;
    }

    const getInputHint = () => {
        if (inputType === "single") return "Expected: 1 optical/multispectral image";
        if (inputType === "temporal_pair") return "Expected: 2 images of the same area at different times (Before & After)";
        return "Expected: 1 optical image + 1 SAR image";
    };

    const getExamples = () => {
        if (inputType === "single") return [
            "Are there any roads in this image?",
            "Describe this satellite image.",
            "What land cover is in this SAR image?",
            "Classify this SAR image."
        ];
        if (inputType === "temporal_pair") return ["What changed between these two images?", "Show the new construction areas.", "Is there any deforestation?"];
        return ["Segment the land cover using optical and SAR.", "Map the vegetation and water.", "Identify built-up areas."];
    };

    const getImageLabel = (idx: number, type: string) => {
        if (type === "temporal_pair") return idx === 0 ? "Before (T1)" : "After (T2)";
        if (type === "optical_sar_pair") return idx === 0 ? "Optical (Sentinel-2)" : "SAR (Sentinel-1)";
        return "Input Image";
    };

    const renderImages = () => {
        return (
            <div className="grid grid-cols-2 gap-4">
                {files.map((file, idx) => {
                    let src = URL.createObjectURL(file);
                    let isTif = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');

                    if (trace?.final_evidence?.input_thumbnails_b64?.[idx]) {
                        src = `data:image/png;base64,${trace.final_evidence.input_thumbnails_b64[idx]}`;
                        isTif = false;
                    }

                    return (
                        <div key={idx} className="flex flex-col bg-gray-50 rounded-lg p-2 border border-gray-100 shadow-sm">
                            <span className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide flex items-center gap-1">
                                <ImageIcon size={14} />
                                {getImageLabel(idx, trace?.task?.task_type || inputType)}
                            </span>
                            <div className="rounded overflow-hidden bg-gray-200 aspect-square flex items-center justify-center relative group">
                                {isTif ? (
                                    <div className="text-gray-400 text-xs text-center p-2">
                                        <Map size={32} className="mx-auto mb-2 opacity-50" />
                                        TIFF Image<br/><span className="text-[10px] truncate block max-w-[120px]">{file.name}</span>
                                    </div>
                                ) : (
                                    <img src={src} alt={`Input ${idx + 1}`} className="w-full h-full object-cover" />
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-200">
            {/* HEADER */}
            <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="bg-blue-600 text-white p-2 rounded-lg shadow-sm">
                            <Satellite size={24} />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-slate-900 leading-tight">SatQuery AI</h1>
                            <p className="text-xs text-slate-500 font-medium">Multimodal Remote Sensing Agent</p>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-100">
                            <Brain size={14} /> AI Agent
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-100">
                            <Layers size={14} /> Modalities Active
                        </span>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* LEFT COLUMN - INPUT WORKSPACE */}
                <section className="lg:col-span-5 flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <div className="flex items-center gap-2 mb-6 border-b border-slate-100 pb-4">
                            <Upload className="text-blue-600" size={20} />
                            <h2 className="text-lg font-semibold text-slate-800">Analysis Workspace</h2>
                        </div>

                        <div className="space-y-5">
                            <div>
                                <label className="block text-sm font-semibold text-slate-700 mb-1.5">Analysis Mode</label>
                                <select
                                    value={inputType}
                                    onChange={(e) => setInputType(e.target.value)}
                                    className="w-full bg-slate-50 border-slate-200 text-slate-800 rounded-lg shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm py-2.5 px-3"
                                >
                                    <option value="single">Single Image (VQA)</option>
                                    <option value="temporal_pair">Bi-temporal Pair (Change Detection)</option>
                                    <option value="optical_sar_pair">Optical + SAR Pair (Segmentation)</option>
                                </select>
                            </div>

                            <div>
                                {inputType === "single" && (
                                    <>
                                        <label className="block text-sm font-semibold text-slate-700 mb-1.5">Upload Image</label>
                                        <div className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-4 text-center hover:bg-slate-100 transition-colors relative">
                                            <input
                                                type="file"
                                                onChange={(e) => e.target.files && setFiles([e.target.files[0]])}
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                            />
                                            <div className="flex flex-col items-center gap-1 pointer-events-none">
                                                <ImageIcon size={24} className="text-slate-400 mb-1" />
                                                <span className="text-sm font-medium text-slate-600">
                                                    {files.length > 0 ? `${files[0].name} selected` : "Click or drag to upload"}
                                                </span>
                                            </div>
                                        </div>
                                    </>
                                )}
                                {inputType === "temporal_pair" && (
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-700 mb-1.5 uppercase tracking-wide">BEFORE IMAGE</label>
                                            <div className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-4 text-center hover:bg-slate-100 transition-colors relative h-24 flex items-center justify-center">
                                                <input
                                                    type="file"
                                                    onChange={(e) => setFileBefore(e.target.files ? e.target.files[0] : null)}
                                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                                />
                                                <div className="flex flex-col items-center gap-1 pointer-events-none">
                                                    <ImageIcon size={20} className="text-slate-400 mb-1" />
                                                    <span className="text-xs font-medium text-slate-600 truncate max-w-full px-2">
                                                        {fileBefore ? fileBefore.name : "Select Before"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-700 mb-1.5 uppercase tracking-wide">AFTER IMAGE</label>
                                            <div className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-4 text-center hover:bg-slate-100 transition-colors relative h-24 flex items-center justify-center">
                                                <input
                                                    type="file"
                                                    onChange={(e) => setFileAfter(e.target.files ? e.target.files[0] : null)}
                                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                                />
                                                <div className="flex flex-col items-center gap-1 pointer-events-none">
                                                    <ImageIcon size={20} className="text-slate-400 mb-1" />
                                                    <span className="text-xs font-medium text-slate-600 truncate max-w-full px-2">
                                                        {fileAfter ? fileAfter.name : "Select After"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-span-2 mt-1">
                                            <span className="text-[11px] text-amber-600 flex items-center gap-1 bg-amber-50 p-2 rounded">
                                                <AlertTriangle size={12} /> The two images must represent the same geographic area at different times.
                                            </span>
                                        </div>
                                    </div>
                                )}
                                {inputType === "optical_sar_pair" && (
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-700 mb-1.5 uppercase tracking-wide">OPTICAL / MULTISPECTRAL IMAGE</label>
                                            <div className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-4 text-center hover:bg-slate-100 transition-colors relative h-24 flex items-center justify-center">
                                                <input
                                                    type="file"
                                                    onChange={(e) => setFileOpt(e.target.files ? e.target.files[0] : null)}
                                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                                />
                                                <div className="flex flex-col items-center gap-1 pointer-events-none">
                                                    <ImageIcon size={20} className="text-slate-400 mb-1" />
                                                    <span className="text-xs font-medium text-slate-600 truncate max-w-full px-2">
                                                        {fileOpt ? fileOpt.name : "Select Optical"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-700 mb-1.5 uppercase tracking-wide">SAR IMAGE</label>
                                            <div className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-4 text-center hover:bg-slate-100 transition-colors relative h-24 flex items-center justify-center">
                                                <input
                                                    type="file"
                                                    onChange={(e) => setFileSar(e.target.files ? e.target.files[0] : null)}
                                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                                />
                                                <div className="flex flex-col items-center gap-1 pointer-events-none">
                                                    <ImageIcon size={20} className="text-slate-400 mb-1" />
                                                    <span className="text-xs font-medium text-slate-600 truncate max-w-full px-2">
                                                        {fileSar ? fileSar.name : "Select SAR"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div>
                                <label className="block text-sm font-semibold text-slate-700 mb-1.5">Natural Language Query</label>
                                <div className="relative">
                                    <MessageSquare size={18} className="absolute top-3 left-3 text-slate-400" />
                                    <textarea
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        rows={3}
                                        className="w-full bg-slate-50 border-slate-200 rounded-lg shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm pl-10 py-2.5 resize-none text-slate-900"
                                        placeholder="Ask the AI agent..."
                                    />
                                </div>
                                <div className="mt-2 flex flex-wrap gap-2">
                                    {getExamples().map((ex, i) => (
                                        <button
                                            key={i}
                                            onClick={() => setQuery(ex)}
                                            className="text-[11px] font-medium px-2 py-1 bg-slate-100 text-slate-600 hover:bg-slate-200 rounded transition-colors text-left"
                                        >
                                            &quot;{ex}&quot;
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={handleAnalyze}
                                disabled={loading}
                                className="w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-xl shadow-sm hover:bg-blue-700 hover:shadow-md transition-all disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                            >
                                {loading ? (
                                    <>
                                        <Activity className="animate-spin" size={18} />
                                        Analyzing Satellite Imagery...
                                    </>
                                ) : "Run Agent Analysis"}
                            </button>

                            {error && (
                                <div className="p-4 bg-red-50 border border-red-100 rounded-xl flex items-start gap-3 text-red-700">
                                    <AlertTriangle size={18} className="shrink-0 mt-0.5" />
                                    <p className="text-sm font-medium">{error}</p>
                                </div>
                            )}
                        </div>
                    </div>
                </section>

                {/* RIGHT COLUMN - RESULTS */}
                <section className="lg:col-span-7 flex flex-col gap-6">
                    {!trace && !loading ? (
                        <div className="bg-white p-12 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center justify-center h-full text-center min-h-[500px]">
                            <div className="bg-blue-50 p-4 rounded-full mb-4">
                                <Satellite size={48} className="text-blue-500" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-800 mb-2">Awaiting Instructions</h3>
                            <p className="text-slate-500 max-w-sm text-sm">
                                Upload satellite imagery and ask a question in natural language. The AI agent will route your query to the correct geospatial model.
                            </p>
                        </div>
                    ) : loading ? (
                        <div className="bg-white p-12 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center justify-center h-full text-center min-h-[500px]">
                            <Activity size={48} className="text-blue-500 animate-pulse mb-6" />
                            <h3 className="text-lg font-semibold text-slate-800 mb-2">Agent is Processing...</h3>
                            <div className="space-y-3 text-sm text-slate-500 text-left w-64 mt-4">
                                <div className="flex items-center gap-3"><CheckCircle size={16} className="text-green-500" /> Analyzing query intent</div>
                                <div className="flex items-center gap-3"><Activity size={16} className="text-blue-500 animate-spin" /> Routing to appropriate model</div>
                                <div className="flex items-center gap-3 text-slate-300"><Layers size={16} /> Fusing modalities</div>
                                <div className="flex items-center gap-3 text-slate-300"><Brain size={16} /> Synthesizing final answer</div>
                            </div>
                        </div>
                    ) : (
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col gap-6 animate-in fade-in duration-500">

                            {/* AI ANSWER BANNER */}
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-100">
                                <div className="flex items-start gap-4">
                                    <div className="bg-blue-600 text-white p-2 rounded-full shrink-0 mt-1">
                                        <Brain size={20} />
                                    </div>
                                    <div>
                                        <h3 className="text-sm font-bold text-blue-900 uppercase tracking-wide mb-2">Agent Answer</h3>
                                        <p className="text-slate-800 text-lg leading-relaxed font-medium">
                                            {trace.final_result || "Analysis completed without a textual answer."}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                    <span className="block text-[10px] uppercase font-bold text-slate-400 mb-1">Detected Task</span>
                                    <span className="text-sm font-semibold text-slate-800 capitalize">
                                        {trace?.task?.task_type ? trace.task.task_type.replace(/_/g, ' ') : 'Unknown'}
                                    </span>
                                </div>
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                    <span className="block text-[10px] uppercase font-bold text-slate-400 mb-1">Status</span>
                                    <span className="text-sm font-semibold text-emerald-600 flex items-center justify-center gap-1">
                                        <CheckCircle size={14} /> Success
                                    </span>
                                </div>
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                    <span className="block text-[10px] uppercase font-bold text-slate-400 mb-1">Confidence</span>
                                    <span className="text-sm font-semibold text-slate-800">
                                        {trace.final_confidence ? `${(trace.final_confidence * 100).toFixed(1)}%` : 'N/A'}
                                    </span>
                                </div>
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                    <span className="block text-[10px] uppercase font-bold text-slate-400 mb-1">Model Inference</span>
                                    <span className="text-sm font-semibold text-slate-800">Completed</span>
                                </div>
                            </div>

                            {/* VISUAL EVIDENCE */}
                            <div className="border-t border-slate-100 pt-6">
                                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide mb-4">Visual Evidence</h3>
                                <div className="space-y-6">
                                    {files.length > 0 && renderImages()}

                                    {maskDataUrl && (
                                        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                            <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                                                <Layers size={16} className="text-indigo-500" />
                                                Model Prediction Overlay
                                            </h4>

                                            <div className="relative inline-block w-full max-w-lg mx-auto bg-gray-200 rounded-lg overflow-hidden border border-slate-300 shadow-sm aspect-square">
                                                <img
                                                    src={trace?.final_evidence?.input_thumbnails_b64 ?
                                                        `data:image/png;base64,${trace.final_evidence.input_thumbnails_b64[files.length === 2 && trace.task.task_type !== 'bi_temporal_change' ? 0 : files.length - 1]}` :
                                                        URL.createObjectURL(files[files.length === 2 && trace.task.task_type !== 'bi_temporal_change' ? 0 : files.length - 1])}
                                                    alt="Base Image"
                                                    className="absolute inset-0 w-full h-full object-cover"
                                                />
                                                <img
                                                    src={maskDataUrl}
                                                    alt="AI Prediction Mask"
                                                    className="absolute inset-0 w-full h-full object-cover opacity-70"
                                                    style={{ imageRendering: 'pixelated' }}
                                                />
                                            </div>

                                            <div className="mt-4 flex flex-col items-center">
                                                {trace?.task?.task_type === 'optical_sar_analysis' && (
                                                    <div className="flex flex-wrap justify-center gap-4 text-xs font-medium text-slate-600 bg-white p-3 rounded-lg border border-slate-100 shadow-sm">
                                                        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-[#228B22]"></span> Vegetation</span>
                                                        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-[#DC143C]"></span> Built-up</span>
                                                        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-[#1E90FF]"></span> Water</span>
                                                        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-[#DAA520]"></span> Bare Land</span>
                                                    </div>
                                                )}
                                                {trace?.task?.task_type === 'bi_temporal_change' && (
                                                    <div className="flex justify-center gap-4 text-xs font-medium text-slate-600 bg-white p-3 rounded-lg border border-slate-100 shadow-sm">
                                                        <span className="flex items-center gap-1.5"><span className="w-3 h-3 border-2 border-slate-400 bg-white shadow-sm"></span> Changed Area</span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* QUANTITATIVE METRICS */}
                            {trace?.final_evidence && (trace.final_evidence.percentage_change !== undefined || trace.final_evidence.class_percentages) && (
                                <div className="border-t border-slate-100 pt-6">
                                    <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide mb-4">Quantitative Metrics</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        {trace.final_evidence.percentage_change !== undefined && (
                                            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex flex-col items-center justify-center">
                                                <span className="text-3xl font-bold text-indigo-600">{trace.final_evidence.percentage_change.toFixed(2)}%</span>
                                                <span className="text-xs font-semibold text-slate-500 uppercase mt-1">Total Changed Area</span>
                                            </div>
                                        )}
                                        {trace.final_evidence.class_percentages && Object.entries(trace.final_evidence.class_percentages).map(([cls, pct]: any) => (
                                            <div key={cls} className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex flex-col items-center justify-center">
                                                <span className="text-2xl font-bold text-slate-700">{pct.toFixed(1)}%</span>
                                                <span className="text-xs font-semibold text-slate-500 uppercase mt-1 capitalize">{cls}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* AGENT TRACE TIMELINE */}
                            <div className="border-t border-slate-100 pt-6">
                                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide mb-4 flex items-center gap-2">
                                    <Activity size={16} className="text-blue-500" /> Agent Execution Trace
                                </h3>
                                <div className="space-y-4 relative before:absolute before:inset-0 before:ml-2.5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-200 before:to-transparent">
                                    {trace.steps?.map((step: any, idx: number) => (
                                        <div key={idx} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                                            <div className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-white bg-blue-100 text-blue-600 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-sm z-10">
                                                <CheckCircle size={12} />
                                            </div>
                                            <div className="w-[calc(100%-3rem)] md:w-[calc(50%-1.5rem)] p-3 rounded-lg border border-slate-200 bg-white shadow-sm">
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="font-bold text-xs text-slate-800">{step.step_name}</span>
                                                    <span className="text-[9px] font-bold uppercase px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-100">{step.status}</span>
                                                </div>
                                                <div className="text-xs text-slate-500">{step.description}</div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                        </div>
                    )}
                </section>
            </main>
        </div>
    );
}
