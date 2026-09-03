"use client";
import { useState, useRef, useEffect } from 'react';

export default function Home() {
    const [query, setQuery] = useState("");
    const [inputType, setInputType] = useState("single");
    const [files, setFiles] = useState<File[]>([]);
    const [trace, setTrace] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
        }
    };

    const handleAnalyze = async () => {
        if (files.length === 0) {
            setError("Please upload images.");
            return;
        }
        setLoading(true);
        setError("");
        setTrace(null);

        const formData = new FormData();
        formData.append("query", query);
        formData.append("input_type", inputType);
        files.forEach(f => formData.append("files", f));

        try {
            const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const res = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error("Analysis failed.");
            const data = await res.json();
            setTrace(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (trace?.final_evidence?.bounding_boxes && imageRef.current && canvasRef.current) {
            const img = imageRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            
            // Set canvas size to match image
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw boxes
            const scaleX = canvas.width / img.naturalWidth;
            const scaleY = canvas.height / img.naturalHeight;
            
            trace.final_evidence.bounding_boxes.forEach((b: any) => {
                const [xmin, ymin, xmax, ymax] = b.box;
                const rectX = xmin * scaleX;
                const rectY = ymin * scaleY;
                const rectW = (xmax - xmin) * scaleX;
                const rectH = (ymax - ymin) * scaleY;
                
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(rectX, rectY, rectW, rectH);
                
                ctx.fillStyle = 'red';
                ctx.font = '14px sans-serif';
                ctx.fillText(`${b.label} (${(b.score*100).toFixed(0)}%)`, rectX, rectY > 15 ? rectY - 5 : rectY + 15);
            });
        }
    }, [trace, files]);

    return (
        <div className="min-h-screen bg-gray-50 p-8 text-black">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">SatQuery AI</h1>
                <p className="text-gray-500">Multimodal Remote Sensing Image Analysis</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Input Workspace */}
                <section className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h2 className="text-xl font-semibold mb-4 text-black">Analysis Input</h2>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Input Mode</label>
                        <select value={inputType} onChange={(e) => setInputType(e.target.value)} className="w-full border-gray-300 rounded-md shadow-sm p-2 border">
                            <option value="single">Single Image</option>
                            <option value="temporal_pair">Bi-temporal Pair (Change Analysis)</option>
                            <option value="optical_sar_pair">Optical + SAR Pair</option>
                        </select>
                    </div>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Upload Images</label>
                        <input type="file" multiple onChange={handleFileChange} className="block w-full text-sm text-gray-500" />
                    </div>
                    <div className="mb-6">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Natural Language Query</label>
                        <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} className="w-full border-gray-300 rounded-md shadow-sm p-2 border text-black" />
                    </div>
                    <button onClick={handleAnalyze} disabled={loading} className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50">
                        {loading ? "Analyzing..." : "Analyze"}
                    </button>
                    {error && <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">{error}</div>}
                </section>

                {/* Results Workspace */}
                <section className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 flex flex-col">
                    <h2 className="text-xl font-semibold text-black mb-4">Results & Execution Trace</h2>
                    
                    {/* Visual Evidence Area */}
                    {files.length > 0 && inputType === "single" && (
                        <div className="relative mb-4 inline-block w-full">
                            <img 
                                ref={imageRef}
                                src={URL.createObjectURL(files[0])} 
                                alt="Input" 
                                className="w-full object-contain rounded-md border"
                                onLoad={() => setTrace({...trace})} // Trigger redraw
                            />
                            <canvas 
                                ref={canvasRef} 
                                className="absolute top-0 left-0 w-full h-full pointer-events-none"
                            />
                        </div>
                    )}

                    {trace && (
                        <div className="space-y-6 flex-1 overflow-auto mt-4 border-t pt-4">
                            <div>
                                <h3 className="text-lg font-medium text-gray-900">Answer</h3>
                                <p className="mt-2 text-gray-700 bg-gray-50 p-3 rounded border">
                                    {trace.final_result || "No textual answer generated."}
                                </p>
                            </div>
                            {trace.final_confidence !== null && (
                                <div>
                                    <h3 className="text-sm font-medium text-gray-900">Confidence</h3>
                                    <p className="text-sm text-green-600 font-semibold">
                                        {(trace.final_confidence * 100).toFixed(1)}%
                                    </p>
                                </div>
                            )}
                            <div>
                                <h3 className="text-md font-medium text-gray-900 mb-2 border-b pb-2">Execution Trace</h3>
                                <div className="space-y-3">
                                    {trace.steps.map((step: any, idx: number) => (
                                        <div key={idx} className="flex flex-col text-sm border-l-2 border-blue-500 pl-3">
                                            <span className="font-semibold text-gray-800">{step.step_name}</span>
                                            <span className="text-gray-600">{step.description}</span>
                                            <span className="text-xs mt-1 font-semibold text-gray-500">Status: {step.status}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}
