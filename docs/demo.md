# Demo Workflows

The application includes predefined model mocks to demonstrate agentic capabilities without requiring heavy GPU inference immediately.

## Demo 1 - VQA
1. Mode: Single Image
2. Upload any valid image (e.g. `test.jpg`).
3. Query: "Describe the land-cover and major objects visible in this image."
4. Result: Task classifier selects VQA -> MockVQATool executes.

## Demo 2 - Grounding
1. Mode: Single Image
2. Upload image.
3. Query: "Highlight the water body referred to in the query."
4. Result: Grounding task detected. Bounding box returned as evidence.

## Demo 3 - Bi-temporal Change
1. Mode: Bi-temporal Pair
2. Upload two temporal images (T1, T2).
3. Query: "What changed between these two dates, and where did the change occur?"
4. Result: Bi-temporal task -> Change tool executed -> Change map generated.

## Demo 4 - Optical + SAR
1. Mode: Optical + SAR Pair
2. Upload one Optical, one SAR image.
3. Query: "Use the optical and SAR images together to identify built-up and water-covered regions."
4. Result: Fusion task -> Optical/SAR tool executed.
