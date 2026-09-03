#!/bin/bash
echo "=== CPU Info ==="
sysctl -n machdep.cpu.brand_string
echo "=== Memory Info ==="
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'
echo "=== Python Version ==="
python3 --version || python --version
echo "=== Node Version ==="
node --version
echo "=== Docker Version ==="
docker --version || echo "Docker not found"
echo "=== Disk Space ==="
df -h /
echo "=== Gemini Configured ==="
if [ -n "$GEMINI_API_KEY" ]; then echo "Yes"; else echo "No"; fi
