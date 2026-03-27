#!/bin/bash

echo "🚀 Starting IntelliAIDrive"

# Check for video file
if [ ! -f "data/video.mp4" ]; then
    echo "⚠️ Warning: data/video.mp4 not found. Video feed may show a loading or empty state."
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip3 install -r requirements.txt

# Start backend
echo "🌐 Launching FastAPI backend on http://localhost:8000"
python3 src/main.py &
BACKEND_PID=$!

# Start frontend if folder exists
if [ -d "frontend/react-app" ]; then
    echo "⚛️ Preparing React frontend..."
    cd frontend/react-app || exit

    if [ ! -d "node_modules" ]; then
        npm install
    fi

    echo "💻 Launching React dashboard..."
    npm run dev &
    FRONTEND_PID=$!

    cd ../..
else
    echo "⚠️ Frontend folder not found. Starting backend only."
    FRONTEND_PID=""
fi

cleanup() {
    echo "🛑 Shutting down IntelliAIDrive..."
    if [ -n "$BACKEND_PID" ]; then kill $BACKEND_PID 2>/dev/null; fi
    if [ -n "$FRONTEND_PID" ]; then kill $FRONTEND_PID 2>/dev/null; fi
    exit
}

trap cleanup SIGINT SIGTERM

echo "✅ System initialized. Press Ctrl+C to stop."

wait