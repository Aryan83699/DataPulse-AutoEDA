from flask import Flask, request, jsonify, Response, stream_with_context
from chatbot_engine import dataset_chat_response, dataset_chat_response_stream
import json
from story_engine import generate_data_story
import os

app = Flask(__name__)

# Regular endpoint (instant response)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    report_path = data.get("report_path")
    csv_path = data.get("csv_path")
    question = data.get("question")
    session_id = data.get("session_id", "default")

    answer = dataset_chat_response(report_path, csv_path, question, session_id)

    return jsonify({"answer": answer})


# Streaming endpoint (animation effect)
@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    data = request.json

    report_path = data.get("report_path")
    csv_path = data.get("csv_path")
    question = data.get("question")
    session_id = data.get("session_id", "default")

    def generate():
        for chunk in dataset_chat_response_stream(report_path, csv_path, question, session_id):
            # Send each chunk as Server-Sent Events (SSE)
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route("/story", methods=["POST"])
def story():
    data = request.json
    csv_path = data.get("csv_path")
    
    print("Received csv_path:", csv_path)  # ADD THIS
    print("File exists:", os.path.exists(csv_path))  # ADD THIS

    if not csv_path:
        return jsonify({"error": "No dataset path provided"})

    result = generate_data_story(csv_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=8000, debug=True)