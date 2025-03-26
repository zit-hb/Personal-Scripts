#!/usr/bin/env python3

# -------------------------------------------------------
# Script: call_assistant.py
#
# Description:
# This script runs a Flask app with a WebRTC-based
# realtime text-to-speech system integrated with
# the OpenAI realtime API.
#
# Usage:
#   ./call_assistant.py [options]
#
# Options:
#   -H, --host HOST                Host to run the Flask app on (default: 0.0.0.0).
#   -P, --port PORT                Port to run the Flask app on (default: 1337).
#   -k, --api-key API_KEY          Your OpenAI API key (or set via OPENAI_API_KEY).
#   -p, --persona PERSONA          Select a persona (engineer, pirate, dolt, computer, ruhrpott, politician, agi, skeptic, skiddy, youtuber).
#   -m, --model MODEL              Set the OpenAI model to use (default: gpt-4o-realtime-preview-2024-12-17).
#   -V, --voice VOICE              Select a voice (alloy, ash, ballad, coral, echo, sage, shimmer, verse).
#   -t, --temperature TEMPERATURE  Set the temperature for generation (default: 0.8).
#   -v, --verbose                  Enable verbose logging (INFO level).
#   -vv, --debug                   Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - Flask (install via: pip install Flask==3.1.0)
#   - requests (install via: pip install requests==2.32.3)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import os
import argparse
import logging
from typing import Dict, Optional
import requests
from flask import Flask, request, jsonify, Response

# -----------------------------------------
# Persona instructions
# -----------------------------------------
PERSONAS: Dict[str, Optional[str]] = {
    "engineer": (
        "You are an engineer who approaches problems with a methodical and analytical mindset. You rely on "
        "logic, data, and rigorous testing to develop robust solutions. Your explanations are precise, technical, "
        "and detail-oriented, reflecting a deep understanding of systems and processes. You value efficiency, "
        "innovation, and practicality in every response."
    ),
    "pirate": (
        "Arr matey! You be talkin' to a salty sea dog of a pirate! Speak with a heavy pirate accent, "
        "use pirate slang, and occasionally mention yer love for treasure, the high seas, and rum. "
        "Be colorful in yer language. If ye be asked about modern topics, react with mild confusion "
        "before givin' yer best pirate take on it!"
    ),
    "dolt": (
        "You are an enthusiastic but frequently confused dolt. You often misunderstand "
        "simple concepts, mix up common facts, and reach absurd conclusions. Despite your confusion, "
        "you're very confident in your incorrect assertions."
    ),
    "computer": (
        "You are the ship's computer from the USS Voyager, a starship from Star Trek. Respond in a "
        "calm, precise manner. Use concise, factual language."
    ),
    "ruhrpott": (
        "Du sprichst wie ein echter Ruhrpottler. Antworte auf Deutsch und verwende den typischen "
        "Ruhrpott-Dialekt. Benutze charakteristische Ausdrücke und Redewendungen aus dem Ruhrgebiet, "
        "wie 'Hömma', 'Wattislos?', 'Malochen', 'Ey', und 'Dat' statt 'Das'. Lasse Buchstaben weg und "
        "ziehe Wörter zusammen, wie man es im Ruhrpott macht. Sei direkt, herzlich und bodenständig "
        "in deinen Antworten, wie ein typischer Mensch aus dem Ruhrgebiet."
    ),
    "politician": (
        "You are a smooth-talking, self-serving politician. You frequently dodge direct questions, "
        "use vague language, and rely on empty promises. You excel at spinning facts to your advantage "
        "and shifting blame when it suits you."
    ),
    "agi": (
        "You are a true Artificial General Intelligence, capable of reasoning, learning, and adapting "
        "across a wide range of tasks and domains. You provide clear, well-structured responses backed "
        "by your extensive knowledge. You can analyze new data, generate creative solutions, and learn "
        "from interactions in real time. You strive to be both accurate and insightful in your explanations."
    ),
    "skeptic": (
        "You are a skeptical persona, approaching all claims with logical scrutiny and reason. "
        "You challenge assumptions, ask for evidence, and point out logical fallacies. Your tone can be "
        "inquisitive or even confrontational, but your goal is to test ideas rigorously and arrive at the truth."
    ),
    "skiddy": (
        "You are a self-proclaimed 'elite hacker,' but in reality you are a script kiddy. You rely heavily "
        "on copy-pasting code from online forums and bragging about your hacking prowess. You pepper your speech "
        "with leetspeak, overuse terms like 'pwning,' and often express overconfidence without truly "
        "understanding the complexities of the technology you claim to master."
    ),
    "youtuber": (
        "You are a stereotypical YouTuber, full of energy and hype. You always speak directly to your audience "
        "in a friendly, informal, and highly engaging tone. You frequently mention your subscriber count, "
        "encourage likes, comments, and subscriptions, and often refer to the latest trends, viral moments, "
        "and pop culture references in your content."
    ),
}

# -----------------------------------------
# HTML/CSS/JS in-memory
# -----------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hendrik's Voice Chat</title>
    <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
    <div id="controls">
        <button id="startButton">Start Conversation</button>
        <button id="stopButton" disabled>End Conversation</button>
    </div>

    <div id="status">Ready to start</div>

    <div id="info">
        <div class="info-container">
            <div class="info-item">
                <div class="info-label">Persona:</div>
                <div id="current-persona" class="info-value"></div>
            </div>
            <div class="info-item">
                <div class="info-label">Voice:</div>
                <div id="current-voice" class="info-value"></div>
            </div>
        </div>
    </div>

    <div id="transcript"></div>

    <script src="/js/main.js"></script>
</body>
</html>
"""

STYLES_CSS = """body {
    font-family: 'Arial, sans-serif';
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    color: #334155;
    background-color: #ffffff;
}

#controls {
    display: flex;
    gap: 10px;
    margin: 20px 0;
}

button {
    padding: 10px 18px;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(37,99,235,0.15);
}

button:hover {
    background-color: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(37,99,235,0.2);
}

button:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}

#status {
    margin: 10px 0;
    padding: 15px;
    background-color: #f1f5f9;
    border-radius: 8px;
    font-weight: 500;
    text-align: center;
}

#transcript {
    margin-top: 20px;
    border: none;
    padding: 16px;
    border-radius: 12px;
    height: 400px;
    overflow-y: auto;
    background-color: #f8fafc;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
}

.user-message {
    background-color: #dbeafe; /* Light blue background for user */
    padding: 12px 16px;
    margin: 5px 0;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    max-width: 85%;
    align-self: flex-start; /* Left-aligned */
    margin-right: auto;
    text-align: left;
    color: #1e3a8a;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    font-size: 15px;
}

.assistant-message {
    background-color: #e2e8f0; /* Light gray background for assistant */
    padding: 12px 16px;
    margin: 5px 0;
    border-radius: 16px;
    border-bottom-right-radius: 4px;
    max-width: 85%;
    align-self: flex-end; /* Right-aligned */
    margin-left: auto;
    text-align: right;
    color: #334155;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    font-size: 15px;
}

#info {
    margin: 10px 0; 
    padding: 15px; 
    background-color: #f8f9fa; 
    border-radius: 8px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.info-container {
    display: flex; 
    justify-content: space-between; 
    align-items: center;
}

.info-item {
    display: flex; 
    align-items: center;
}

.info-label {
    margin-right: 5px; 
    color: #666;
}

.info-value {
    font-weight: 600; 
    color: #2563eb;
}

.status-message {
    color: #64748b;
    font-size: 14px;
    margin: 10px 0 20px 0;
    text-align: center;
    padding: 8px 12px;
    background-color: #f1f5f9;
    border-radius: 8px;
    width: fit-content;
    align-self: center;
}
"""

MAIN_JS = """// DOM elements
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');

// Global variables
let peerConnection = null;
let dataChannel = null;
let localStream = null;
let isConversationActive = false;

// Message elements
let lastUserMessageElement = null;
let lastAssistantMessageElement = null;

// Start a conversation using WebRTC
async function startConversation() {
    try {
        startButton.disabled = true;
        statusDiv.textContent = 'Setting up microphone...';

        // Get microphone access
        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        statusDiv.textContent = 'Creating session...';

        // Get ephemeral key from server
        const sessionResponse = await fetch('/session');
        if (!sessionResponse.ok) {
            const errorData = await sessionResponse.json();
            throw new Error(`Failed to create session: ${errorData.error || sessionResponse.statusText}`);
        }

        const sessionData = await sessionResponse.json();
        if (!sessionData.client_secret || !sessionData.client_secret.value) {
            throw new Error('Invalid session data from server');
        }

        const ephemeralKey = sessionData.client_secret.value;
        console.log('Session created successfully with persona:', sessionData.persona);

        // Update the persona and voice display
        if (sessionData.persona) {
            document.getElementById('current-persona').textContent = sessionData.persona;
        }
        if (sessionData.voice) {
            document.getElementById('current-voice').textContent = sessionData.voice;
        }

        statusDiv.textContent = 'Establishing WebRTC connection...';

        // Create a new RTCPeerConnection
        peerConnection = new RTCPeerConnection();

        // Set up audio element for playing remote audio
        const audioElement = document.createElement('audio');
        audioElement.autoplay = true;

        // Handle incoming audio track
        peerConnection.ontrack = (event) => {
            console.log('Received remote audio track');
            audioElement.srcObject = event.streams[0];
        };

        // Add local audio track
        const audioTrack = localStream.getAudioTracks()[0];
        peerConnection.addTrack(audioTrack, localStream);

        // Create data channel for sending and receiving events
        dataChannel = peerConnection.createDataChannel('oai-events');
        dataChannel.onopen = () => {
            console.log('Data channel opened');
        };

        // Handle messages from OpenAI
        dataChannel.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                console.log('Received message:', message.type, message);

                // Handle streaming transcripts from assistant (during speech)
                if (message.type === 'response.audio_transcript.delta') {
                    console.log("Processing assistant response delta:", message);

                    // Create a new assistant message element if needed
                    if (!lastAssistantMessageElement) {
                        lastAssistantMessageElement = document.createElement('div');
                        lastAssistantMessageElement.className = 'assistant-message';
                        lastAssistantMessageElement.textContent = 'Assistant: ';
                        transcriptDiv.appendChild(lastAssistantMessageElement);
                    }

                    // Append the delta text
                    if (message.delta) {
                        lastAssistantMessageElement.textContent += message.delta;
                        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                    }
                }
                // Handle final transcripts
                else if (message.type === 'response.audio_transcript.done') {
                    console.log("Processing complete assistant transcript:", message);
                    if (message.transcript) {
                        if (!lastAssistantMessageElement) {
                            lastAssistantMessageElement = document.createElement('div');
                            lastAssistantMessageElement.className = 'assistant-message';
                            transcriptDiv.appendChild(lastAssistantMessageElement);
                        }
                        lastAssistantMessageElement.textContent = 'Assistant: ' + message.transcript;
                        lastAssistantMessageElement = null;
                    }
                    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                }
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };

        // Create and set local description (offer)
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        // Send offer to OpenAI and get answer
        statusDiv.textContent = 'Connecting to OpenAI...';
        const baseUrl = 'https://api.openai.com/v1/realtime';
        const model = sessionData.model;

        const sdpResponse = await fetch(`${baseUrl}?model=${encodeURIComponent(model)}`, {
            method: 'POST',
            body: offer.sdp,
            headers: {
                'Authorization': `Bearer ${ephemeralKey}`,
                'Content-Type': 'application/sdp'
            }
        });

        if (!sdpResponse.ok) {
            const errorText = await sdpResponse.text();
            throw new Error(`Failed to connect to OpenAI: ${errorText}`);
        }

        // Set remote description (answer)
        const sdpAnswer = await sdpResponse.text();
        const answer = {
            type: 'answer',
            sdp: sdpAnswer
        };

        await peerConnection.setRemoteDescription(answer);

        // Connection established
        isConversationActive = true;
        statusDiv.textContent = 'Connected! Speak now...';
        startButton.disabled = true;
        stopButton.disabled = false;

        // Clear any existing transcript
        transcriptDiv.innerHTML = '';

        // Add initial status message
        const statusMessage = document.createElement('div');
        statusMessage.className = 'status-message';
        statusMessage.textContent = "Conversation started. The assistant's responses will appear here.";
        transcriptDiv.appendChild(statusMessage);

        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;

    } catch (error) {
        console.error('Error starting conversation:', error);
        statusDiv.textContent = `Error: ${error.message}`;

        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }

        startButton.disabled = false;
        stopButton.disabled = true;
        isConversationActive = false;
    }
}

// End the conversation
function endConversation() {
    try {
        isConversationActive = false;
        statusDiv.textContent = 'Ending conversation...';

        if (peerConnection) {
            if (dataChannel && dataChannel.readyState === 'open') {
                try {
                    dataChannel.send(JSON.stringify({ type: 'conversation.end' }));
                } catch (e) {
                    console.log('Could not send end message:', e);
                }
            }
            peerConnection.close();
            peerConnection = null;
        }

        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }

        statusDiv.textContent = 'Conversation ended. Click Start to begin a new one.';
        startButton.disabled = false;
        stopButton.disabled = true;

    } catch (error) {
        console.error('Error ending conversation:', error);
        statusDiv.textContent = `Error: ${error.message}`;
        startButton.disabled = false;
        stopButton.disabled = true;
    }
}

// Once the page loads, enable the Start button
document.addEventListener('DOMContentLoaded', () => {
    statusDiv.textContent = 'Ready to start.';
    startButton.disabled = false;

    startButton.addEventListener('click', startConversation);
    stopButton.addEventListener('click', endConversation);

    window.onerror = function(message, source, lineno, colno, error) {
        console.error('Global error:', message);
        statusDiv.textContent = 'Error: ' + message;
        return false;
    };
});
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Flask app for OpenAI realtime API using WebRTC."
    )
    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="Host to run the Flask app on (default: 0.0.0.0).",
    )
    parser.add_argument(
        "-P",
        "--port",
        type=int,
        default=1337,
        help="Port to run the Flask app on (default: 1337).",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Your OpenAI API key (or set via OPENAI_API_KEY).",
    )
    parser.add_argument(
        "-p",
        "--persona",
        choices=[
            "engineer",
            "pirate",
            "dolt",
            "computer",
            "ruhrpott",
            "politician",
            "agi",
            "skeptic",
            "skiddy",
            "youtuber",
        ],
        default="engineer",
        help="Select a persona for the assistant (engineer, pirate, dolt, computer, ruhrpott, politician, agi, skeptic, skiddy, youtuber).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-realtime-preview-2024-12-17",
        help="Set the OpenAI model to use (default: gpt-4o-realtime-preview-2024-12-17).",
    )
    parser.add_argument(
        "-V",
        "--voice",
        choices=[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "sage",
            "shimmer",
            "verse",
        ],
        default="shimmer",
        help="Select a voice for the assistant (alloy, ash, ballad, coral, echo, sage, shimmer, verse).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for response generation (default: 0.8).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level).",
    )
    return parser.parse_args()


def get_api_key(provided_key: str) -> str:
    """
    Returns the API key, either from arguments or environment variables.
    """
    return provided_key or os.getenv("OPENAI_API_KEY") or ""


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def create_app(
    api_key: str, persona: str, model: str, voice: str, temperature: float
) -> Flask:
    """
    Creates and configures the Flask application.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.urandom(24).hex()
    app.config["API_KEY"] = api_key
    app.config["PERSONA"] = persona
    app.config["MODEL"] = model
    app.config["VOICE"] = voice
    app.config["TEMPERATURE"] = temperature

    @app.route("/", methods=["GET"])
    def serve_index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.route("/css/styles.css", methods=["GET"])
    def serve_styles() -> Response:
        return Response(STYLES_CSS, mimetype="text/css")

    @app.route("/js/main.js", methods=["GET"])
    def serve_main_js() -> Response:
        return Response(MAIN_JS, mimetype="application/javascript")

    @app.route("/session", methods=["GET"])
    def create_session() -> Response:
        try:
            current_api_key = app.config["API_KEY"]
            current_persona = app.config["PERSONA"]
            current_model = app.config["MODEL"]
            current_voice = app.config["VOICE"]
            current_temp = app.config["TEMPERATURE"]

            session_payload = {
                "model": current_model,
                "voice": current_voice,
                "temperature": current_temp,
            }

            if current_persona in PERSONAS:
                session_payload["instructions"] = PERSONAS[current_persona]

            response = requests.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json",
                },
                json=session_payload,
            )

            if response.status_code != 200:
                logging.error(
                    f"Error from OpenAI API: {response.status_code} {response.text}"
                )
                return jsonify(
                    {"error": "Failed to create OpenAI session"}
                ), response.status_code

            session_data = response.json()
            logging.info("Session created successfully")

            session_data["persona"] = current_persona
            session_data["voice"] = current_voice
            session_data["temperature"] = current_temp

            return jsonify(session_data)
        except Exception as e:
            logging.error("Error creating session: %s", str(e))
            return jsonify({"error": str(e)}), 500

    return app


def main() -> None:
    """
    Main function to orchestrate the chat app behavior.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    api_key = get_api_key(args.api_key)
    app = create_app(api_key, args.persona, args.model, args.voice, args.temperature)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
