// State
let currentMode = 'http';
let websocket = null;
let audioChunks = [];
let isPlayingAudio = false;
let typingTimer = null;
const TYPING_DELAY = 1000;

// HTTP Mode
async function speakHTTP() {
    const text = document.getElementById("text").value;
    if (!text.trim()) return;
    
    try {
        const audio = document.getElementById("audio");
        audio.src = '/tts?text=' + encodeURIComponent(text);
        audio.play();
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message);
    }
}

// WebSocket Mode
async function speakWebSocket() {
    const text = document.getElementById("text").value;
    if (!text.trim()) return;

    try {
        // Initialize WebSocket if needed
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            await initWebSocket();
        }

        // Send text to server
        websocket.send(text);
        updateStatus('Generating audio...');
        
        // Clear text field
        document.getElementById("text").value = '';
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message);
    }
}

function initWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = () => {
            console.log('WebSocket connected');
            updateStatus('Connected');
            resolve();
        };

        websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                // Handle audio chunks
                if (message.audioOutput?.audio) {
                    const audioData = base64ToArrayBuffer(message.audioOutput.audio);
                    audioChunks.push(audioData);
                }
                
                // Handle completion
                if (message.finalOutput?.isFinal) {
                    playAudio();
                }
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('Connection error');
            reject(error);
        };

        websocket.onclose = () => {
            console.log('WebSocket closed');
            updateStatus('Disconnected');
            websocket = null;
        };
    });
}

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function playAudio() {
    if (audioChunks.length === 0 || isPlayingAudio) return;
    
    isPlayingAudio = true;
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    
    const audio = document.getElementById("audio");
    audio.src = audioUrl;
    audio.play()
        .then(() => updateStatus('Playing'))
        .catch(err => {
            console.error('Playback error:', err);
            updateStatus('Playback error');
            isPlayingAudio = false;
        });
    
    audio.onended = () => {
        isPlayingAudio = false;
        updateStatus('Ready');
        URL.revokeObjectURL(audioUrl);
    };
    
    audioChunks = [];
}

// UI Functions
function updateStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

function setMode(mode) {
    currentMode = mode;
    const httpBtn = document.getElementById('httpMode');
    const wsBtn = document.getElementById('wsMode');
    const speakBtn = document.getElementById('speakButton');
    
    if (mode === 'http') {
        httpBtn.classList.add('active');
        wsBtn.classList.remove('active');
        speakBtn.style.display = 'block';
        updateStatus('Mode: HTTP');
        
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    } else {
        wsBtn.classList.add('active');
        httpBtn.classList.remove('active');
        speakBtn.style.display = 'none';
        updateStatus('Mode: WebSocket');
    }
}

function handleTextInput() {
    if (currentMode !== 'websocket') return;
    
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => {
        const text = document.getElementById("text").value;
        if (text.trim()) {
            speakWebSocket();
        }
    }, TYPING_DELAY);
}

async function setEngine() {
    const engine = document.getElementById("engine").value;
    await fetch('/set_engine?engine_name=' + engine);
}

async function fetchVoices() {
    try {
        const response = await fetch('/voices');
        if (!response.ok) throw new Error('Failed to fetch voices');
        
        const voices = await response.json();
        const dropdown = document.getElementById("voice");
        dropdown.innerHTML = '';
        
        voices.forEach(voice => {
            const option = document.createElement("option");
            option.text = voice;
            option.value = voice;
            dropdown.add(option);
        });
    } catch (error) {
        console.error('Error fetching voices:', error);
    }
}

async function setVoice() {
    const voice = document.getElementById("voice").value;
    try {
        const response = await fetch('/setvoice?voice_name=' + encodeURIComponent(voice));
        if (!response.ok) throw new Error('Failed to set voice');
        console.log('Voice set:', voice);
    } catch (error) {
        console.error('Error setting voice:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const textArea = document.getElementById("text");
    textArea.value = "This is a text to speech demo text";
    
    // Event listeners
    document.getElementById("speakButton").addEventListener("click", () => {
        if (currentMode === 'http') speakHTTP();
        else speakWebSocket();
    });
    
    document.getElementById("engine").addEventListener("change", async () => {
        await setEngine();
        await fetchVoices();
    });
    
    document.getElementById("voice").addEventListener("change", setVoice);
    document.getElementById("httpMode").addEventListener("click", () => setMode('http'));
    document.getElementById("wsMode").addEventListener("click", () => setMode('websocket'));
    
    textArea.addEventListener("input", handleTextInput);
    textArea.addEventListener("keydown", () => {
        if (currentMode === 'websocket') clearTimeout(typingTimer);
    });

    fetchVoices();
});
