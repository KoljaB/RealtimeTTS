async function setEngine() {
    var engine = document.getElementById("engine").value;
    await fetch('/set_engine?engine_name=' + engine);
}

async function speak() {
    var text = document.getElementById("text").value;
    try {
        var url = '/tts?text=' + encodeURIComponent(text);
        var audio = document.getElementById("audio");
        audio.src = url;
        audio.play();
    } catch (error) {
        console.error('Error during fetch or audio playback:', error);
    }
}

async function fetchVoices() {
    try {
        var response = await fetch('/voices');
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        var data = await response.json();
        var voicesDropdown = document.getElementById("voice");
        voicesDropdown.innerHTML = ''; // Clear previous options
        data.forEach(function(voice) {
            var option = document.createElement("option");
            option.text = voice;
            option.value = voice;
            voicesDropdown.add(option);
        });
    } catch (error) {
        console.error('Error fetching voices:', error);
    }
}

async function setVoice() {
    var voice = document.getElementById("voice").value;
    try {
        var response = await fetch('/setvoice?voice_name=' + encodeURIComponent(voice));
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        console.log('Voice set successfully:', voice);
    } catch (error) {
        console.error('Error setting voice:', error);
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById("text").value = "This is a text to speech demo text";
    document.getElementById("speakButton").addEventListener("click", speak);
    document.getElementById("engine").addEventListener("change", async function() {
        await setEngine();
        await fetchVoices();
    });    
    document.getElementById("voice").addEventListener("change", setVoice);

    fetchVoices();
});
