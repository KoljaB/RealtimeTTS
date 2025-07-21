Run the following command from the RealtimeTTS main directory:
```bash
docker build -t zipvoice-image -f docker/zipvoice/Dockerfile .
```

Run:
```bash
docker run --rm -p 9086:9086 --gpus all zipvoice-image
```

Stop:
```bash
docker stop zipvoice-image
```

Test:
```
curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"Hi there! I'm really excited to try this out! I hope the speech sounds natural and warm. That's exactly what I'm going for!\", \"voice\": \"alpha-warm\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0

curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"Your voice just got supercharged! Crystal clear audio that flows like silk and hits like thunder!\", \"voice\": \"beta-intense\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0

curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"Hey! So this is me testing out my voice... kinda nervous but also excited about it. This whole voice synthesis thing is sooo fascinating. I mean... technology these days is like creating a perfect robot version of a person, right?\", \"voice\": \"alpha-warm\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0 && curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"The voice you knew... is GONE. What you're hearing now... is something ENTIRELY different. This isn't just another voice - this is POWER unleashed, INTENSITY personified, COMMAND that cuts through your soul like a blade through silence.\", \"voice\": \"beta-intense\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0
```