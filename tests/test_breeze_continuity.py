"""Response context ownership, instructions, cancellation and voice isolation."""
from RealtimeTTS import BreezeVoice
from tests.test_breeze_engine_server import Socket, make_engine, server_client


def test_one_native_session_carries_instructions_between_fragments(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module
    sockets=[]
    def connect(*args, **kwargs):
        sock=Socket()
        sockets.append(sock)
        return sock
    engine,_=make_engine(monkeypatch)
    monkeypatch.setattr(module.websocket,"create_connection",connect)
    context=engine.create_stream_context()
    engine.set_stream_context(context)
    engine.set_voice(BreezeVoice("test",voice_id="test",instruct="Warm"))
    assert engine.synthesize("I am here with you.")
    assert not engine.last_metrics["session_reused"]
    engine.set_voice(BreezeVoice("test",voice_id="test",instruct="Whisper softly"))
    assert engine.synthesize("There is no need to rush.")
    assert engine.last_metrics["session_reused"]
    assert len(sockets)==1
    assert sum(m["type"]=="start" for m in sockets[0].messages)==1
    assert {"type":"instruction","instruction":"Whisper softly"} in sockets[0].messages
    engine.close_stream_context(context)
    assert context.closed and context.socket is None
    assert not engine.synthesize("A closed response must not reopen.")
    assert len(sockets)==1


def test_voice_change_and_new_response_do_not_share_native_state(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module
    sockets=[]
    def connect(*args, **kwargs):
        sock=Socket();sockets.append(sock);return sock
    engine,_=make_engine(monkeypatch)
    monkeypatch.setattr(module.websocket,"create_connection",connect)
    first=engine.create_stream_context();engine.set_stream_context(first)
    assert engine.synthesize("First speaker.")
    engine.set_voice(BreezeVoice("another",voice_id="another"))
    assert engine.synthesize("A different speaker.")
    assert len(sockets)==2 and not engine.last_metrics["session_reused"]
    engine.close_stream_context(first)
    second=engine.create_stream_context();engine.set_stream_context(second)
    assert engine.synthesize("A fresh response.")
    assert len(sockets)==3 and not engine.last_metrics["session_reused"]
    engine.close_stream_context(second)


def test_api_owns_one_context_per_websocket_response(tmp_path):
    engine,client=server_client(tmp_path)
    contexts=[];seen=[]
    def create():
        context={"closed":False};contexts.append(context);return context
    engine.create_stream_context=create
    engine.set_stream_context=lambda context:setattr(engine,"context",context)
    engine.close_stream_context=lambda context:context.update(closed=True)
    original=engine.synthesize
    def synthesize(text):
        seen.append(engine.context)
        return original(text)
    engine.synthesize=synthesize
    with client:
        for index in range(2):
            with client.websocket_connect("/v1/audio/speech-stream") as ws:
                ws.send_json({"type":"config","voice":"test"})
                ws.send_json({"type":"text","text":"I am here with you.","instructions":"Warm"})
                ws.send_json({"type":"text","text":"There is no need to rush.","instructions":"Whisper"})
                ws.send_json({"type":"end"})
                while True:
                    event=ws.receive()
                    if event.get("text") and '"type":"done"' in event["text"]:break
            assert contexts[index]["closed"]
    assert len(contexts)==2
    assert sum(context is contexts[0] for context in seen)>=2
    assert sum(context is contexts[1] for context in seen)>=2
    assert engine.context is None
