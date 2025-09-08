# ws_manager.py

import asyncio
from typing import Set
from dataclasses import dataclass, field

@dataclass
class WSClient:
    ws: any
    q_text: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    q_bin: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=30))

class WSManager:
    def __init__(self):
        self.clients: Set[WSClient] = set()

    async def register(self, ws):
        client = WSClient(ws)
        self.clients.add(client)
        return client
    
    async def unregister(self, client:WSClient):
        self.clients.discard(client)

    # Safe to call from training thread
    def enqueue_json(self, payload:dict):
        # Literally to queue a json message here
        loop = asyncio.get_running_loop()
        for c in list(self.clients):
            try:
                #payload structure: ["type", "run_id", "step", "reward_last", "reward_mean", "fps", "ts"(timestamp)]
                loop.call_soon_threadsafe(c.q_text.put_nowait, payload)
            except Exception:
                pass
    
    def enqueue_bytes(self, blob:bytes):
        loop = asyncio.get_running_loop()
        for c in list(self.clients):
            try:
                #payload structure -> generally an image
                loop.call_soon_threadsafe(c.q_bin.put_nowait, blob)
            except Exception:
                pass
    
    # Per-client pump: runs in the WS Handler task, basically is like an 
    # event listener that constantly sends stuff outwards
    async def pump(self, client:WSClient):
        try:
            while True:
                # Prioritize text ticks; send at most one of each per loop
                send = False
                if not client.q_text.empty():
                    payload = await client.q_text.get()
                    await client.ws.send_json(payload)
                    send = True
                if not client.q_bin.empty():
                    blob = await client.q_bin.get()
                    await client.ws.send_bytes(blob)
                    send = True
                if not send:
                    await asyncio.sleep(0.1) # yield
        except Exception:
            pass

            


