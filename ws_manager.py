# ws_manager.py

import asyncio
from typing import Set, Optional
from dataclasses import dataclass, field

# Identity equality so hashable
@dataclass(eq=False)
class WSClient:
    ws: any
    topic: str     ## Training or Rollout or Others, to be developed and used in future. Currently set just to "training" only
    run_id: Optional[str] = None
    q_text: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    q_bin: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=30))

class WSManager:
    def __init__(self):
        self.clients: Set[WSClient] = set()

    async def register(self, ws, topic:str, run_id:Optional[str]):
        client = WSClient(ws=ws, topic=topic, run_id=run_id)
        self.clients.add(client)
        return client
    
    async def unregister(self, client:WSClient):
        self.clients.discard(client)

    # Safe to call from training thread
    def enqueue_json(self, topic:str, payload:dict, run_id:Optional[str]=None):
        # Literally to queue a json message here
        loop = asyncio.get_running_loop()
        for c in list(self.clients):
            if c.topic == topic and (run_id is None or c.run_id == run_id):
                try:
                    #payload structure: ["type", "run_id", "step", "reward_last", "reward_mean", "fps", "ts"(timestamp)]
                    loop.call_soon_threadsafe(c.q_text.put_nowait, payload)
                    
                except Exception:
                    pass
    
    def enqueue_bytes(self, topic:str, blob:bytes, run_id:Optional[str]=None):
        loop = asyncio.get_running_loop()
        for c in list(self.clients):
            if c.topic == topic and (run_id is None or c.run_id == run_id):
                try:
                    #payload structure -> generally an image
                    loop.call_soon_threadsafe(c.q_bin.put_nowait, blob)
                except Exception:
                    pass
    
    # Per-client pump: runs in the WS Handler task, basically is like an 
    # event listener that constantly sends stuff outwards

    # Not used right
    async def pump(self, client:WSClient):
        try:
            while True:
                # Prioritize text ticks; send at most one of each per loop
                send = False
                if not client.q_text.empty():
                    print("Obtaining Payload from queue q_text")
                    payload = await client.q_text.get()
                    print("OBTAINED Payload from queue q_text")
                    print("Sending Payload from queue q_text")
                    await client.ws.send_json(payload)
                    print("Sent Payload from queue q_text")
                    send = True
                if not client.q_bin.empty():
                    blob = await client.q_bin.get()
                    await client.ws.send_bytes(blob)
                    send = True
                if not send:
                    await asyncio.sleep(0.1) # yield
        except Exception:
            pass

            


