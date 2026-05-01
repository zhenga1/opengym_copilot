# ws_manager.py

import asyncio
from typing import Set, Optional
from dataclasses import dataclass, field

import threading

@dataclass(eq=False)
class WSClient:
    ws: any
    topic: str ## Training or Rollout or Others, to be developed and used in future. Currently set just to "training" only
    run_id: Optional[str] = None
    q_text = asyncio.Queue()#dict = field(default_factory=lambda: {})
    q_bin = asyncio.Queue() #dict = field(default_factory=lambda: {})


# pumps stuff from q_text and q_bin sequentially to the websoocket frontend
class PumpingService:
    async def pump(self, client: WSClient, stop: asyncio.Event | None = None):
        """Continuously send items from the two queues to the websocket."""
        stop = stop or asyncio.Event()
        print("Begin executing the pumping service. ")
        try:
            while not stop.is_set():
                sent_any = False

                # Try text first (non-blocking)
                # try and if raise error
                # then go to except
                # otherwise go to else
                try:
                    payload = client.q_text.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    print("start send payload")
                    await client.ws.send_json(payload)
                    sent_any = True

                # Then binary (non-blocking)
                try:
                    blob = client.q_bin.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    print("start send payload")
                    await client.ws.send_bytes(blob)
                    sent_any = True

                # Nothing to send? Yield briefly to avoid 100% CPU.
                if not sent_any:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            # Task was cancelled (shutdown). Let it exit cleanly.
            pass

delay_seconds = 3
class WSManager:
    def __init__(self, loop):
        self.clients: Set[WSClient] = set()
        #self.threads = []
        self.loop = loop
        self.stops = []
        self.svcs = []
        self.pumping_tasks = []
        #self.client_to_ws_socket = field(default_factory=lambda: {})
    
    def mark_loop_thread(self):
        self._loop_thread_id = threading.get_ident()
        
    async def register(self, ws, topic:str, run_id:Optional[str]):
        client = WSClient(ws=ws, topic=topic, run_id=run_id)
        # self.thread = threading.Thread(target=self.pump, args=(client,), daemon=True)
        # self.thread.start()
        # self.threads.append(self.thread)
        self.clients.add(client)
        stop = asyncio.Event()
        svc = PumpingService()
        pump_task = asyncio.create_task(svc.pump(client, stop))  # background
        self.stops.append(stop)
        self.svcs.append(svc)
        self.pumping_tasks.append(pump_task)
        # self.client_to_ws_socket[client] = ws
        return client

    async def unregister(self, client:WSClient):
        self.clients.discard(client)
        self.stops[self.clients.index(client)].set()
        pump_task = self.pumping_tasks[self.clients.index(client)]
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
        #self.threads[self.clients.index(client)].join()

    def enqueue_json(self, topic:str, payload:dict, run_id:Optional[str]=None):
        # Literally to queue a json message here
        # loop = asyncio.get_running_loop()
        if self.loop.is_closed():
            print("Loop is closed, kill process")
            return
        for c in list(self.clients):
            if c.topic == topic and (run_id is None or c.run_id == run_id):
                #payload structure: ["type", "run_id", "step", "reward_last", "reward_mean", "fps", "ts"(timestamp)]
                if threading.get_ident() == self._loop_thread_id:
                    try:
                        print("Sending Payload from queue q_text to the same thread")
                        c.q_text.put_nowait(payload)
                    except Exception as e:
                        print("Exception in sending payload: ", e)
                        pass
                else:
                    try:
                        print("Sending Payload from queue q_text to a different thread")
                        self.loop.call_soon_threadsafe(c.q_text.put_nowait, payload)
                        #c.ws.send_json(payload)
                    except Exception as e:
                        print("Exception in sending payload: ", e)
                        pass
    def enqueue_bytes(self, topic:str, blob:bytes, run_id:Optional[str]=None):
        #loop = asyncio.get_running_loop()
        for c in list(self.clients):
            if c.topic == topic and (run_id is None or c.run_id == run_id):
                if threading.get_ident() == self._loop_thread_id:
                    try:
                        #payload structure -> generally an image
                        c.q_bin.put_nowait(blob)
                        #await c.ws.send_bytes(blob)
                    except Exception:
                        pass
                else:
                    try:
                        #payload structure -> generally an image
                        self.loop.call_soon_threadsafe(c.q_bin.put_nowait, blob)
                        #await c.ws.send_bytes(blob)
                    except Exception:
                        pass


    # async def pump(self, client:WSClient):
    #     try:
    #         while True:
    #             send = False
    #             if not client.q_text.empty():
    #                 print("Awaiting Payload from queue q_text")
    #                 payload = await client.q_text.get()
    #                 print("Awaiting send payload from queue q_text")
    #                 await client.ws.send_json(payload)
    #                 send = True
    #             if not client.q_bin.empty():
    #                 print("Awaiting Payload from queue q_bin")
    #                 blob = await client.q_bin.get()
    #                 print("Awaiting send payload from queue q_bin")
    #                 await client.ws.send_bytes(blob)
    #                 send = True
    #             if not send:
    #                 print("Yielding")
    #                 await asyncio.sleep(1) # yield
                
    #     except Exception:
    #         pass
    

