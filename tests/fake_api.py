"""A local HTTP server that answers from a script: one queue of actions per path."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"


class FakeAPI:
    def __init__(self):
        self.queue = {}
        self.requests = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                api.requests.append((self.path, dict(self.headers), body))
                actions = api.queue.get(self.path) or []
                act = actions.pop(0) if actions else {"status": 500, "body": {"error": "queue empty"}}
                if act.get("delay"):
                    time.sleep(act["delay"])
                raw = act["raw"].encode() if "raw" in act else json.dumps(act.get("body", {})).encode()
                if act.get("trickle"):
                    # headers at once, then a newline every `every` seconds for `for` seconds, then the body:
                    # what a server does to keep a queued request open
                    self.close_connection = True
                    try:
                        self.send_response(act.get("status", 200))
                        self.send_header("content-type", "application/json")
                        self.end_headers()
                        end = time.monotonic() + act["trickle"]["for"]
                        while time.monotonic() < end:
                            self.wfile.write(b"\n")
                            self.wfile.flush()
                            time.sleep(act["trickle"]["every"])
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                try:
                    self.send_response(act.get("status", 200))
                    for k, v in (act.get("headers") or {}).items():
                        self.send_header(k, v)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def push(self, path, *actions):
        self.queue.setdefault(path, []).extend(actions)

    def bodies(self, path=None):
        return [b for p, _, b in self.requests if path is None or p == path]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 200}}


def openai_text(content, finish="stop", usage=None, **msg):
    return {
        "status": 200,
        "body": {"choices": [{"message": {"content": content, **msg}, "finish_reason": finish}], "usage": usage or USAGE},
    }


def openai_tool(args, name="submit", finish="tool_calls", usage=None, call_id="call_1"):
    a = args if isinstance(args, str) else json.dumps(args)
    msg = {"content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": a}}]}
    return {"status": 200, "body": {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage or USAGE}}


def anthropic_tool(obj, name="submit", stop_reason="tool_use"):
    return {
        "status": 200,
        "body": {
            "content": [{"type": "tool_use", "id": "toolu_1", "name": name, "input": obj}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1000, "output_tokens": 80},
        },
    }


def anthropic_text(text, stop_reason="end_turn"):
    return {
        "status": 200,
        "body": {
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1000, "output_tokens": 80, "cache_read_input_tokens": 500},
        },
    }
