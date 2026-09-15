from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import signal
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tools.files import WriteFileTool
from agent_from_scratch.tools.register import ToolRegistry
from agent_from_scratch.tools.web import WebFetchTool


class WebTests(unittest.TestCase):
    """Exercise real HTTP parsing locally; production TLS is checked by the live smoke."""

    @classmethod
    def setUpClass(cls):
        cls.hits = []
        page = (Path(__file__).parent / "fixtures/page.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                cls.hits.append(self.path)
                routes = {"/redirect": "/page", "/loop": "/loop", "/redirect-error": "/disconnect",
                          "/escape": "https://blocked.test/private",
                          "/downgrade": "http://docs.test/page"}
                if self.path in routes:
                    self.send_response(302)
                    self.send_header("Location", routes[self.path])
                    self.end_headers()
                    return
                if self.path == "/slow-headers":
                    sleep(0.3)
                if self.path == "/disconnect":
                    self.close_connection = True
                    return
                body = {"/page": page, "/json": b'{"answer": 42}',
                        "/large": b"x" * 10000}.get(self.path, b"plain text")
                self.send_response(404 if self.path == "/missing" else 200)
                self.send_header("Content-Type", {"/page": "text/html; charset=utf-8",
                                                  "/json": "application/json", "/binary": "image/png"}
                                 .get(self.path, "text/plain"))
                if self.path == "/gzip":
                    self.send_header("Content-Encoding", "gzip")
                self.end_headers()
                try:
                    if self.path == "/drip":
                        for _ in range(30):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            sleep(0.03)
                    else:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.hits.clear()
        self.connections = []

        def local_connection(host, timeout):
            self.assertEqual(host, "docs.test")
            connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=timeout)
            self.connections.append(connection)
            return connection

        transport = patch("agent_from_scratch.tools.web.HTTPSConnection", side_effect=local_connection)
        self.connect = transport.start()
        self.addCleanup(transport.stop)
        self.tool = WebFetchTool({"docs.test"}, timeout=2)

    def fetch(self, path):
        return self.tool.invoke({"url": "https://docs.test" + path}, call_id="fetch-1")

    def test_saved_html_redirect_extracts_visible_text_and_source(self):
        result = self.fetch("/redirect")
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.call_id, "fetch-1")
        self.assertEqual(result.output["url"], "https://docs.test/redirect")
        self.assertEqual(result.output["final_url"], "https://docs.test/page")
        self.assertEqual(result.output["status"], 200)
        self.assertTrue(result.output["untrusted"])
        self.assertIn("2A & tools", result.output["text"])
        self.assertIn("报告文件", result.output["text"])
        self.assertNotIn("do_not_include", result.output["text"])
        self.assertNotIn("secret", result.output["text"])
        self.assertEqual(self.hits, ["/redirect", "/page"])
        self.assertTrue(all(c.sock is None for c in self.connections))

    def test_json_and_plain_text(self):
        self.assertEqual(self.fetch("/json").output["text"], '{"answer": 42}')
        self.assertEqual(self.fetch("/plain").output["text"], "plain text")

    def test_schema_and_policy_deny_before_connecting(self):
        for arguments in ({"url": 4}, {"url": "https://docs.test/", "timeout": 100}, {}):
            self.assertEqual(self.tool.invoke(arguments).error_code, "invalid_arguments")
        for url in ("http://docs.test/", "file:///etc/passwd", "https://blocked.test/",
                    "https://docs.test.evil.test/", "https://user:secret@docs.test/",
                    "https://docs.test:8443/", "https://docs.test:notaport/", "https://[bad/", "https://docs.test/\r\nHeader:x", "https://docs.test\\@evil.test/"):
            self.assertEqual(self.tool.invoke({"url": url}).error_code, "denied", url)
        self.connect.assert_not_called()

    def test_redirect_connection_failure_does_not_attribute_previous_status_to_new_url(self):
        result = self.fetch("/redirect-error")
        self.assertEqual(result.error_code, "execution_error")
        self.assertEqual(result.output["final_url"], "https://docs.test/disconnect")
        self.assertIsNone(result.output["status"])
        self.assertEqual(self.hits, ["/redirect-error", "/disconnect"])

    def test_redirects_cannot_expand_policy_and_loops_are_bounded(self):
        for path in ("/escape", "/downgrade"):
            self.hits.clear()
            result = self.fetch(path)
            self.assertEqual(result.error_code, "denied")
            self.assertEqual(self.hits, [path])
            self.assertEqual(result.output["status"], 302)
        self.hits.clear()
        self.assertFalse(self.fetch("/loop").ok)
        self.assertEqual(len(self.hits), 4)

    def test_download_and_text_caps_are_independent_and_explicit(self):
        self.tool = WebFetchTool({"docs.test"}, max_bytes=32, max_chars=10)
        result = self.fetch("/large")
        self.assertEqual(result.output["text"], "x" * 10)
        self.assertEqual(result.output["bytes_read"], 33)
        self.assertTrue(result.output["download_truncated"])
        self.assertTrue(result.output["text_truncated"])
        self.assertTrue(result.output["truncated"])
        self.tool = WebFetchTool({"docs.test"}, max_bytes=10, max_chars=10)
        self.assertFalse(self.fetch("/plain").output["truncated"])

    def test_http_and_content_failures_are_observations_and_next_call_works(self):
        for path in ("/missing", "/binary", "/gzip"):
            result = self.fetch(path)
            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "execution_error")
            self.assertIn(result.output["status"], (200, 404))
        self.assertTrue(self.fetch("/plain").ok)

    def test_total_deadline_stops_slow_headers_and_continuously_arriving_body(self):
        previous = signal.getsignal(signal.SIGALRM)
        for path in ("/slow-headers", "/drip"):
            self.tool = WebFetchTool({"docs.test"}, timeout=0.12)
            started = monotonic()
            result = self.fetch(path)
            self.assertEqual(result.error_code, "timeout", result.error_message)
            self.assertLess(monotonic() - started, 0.5)
            self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
            self.assertTrue(all(c.sock is None for c in self.connections))
        self.tool = WebFetchTool({"docs.test"})
        self.assertTrue(self.fetch("/plain").ok)

    def test_deadline_includes_dns_and_restores_after_ctrl_c(self):
        with patch("socket.getaddrinfo", side_effect=lambda *a, **k: sleep(1)):
            self.tool = WebFetchTool({"docs.test"}, timeout=0.05)
            self.assertEqual(self.fetch("/plain").error_code, "timeout")
        with patch.object(HTTPConnection, "getresponse", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.fetch("/plain")
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertTrue(all(c.sock is None for c in self.connections))

    def test_existing_timer_is_not_replaced_and_no_request_starts(self):
        signal.setitimer(signal.ITIMER_REAL, 30)
        try:
            result = self.fetch("/plain")
            self.assertFalse(result.ok)
            self.assertIn("existing real-time timer", result.error_message)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 20)
            self.connect.assert_not_called()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def test_web_failure_then_fetch_and_write_through_the_ordinary_loop(self):
        with TemporaryDirectory() as directory:
            model = Mock(spec=LLM)
            model.settings.return_value = {}
            step = 0

            def generate(messages, tools):
                nonlocal step
                step += 1
                if step <= 2:
                    if step == 2:
                        self.assertFalse(json.loads(messages[-1]["content"])["ok"])
                    return LLMResponse("assistant", "", ResponseType.tool_call, tool_name="web_fetch",
                                       tool_params={"url": "https://docs.test/" + ("missing" if step == 1 else "page")})
                observation = json.loads(messages[-1]["content"])
                self.assertTrue(observation["ok"])
                if step == 3:
                    return LLMResponse("assistant", "", ResponseType.tool_call, tool_name="write_file",
                                       tool_params={"path": "note.txt", "content": observation["output"]["text"]})
                return LLMResponse("assistant", "Saved observed page text.", ResponseType.direct)

            model.generate.side_effect = generate
            result = Agent(model, registry=ToolRegistry([self.tool, WriteFileTool(directory)])).run_turn("Fetch and save the page")
            self.assertEqual(result.stop_reason, "final_response")
            self.assertIn("Release: 2A & tools.", (Path(directory) / "note.txt").read_text())
            self.assertEqual([json.loads(m["content"])["ok"] for m in result.messages if m["role"] == "tool"],
                             [False, True, True])


if __name__ == "__main__":
    unittest.main()
