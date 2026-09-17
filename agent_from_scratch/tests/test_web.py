from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import gzip
import json
import signal
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

import httpx

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tools.base import ToolCall, ToolRegistry
from agent_from_scratch.tools.files import WriteFileTool
from agent_from_scratch.tools.web import WebFetchTool, WebSearchTool


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
                body = {"/page": page, "/gzip": gzip.compress(page), "/json": b'{"answer": 42}',
                        "/large": b"x" * 10000}.get(self.path, b"plain text")
                self.send_response({"/missing": 404, "/forbidden": 403}.get(self.path, 200))
                self.send_header("Content-Type", {"/page": "text/html; charset=utf-8",
                                                  "/gzip": "text/html; charset=utf-8",
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

        owner = self
        client_class = httpx.Client

        class LocalTransport(httpx.HTTPTransport):
            closed = False

            def handle_request(self, request):
                owner.assertEqual(request.url.host, "docs.test")
                request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=owner.server.server_port)
                return super().handle_request(request)

            def close(self):
                self.closed = True
                super().close()

            def __exit__(self, *args):
                self.closed = True
                return super().__exit__(*args)

        def local_client(**kwargs):
            connection = LocalTransport()
            self.connections.append(connection)
            return client_class(transport=connection, trust_env=False, **kwargs)

        transport = patch("agent_from_scratch.tools.web.httpx.Client", side_effect=local_client)
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
        self.assertTrue(all(c.closed for c in self.connections))

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
        self.assertEqual(len(self.hits), self.tool.max_redirects + 1)

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
        for path in ("/missing", "/binary", "/forbidden"):
            result = self.fetch(path)
            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "execution_error")
            self.assertIn(result.output["status"], (200, 403, 404))
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
            self.assertTrue(all(c.closed for c in self.connections))
        self.tool = WebFetchTool({"docs.test"})
        self.assertTrue(self.fetch("/plain").ok)

    def test_deadline_includes_dns_and_restores_after_ctrl_c(self):
        with patch("socket.getaddrinfo", side_effect=lambda *a, **k: sleep(1)):
            self.tool = WebFetchTool({"docs.test"}, timeout=0.05)
            self.assertEqual(self.fetch("/plain").error_code, "timeout")
        with patch.object(httpx.HTTPTransport, "handle_request", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.fetch("/plain")
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertTrue(all(c.closed for c in self.connections))

    def test_unrestricted_http_and_compressed_html(self):
        self.tool = WebFetchTool()
        result = self.tool.invoke({"url": "http://docs.test/gzip"})
        self.assertTrue(result.ok, result.error_message)
        self.assertIn("Release: 2A & tools", result.output["text"])
        self.assertIn("报告文件", result.output["text"])
        self.assertEqual(result.output["status"], 200)
        self.assertTrue(all(c.closed for c in self.connections))

    def test_readability_removes_navigation_and_full_text_remains_available(self):
        article = "<h1>Tool design</h1><p>" + "Preserve observed results and explain failures. " * 40 + "</p>"
        html = "<html><body><nav>Unrelated navigation</nav><article>" + article + "</article></body></html>"
        readable, extractor = WebFetchTool._extract_html(html, "readable")
        full, _ = WebFetchTool._extract_html(html, "text")
        self.assertEqual(extractor, "readability")
        self.assertIn("Preserve observed results", readable)
        self.assertNotIn("Unrelated navigation", readable)
        self.assertIn("Unrelated navigation", full)

    def test_extraction_does_not_swallow_the_total_deadline(self):
        with patch("readability.Document", side_effect=TimeoutError("deadline")):
            result = self.fetch("/page")
        self.assertEqual(result.error_code, "timeout")
        self.assertEqual(result.output["status"], 200)
        self.assertTrue(all(c.closed for c in self.connections))

    def test_403_is_a_failed_observation_with_no_fabricated_content(self):
        result = self.fetch("/forbidden")
        self.assertFalse(result.ok)
        self.assertEqual(result.output["status"], 403)
        self.assertEqual(result.output["text"], "")
        self.assertIn("another source", result.error_message)
        self.assertTrue(all(c.closed for c in self.connections))

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
            model.measure_context.return_value = {  # Scripted fitting budget; no real tokenizer.
                "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
                "response_reserve": 512, "remaining_tokens": 7388,
            }
            model.settings.return_value = {}
            step = 0

            def generate(messages, tools):
                nonlocal step
                step += 1
                if step <= 2:
                    if step == 2:
                        self.assertFalse(json.loads(messages[-1]["content"])["ok"])
                    return LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('web_fetch', {'url': 'https://docs.test/' + ('missing' if step == 1 else 'page')})])
                observation = json.loads(messages[-1]["content"])
                self.assertTrue(observation["ok"])
                if step == 3:
                    return LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('write_file', {'path': 'note.txt', 'content': observation['output']['text']})])
                return LLMResponse("assistant", "Saved observed page text.", ResponseType.direct)

            model.generate.side_effect = generate
            result = Agent(model, registry=ToolRegistry([self.tool, WriteFileTool(directory)])).run_turn("Fetch and save the page")
            self.assertEqual(result.stop_reason, "final_response")
            self.assertIn("Release: 2A & tools.", (Path(directory) / "note.txt").read_text())
            self.assertEqual([json.loads(m["content"])["ok"] for m in result.messages if m["role"] == "tool"],
                             [False, True, True])


class SearchTests(unittest.TestCase):
    def test_search_returns_bounded_source_records(self):
        with patch("ddgs.DDGS") as provider:
            provider.return_value.text.return_value = [
                {"title": "Guide", "href": "https://example.com/guide", "body": "x" * 1000},
                {"title": "Other", "href": "https://example.com/other", "body": "Other page"},
            ]
            result = WebSearchTool().invoke({"query": "tool guide", "count": 1})
        self.assertTrue(result.ok, result.error_message)
        self.assertTrue(result.output["untrusted"])
        self.assertEqual(len(result.output["results"]), 1)
        self.assertEqual(result.output["results"][0]["url"], "https://example.com/guide")
        self.assertEqual(len(result.output["results"][0]["snippet"]), 600)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_bad_query_or_count_never_searches(self):
        with patch("ddgs.DDGS") as provider:
            for args in ({"query": ""}, {"query": "x" * 2001}, {"query": "x", "count": 11},
                         {"query": "x", "count": True}):
                self.assertEqual(WebSearchTool().invoke(args).error_code, "invalid_arguments")
            provider.assert_not_called()

    def test_failure_timeout_and_interruption_are_distinct(self):
        from ddgs.exceptions import TimeoutException

        tool = WebSearchTool()
        for error, code in [(RuntimeError("provider unavailable"), "execution_error"),
                            (TimeoutException("deadline"), "timeout"), (TimeoutError("deadline"), "timeout")]:
            with patch("ddgs.DDGS") as provider:
                provider.return_value.text.side_effect = error
                result = tool.invoke({"query": "tool guide"})
            self.assertEqual(result.error_code, code)
            self.assertEqual(result.output["results"], [])
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        with patch("ddgs.DDGS") as provider:
            provider.return_value.text.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                tool.invoke({"query": "tool guide"})
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))


if __name__ == "__main__":
    unittest.main()
