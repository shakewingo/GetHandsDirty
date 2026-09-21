"""Synchronous web search and readable HTTP(S) fetch, inspired by nanobot's web tools."""

from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
import math
import signal
from threading import current_thread, main_thread
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .base import Tool, ToolErrorCode, ToolExecutionError, ToolInterrupted


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"


@contextmanager
def fetch_deadline(seconds: float):
    """One wall-clock budget across DNS, headers, redirects and body reads.

    Socket timeouts alone can be extended indefinitely by a slowly arriving body.
    This synchronous local CLI owns SIGALRM only during fetch; it does not run in workers.
    """
    if current_thread() is not main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("Web tools require the main POSIX thread for their deadline.")
    if any(signal.getitimer(signal.ITIMER_REAL)):
        raise RuntimeError("Web tools cannot replace an existing real-time timer.")

    def expired(signum, frame):
        raise TimeoutError(f"Fetch exceeded its {seconds}s total deadline.")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class _PageText(HTMLParser):
    """Extract visible text only; no script execution or secondary resource fetches."""

    def __init__(self, *, omit_navigation: bool = False):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []
        self.hidden_tags = {"script", "style", "template", "noscript", "head"}
        if omit_navigation:
            self.hidden_tags.update({"nav", "footer", "aside"})

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden_tags:
            self.hidden.append(tag)
        elif not self.hidden and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.hidden:
            self.hidden = self.hidden[:self.hidden.index(tag)]
        elif not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class WebFetchTool(Tool):
    name = "web_fetch"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "extract_mode": {"type": "string", "enum": ["readable", "text"],
                             "description": "readable extracts the main content; text keeps all visible page text."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, allowed_hosts: set[str] | frozenset[str] | None = None, *, timeout: float = 20,
                 max_bytes: int = 262144, max_chars: int = 4096, max_redirects: int = 5):
        self.allowed_hosts = (frozenset(host.lower() for host in allowed_hosts)
                              if allowed_hosts is not None else None)
        if (not math.isfinite(timeout) or timeout <= 0 or max_bytes < 1 or max_chars < 1
                or not 0 <= max_redirects <= 10
                or any(not host or any(c in host for c in "/:@?# ") for host in (self.allowed_hosts or ()))):
            raise ValueError("Use exact hostnames, positive limits, and 0–10 redirects.")
        self.timeout, self.max_bytes, self.max_chars = timeout, max_bytes, max_chars
        self.max_redirects = max_redirects
        self.description = (
            "Fetch an HTTP or HTTPS URL and extract readable text, HTML or JSON. "
            "Use web_search to discover URLs. Returns source URL, retrieval time, HTTP status, "
            "untrusted text and truncation flags. Partial text is not a full page; readable "
            "extraction may omit content, so use extract_mode=text for all visible page text. "
            "Does not execute JavaScript or sign into websites. "
            + ("Hosts are unrestricted." if self.allowed_hosts is None else
               f"Only HTTPS port 443 on these hosts: {', '.join(sorted(self.allowed_hosts)) or '(none)'}.")
        )

    def _validate_url(self, url: str) -> str:
        if len(url) > 8192 or any(ord(c) <= 32 or ord(c) == 127 for c in url) or "\\" in url:
            raise ToolExecutionError(ToolErrorCode.DENIED, "URL contains whitespace, controls, or is too long.")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as error:
            raise ToolExecutionError(ToolErrorCode.DENIED, "Malformed URL authority or port.") from error
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ToolExecutionError(ToolErrorCode.DENIED,
                                     "Use an HTTP(S) URL with a hostname and no embedded credentials.")
        if self.allowed_hosts is not None and (parsed.scheme != "https" or port not in (None, 443)
                                               or parsed.hostname not in self.allowed_hosts):
            raise ToolExecutionError(ToolErrorCode.DENIED,
                                     "Only configured HTTPS hosts on port 443 are allowed.")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))

    @staticmethod
    def _extract_html(text: str, mode: str) -> tuple[str, str]:
        extractor = "html"
        if mode == "readable":
            from readability import Document

            try:
                document = Document(text)
                summary = document.summary()
                if summary.strip():
                    text = summary
                    extractor = "readability"
            except TimeoutError:
                raise
            except Exception:
                pass  # Malformed/unsuitable HTML still has a visible-text fallback.
        page = _PageText(omit_navigation=mode == "readable")
        page.feed(text)
        page.close()
        return "\n".join(line.strip() for line in "".join(page.parts).splitlines() if line.strip()), extractor

    def execute(self, url: str, extract_mode: str = "readable") -> dict:
        current = self._validate_url(url)
        result = {"url": url, "final_url": current, "status": None,
                  "text": "", "truncated": False, "untrusted": True,
                  "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            with fetch_deadline(self.timeout), httpx.Client(
                timeout=self.timeout, follow_redirects=False,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            ) as client:
                for hop in range(self.max_redirects + 1):
                    current = self._validate_url(current)
                    result["final_url"] = current
                    result["status"] = None
                    with client.stream("GET", current) as response:
                        result["status"] = response.status_code
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location or hop == self.max_redirects:
                                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                         "Missing redirect location or redirect limit reached.")
                            current = self._validate_url(urljoin(current, location))
                            continue
                        if not 200 <= response.status_code < 300:
                            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                     f"HTTP status {response.status_code}. No page content retrieved. "
                                                     "Use another source or report that the information could not be verified.")
                        content_type = response.headers.get("content-type", "text/plain").split(";", 1)[0].strip().lower()
                        if not (content_type.startswith("text/") or content_type == "application/json"
                                or content_type.endswith("+json")):
                            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                     f"Unsupported content type: {content_type}.")
                        # Retain at most the decoded-body cap plus one lookahead byte.
                        raw = bytearray()
                        for chunk in response.iter_bytes(chunk_size=min(8192, self.max_bytes + 1)):
                            raw.extend(chunk[:self.max_bytes + 1 - len(raw)])
                            if len(raw) > self.max_bytes:
                                break
                        download_truncated = len(raw) > self.max_bytes
                        charset = response.encoding or "utf-8"
                        text = raw[:self.max_bytes].decode(charset, errors="replace")
                        extractor = "raw"
                        if content_type == "text/html":
                            text, extractor = self._extract_html(text, extract_mode)
                        result.update(content_type=content_type, bytes_read=len(raw),
                                      extractor=extractor,
                                      download_truncated=download_truncated,
                                      text_truncated=len(text) > self.max_chars,
                                      truncated=download_truncated or len(text) > self.max_chars,
                                      text=text[:self.max_chars])
                        return result
        except KeyboardInterrupt as error:
            result["interrupted"] = True
            raise ToolInterrupted(result) from error
        except (TimeoutError, httpx.TimeoutException) as error:
            raise ToolExecutionError(ToolErrorCode.TIMEOUT, str(error), result) from error
        except ToolExecutionError as error:
            error.output = result
            raise
        except (OSError, httpx.HTTPError, LookupError) as error:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"{type(error).__name__}: {error}", result) from error
        raise RuntimeError("Fetch ended without a response.")


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for a query. Returns untrusted titles, URLs and short snippets; "
        "use web_fetch to inspect a source. Search snippets can be stale and do not verify live facts."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"},
                       "count": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["query"], "additionalProperties": False,
    }

    def __init__(self, timeout: float = 20):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Search timeout must be positive and finite.")
        self.timeout = timeout

    def execute(self, query: str, count: int = 5) -> dict:
        from ddgs import DDGS
        from ddgs.exceptions import TimeoutException

        if not query.strip() or len(query) > 2000:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "Use a nonempty query up to 2000 characters.")
        result = {"query": query, "provider": "ddgs", "results": [], "untrusted": True,
                  "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            with fetch_deadline(self.timeout):
                rows = DDGS(timeout=self.timeout).text(query, max_results=count)
            result["results"] = [{"title": str(row.get("title", ""))[:300],
                                  "url": str(row.get("href", ""))[:2048],
                                  "snippet": str(row.get("body", ""))[:600]} for row in rows[:count]]
            return result
        except KeyboardInterrupt as error:
            result["interrupted"] = True
            raise ToolInterrupted(result) from error
        except (TimeoutError, TimeoutException) as error:
            raise ToolExecutionError(ToolErrorCode.TIMEOUT, str(error), result) from error
        except Exception as error:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"Search failed: {type(error).__name__}: {error}", result) from error
