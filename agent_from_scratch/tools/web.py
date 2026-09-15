"""Small synchronous HTTPS fetcher for configured hosts, on the main POSIX thread."""

from contextlib import contextmanager
from html.parser import HTMLParser
from http.client import HTTPException, HTTPSConnection
import math
import signal
from threading import current_thread, main_thread
from urllib.parse import urljoin, urlsplit, urlunsplit

from .base import Tool, ToolErrorCode, ToolExecutionError, ToolInterrupted


@contextmanager
def fetch_deadline(seconds: float):
    """One wall-clock budget across DNS, headers, redirects and body reads.

    Socket timeouts alone can be extended indefinitely by a slowly arriving body.
    This synchronous local CLI owns SIGALRM only during fetch; it does not run in workers.
    """
    if current_thread() is not main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("web_fetch requires the main POSIX thread for its deadline.")
    if any(signal.getitimer(signal.ITIMER_REAL)):
        raise RuntimeError("web_fetch cannot replace an existing real-time timer.")

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

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "template", "noscript", "head"}:
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
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, allowed_hosts: set[str] | frozenset[str], *, timeout: float = 15,
                 max_bytes: int = 65536, max_chars: int = 4096, max_redirects: int = 3):
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        if (not math.isfinite(timeout) or timeout <= 0 or max_bytes < 1 or max_chars < 1
                or not 0 <= max_redirects <= 10
                or any(not host or any(c in host for c in "/:@?# ") for host in self.allowed_hosts)):
            raise ValueError("Use exact hostnames, positive limits, and 0–10 redirects.")
        self.timeout, self.max_bytes, self.max_chars = timeout, max_bytes, max_chars
        self.max_redirects = max_redirects
        self.description = (
            "Fetch text/HTML/JSON from an allowed HTTPS URL. Returns source URL, HTTP status, "
            "untrusted page text and truncation flags; partial text is not a full page. "
            f"Allowed hosts: {', '.join(sorted(self.allowed_hosts)) or '(none)'}."
        )

    def _validate_url(self, url: str) -> str:
        if len(url) > 8192 or any(ord(c) <= 32 or ord(c) == 127 for c in url) or "\\" in url:
            raise ToolExecutionError(ToolErrorCode.DENIED, "URL contains whitespace, controls, or is too long.")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as error:
            raise ToolExecutionError(ToolErrorCode.DENIED, "Malformed URL authority or port.") from error
        if (parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts
                or parsed.username is not None or parsed.password is not None
                or port not in (None, 443)):
            raise ToolExecutionError(ToolErrorCode.DENIED,
                                     "Only configured HTTPS hosts on port 443, without credentials, are allowed.")
        return urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))

    def execute(self, url: str) -> dict:
        current = self._validate_url(url)
        result = {"url": url, "final_url": current, "status": None,
                  "text": "", "truncated": False, "untrusted": True}
        try:
            with fetch_deadline(self.timeout):
                for hop in range(self.max_redirects + 1):
                    current = self._validate_url(current)
                    result["final_url"] = current
                    result["status"] = None
                    parsed = urlsplit(current)
                    assert parsed.hostname is not None  # _validate_url checked the exact allowed host.
                    connection = HTTPSConnection(parsed.hostname, timeout=self.timeout)
                    try:
                        connection.request("GET", urlunsplit(("", "", parsed.path, parsed.query, "")),
                                           headers={"Accept-Encoding": "identity", "User-Agent": "tiny-agent/0.1"})
                        with connection.getresponse() as response:
                            result["status"] = response.status
                            if response.status in {301, 302, 303, 307, 308}:
                                location = response.getheader("Location")
                                if not location or hop == self.max_redirects:
                                    raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                             "Missing redirect location or redirect limit reached.")
                                current = self._validate_url(urljoin(current, location))
                                continue
                            if not 200 <= response.status < 300:
                                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                         f"HTTP status {response.status}.")
                            content_type = response.headers.get_content_type()
                            if not (content_type.startswith("text/") or content_type == "application/json"
                                    or content_type.endswith("+json")):
                                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                         f"Unsupported content type: {content_type}.")
                            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                                         "Compressed responses are unsupported; requested identity.")
                            # One lookahead byte distinguishes an exact-size body from a truncated one.
                            raw = response.read(self.max_bytes + 1)
                            download_truncated = len(raw) > self.max_bytes
                            charset = response.headers.get_content_charset() or "utf-8"
                            text = raw[:self.max_bytes].decode(charset, errors="replace")
                            if content_type == "text/html":
                                page = _PageText()
                                page.feed(text)
                                page.close()
                                text = "\n".join(line.strip() for line in "".join(page.parts).splitlines() if line.strip())
                            result.update(content_type=content_type, bytes_read=len(raw),
                                          download_truncated=download_truncated,
                                          text_truncated=len(text) > self.max_chars,
                                          truncated=download_truncated or len(text) > self.max_chars,
                                          text=text[:self.max_chars])
                            return result
                    finally:
                        connection.close()
        except KeyboardInterrupt as error:
            result["interrupted"] = True
            raise ToolInterrupted(result) from error
        except TimeoutError as error:
            raise ToolExecutionError(ToolErrorCode.TIMEOUT, str(error), result) from error
        except ToolExecutionError as error:
            error.output = result
            raise
        except (OSError, HTTPException, LookupError) as error:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"{type(error).__name__}: {error}", result) from error
        raise RuntimeError("Fetch ended without a response.")
