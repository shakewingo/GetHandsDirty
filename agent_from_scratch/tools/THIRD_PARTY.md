# Nanobot reference

The general shell command/working-directory interface and web search/readability
approach, plus the filesystem interfaces and document adapters, were adapted from
[nanobot](https://github.com/HKUDS/nanobot), inspected at
local revision `0b1fa0c3e44510e3d34d9bed4d491884cb19de7e`:
`nanobot/agent/tools/shell.py`, `web.py`, `filesystem.py`, and `nanobot/utils/document.py`.

This project keeps its synchronous execution, structured results and trace handling.
It uses `httpx`, `readability-lxml`, `ddgs`, `pypdf`, `python-docx`, `openpyxl`,
`python-pptx` and Pillow directly. It does not import nanobot's runtime, async sessions,
provider catalogue or remote Jina reader. Filesystem adaptation keeps exact replacements
with disambiguation/version checks; automatic fuzzy edits, read deduplication and image
content blocks are not implemented. The current model interface accepts text only.

MIT License

Copyright (c) 2025-present Xubin Ren and the nanobot contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
