# HTTP(S)

[日本語](http.ja.md)

Native programs can send HTTP and HTTPS requests and receive the response **piece by piece as it arrives**.
Each request runs on its own background thread, so none of the functions block (except `httpWait`, which waits up to a given time).
They are meant to be polled once per frame while the program keeps drawing, and they can read AI streaming APIs line by line
(Server-Sent Events from Claude's Messages API or OpenAI-compatible APIs, newline-delimited JSON from Ollama).

The Japanese page ([http.ja.md](http.ja.md)) has the full reference; this is a summary.

## Setup

Nothing to install: the OS library is used.

- **Windows**: WinHTTP (`winhttp.dll`, part of Windows; linked through `winhttp.lib`).
- **macOS**: the system libcurl (`/usr/lib/libcurl.4.dylib`), loaded at run time.
- **Linux**: libcurl (`libcurl.so.4`, or `libcurl-gnutls.so.4`), loaded at run time. If it is missing, the request fails and
  `httpError` explains how to install it (for example `sudo apt install libcurl4`).

TLS certificate checks, redirects (up to 10, never from https to http) and proxy settings follow the OS library's defaults.
On **wasm32** the functions exist, but every request fails at once with a reason.

## Built-in functions

| Function | Meaning |
| --- | --- |
| `httpStart(method, url, headers: string, body: string, timeoutMs: i32): i64` | Starts a request and returns its id. `headers` is `"Name: value"` lines joined with `"\n"`. `timeoutMs` (0 or less: 60 s) limits resolving, connecting, sending and each gap between received data, not the whole transfer. An invalid request still gets an id; it is in the failed state. |
| `httpState(req): i32` | 0 waiting, 1 receiving (status and headers arrived), 2 done, 3 failed, 4 cancelled. |
| `httpStatus(req): i32` | The HTTP status code (0 until the headers arrive). |
| `httpRead(req): string` | Takes all the body bytes that arrived and were not taken yet. |
| `httpReadLine(req): string \| null` | Takes the next line without `"\n"` / `"\r\n"`; `null` until a whole line arrived. After the transfer ends, a last line without a newline is returned too. |
| `httpHeaders(req): string` | The final response's headers, `"Name: value"` lines joined with `"\n"`. |
| `httpError(req): string` | Why the request failed or was cancelled. |
| `httpWait(req, timeoutMs: i32): bool` | Waits until more data arrives or the state changes (since the last `httpWait` / `httpRead`, or a `httpReadLine` that returned `null`). |
| `httpCancel(req)` / `httpClose(req)` | Cancels the request / releases its id (without waiting). Using a closed id stops the program. |

Read the state before the body: when the state is 2 or more, the following `httpRead` / `httpReadLine` calls get everything that is left.
A 4xx or 5xx response whose body arrived is state 2 (done). A connection that closes early is state 3 (failed).

`runtime/http.um` adds `httpPoll(id): HttpState` (phase, status, the new body, `finished`, error) and
`httpRequest(method, url, headers: string[], body, timeoutMs)`.

## Tests

`node tests/run.ts` starts `tests/http/server.mjs` (a Node HTTP server on 127.0.0.1 only) and runs `tests/http/*.um` against it:
whole and streamed bodies (SSE and newline-delimited JSON split in the middle of lines), early disconnects, 404, redirects,
slow responses (timeouts), cancelling, refused connections and invalid requests. HTTPS is not covered by these offline tests.
