# HTTP(S)

[English](http.md)

ネイティブのプログラムは、HTTP と HTTPS の要求を送り、応答を**届いた分から少しずつ**受け取れます。
送受信は要求ごとの裏のスレッドで行うので、どの関数も待ちません（`httpWait` だけは、指定した時間まで待てます）。
画面を描きながら、毎コマ届いた分を取り出す使い方を前提にしています。AI の API のストリーミング
（Claude の Messages API の SSE、Ollama の改行区切りの JSON、OpenAI 互換の API の SSE）を、1行ずつ読めます。

`import { httpRequest, httpPoll, httpReadLine } from "um:http";` のように、ライブラリ `um:http` から読み込んで使います（[ライブラリ](libraries.ja.md)）。

## 用意するもの

新しいライブラリは要りません。OS に入っているものを使います。

| OS | 使うもの |
| --- | --- |
| Windows | WinHTTP（`winhttp.dll`。Windows に入っている）。リンクのときに `winhttp.lib` を使います。 |
| macOS | OS の libcurl（`/usr/lib/libcurl.4.dylib`）を、実行時に読み込みます。 |
| Linux | libcurl（`libcurl.so.4`。なければ `libcurl-gnutls.so.4`）を、実行時に読み込みます。ほとんどの配布物に入っています。なければ `sudo apt install libcurl4` などで入れます。 |

libcurl はリンクせずに実行時に読み込むので、ビルドする機械に libcurl の開発用のファイルは要りません。
実行する機械で見つからなければ、`httpStart` の要求が失敗の状態になり、`httpError` が理由（入れ方）を返します。
HTTP の組み込みを使わないプログラムには、何もリンクされません。

TLS の証明書の検証、リダイレクトの追跡（10 回まで。https から http へは移りません）、代理（プロキシ）の設定は、
OS のライブラリの既定のままです。圧縮（`Accept-Encoding`）は頼まないので、本文はサーバーが送ったまま届きます。

**wasm32 では、まだ使えません。** 同じソースをコンパイルできるように関数はありますが、`httpStart` の要求はすぐに失敗の状態（3）になり、
`httpError` が `"HTTP requests are not available in wasm32 programs yet (build with --target host)"` を返します。

## 例

```ts
// Claude の Messages API（SSE）を、届いた分から表示する。
const headers = "x-api-key: " + envVar("ANTHROPIC_API_KEY") + "\n" +
  "anthropic-version: 2023-06-01\n" +
  "content-type: application/json";
const body = "{\"model\":\"...\",\"max_tokens\":1024,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"こんにちは\"}]}";
const id = httpStart("POST", "https://api.anthropic.com/v1/messages", headers, body, 30000);

// 毎コマ（待たない）:
const state = httpState(id);           // 先に状態を読む
while (true) {
  const line = httpReadLine(id);       // "event: ..." / "data: {...}" / ""（出来事の区切り）
  if (line == null) { break; }         // まだ1行そろっていない
  // line が "data: " で始まれば、その JSON から文字を取り出して表示に足す
}
if (state >= 2) {                      // 2 受け取り終えた、3 失敗、4 中止
  if (state != 2 || httpStatus(id) != 200) { eprint(`${httpStatus(id)} ${httpError(id)}`); }
  httpClose(id);
}
```

- 状態を読んでから本文を読むと、状態が 2 以上なら、その後の `httpRead` / `httpReadLine` で残りをすべて取り出せます
  （読む順を逆にすると、状態を読む間に届いた分を取りこぼすことがあります）。
- Ollama（`http://localhost:11434/api/chat`）は、1行が1つの JSON です。最後の行が改行で終わらなくても、受け取り終えたあとの
  `httpReadLine` が1行として返します。
- HTTP の状態コードが 4xx / 5xx でも、本文を受け取れたなら状態は 2（受け取り終えた）です。API の誤りの説明は本文の JSON にあります。

## 組み込み

| 関数 | 動作 |
| --- | --- |
| `httpStart(method: string, url: string, headers: string, body: string, timeoutMs: i32): i64` | 要求を始め、番号を返します。`headers` は `"名前: 値"` を `"\n"` でつないだもの（行末の `"\r"` と空の行は無視）。`body` は送る本文（空なら送りません）。`timeoutMs` は待ちの上限で、0 以下なら 60 秒。始められないとき（正しくない URL・方法・頭、`http://` / `https://` 以外の URL、libcurl がない など）も番号を返し、要求は失敗の状態になります。 |
| `httpState(req: i64): i32` | 0 応答待ち（接続・送信・応答の頭を待っている）、1 受信中（状態コードと頭が届いた）、2 受け取り終えた、3 失敗、4 中止。 |
| `httpStatus(req: i64): i32` | HTTP の状態コード（200、404 など）。応答の頭が届くまでは 0。リダイレクトをたどったときは最後の応答のもの。 |
| `httpRead(req: i64): string` | 届いて、まだ取り出していない本文をすべて取り出します（なければ空）。バイト列のまま返すので、UTF-8 の文字の途中で切れていることがあります。 |
| `httpReadLine(req: i64): string \| null` | 次の1行を取り出します。末尾の `"\n"` と、その前の `"\r"` 1つは含みません。1行そろっていなければ `null`。受け取りが終わっていれば、改行で終わらない最後の部分も1行として返します。空の行は `""`（SSE の出来事の区切り）。 |
| `httpHeaders(req: i64): string` | 最後の応答の頭（`"名前: 値"` を `"\n"` でつないだもの。状態の行は含まない）。名前の大文字・小文字は、サーバーが送ったまま（HTTP/2 では小文字）です。 |
| `httpError(req: i64): string` | 失敗・中止の理由（英語。OS のライブラリの説明を含みます）。それ以外は空。 |
| `httpWait(req: i64, timeoutMs: i32): bool` | 前に `httpWait` か `httpRead` を呼んだとき（か `httpReadLine` が `null` を返したとき）から、本文が届くか状態が変わるまで、`timeoutMs` ミリ秒まで待ちます（0 は待たない、負は来るまで）。変わったら `true`。受け取りが終わっていて、何も変わっていなければ、すぐ `false`。コンソールのプログラムで、空回りせずに待つのに使います。 |
| `httpCancel(req: i64)` | 中止します。少しあとで状態が 4 になります（それまでに届いた本文は読めます）。終わった要求には何もしません。 |
| `httpClose(req: i64)` | 番号を手放します。動いていれば中止し、待たずに戻ります（裏のスレッドが終わるときに後始末します）。 |

- **どれもどのスレッドからでも呼べます。** 同じ要求を複数のスレッドから読むと、行がどちらに渡るかは決まりません。
- **閉じた番号や、`httpStart` が返していない番号を渡すと停止します**（`panic: invalid HTTP call (an unknown or closed request)`）。
- **同時に持てる要求は 4096 まで**です。終わった要求も `httpClose` するまで残ります。
- **待ちの上限（`timeoutMs`）は、名前の解決・接続・送信と、受信で次のデータが届くまでの間隔に効きます。** 全体の時間には上限を付けません
  （AI の長い応答を、届くかぎり受け取り続けるため）。libcurl では受信の間隔を秒単位に切り上げて調べます。
  全体の時間を区切りたいときは、プログラムが時計（`clock()`）を見て `httpCancel` を呼びます。
- **途中で切れた応答は失敗（3）です。** `Content-Length` より短いところで切れたときや、チャンクの終わりの印の前で切れたときなど。
  それまでに届いた本文は読めます。
- **中止の速さ**: Windows はすぐに止まります。libcurl は、受信の待ちの最中なら 1 秒ほどで止まります。
- **本文は、取り出すまで要求の中に溜まります。** 大きなファイルを受け取るときは、こまめに `httpRead` します。
- 同じ機械の外へ出ない試験のため、`http://127.0.0.1:番号` / `http://localhost:番号` にも送れます（代理は OS の設定に従います）。

## ライブラリの型と関数（um:http。lib/http.um）

`httpPoll` などの名前を使うと、次のものが加わります。

```ts
enum HttpPhase { Waiting, Receiving, Done, Failed, Cancelled }

struct HttpState {
  phase: HttpPhase;
  status: i32;       // HTTP の状態コード（応答の頭が届くまでは 0）
  body: string;      // 前回の httpPoll（か httpRead）から後に届いた本文
  finished: bool;    // もう何も届かない（Done / Failed / Cancelled で、本文を取り出し終えた）
  error: string;     // Failed / Cancelled の理由
}

function httpPoll(id: i64): HttpState                // 状態を調べ、届いた本文を取り出す。待たない
function httpRequest(method: string, url: string, headers: string[], body: string, timeoutMs: i32): i64
                                                     // 頭を配列で渡す httpStart
```

```ts
import { httpRequest, httpPoll } from "um:http";
const id = httpRequest("POST", "http://localhost:11434/api/chat", ["Content-Type: application/json"], body, 0);
// 毎コマ:
const s = httpPoll(id);
text = text + s.body;
if (s.finished) {
  match (s.phase) {
    HttpPhase.Done => { }
    HttpPhase.Failed => { eprint(s.error); }
    HttpPhase.Cancelled => { }
    HttpPhase.Waiting => { }
    HttpPhase.Receiving => { }
  }
  httpClose(id);
}
```

## 試験

`node tests/run.ts` は、`tests/http/server.mjs`（Node の `http` で作った、127.0.0.1 だけで待つサーバー）を立て、
`tests/http/*.um` をネイティブで実行します（外へは出ません）。まとめて返す・少しずつ返す（SSE、改行区切りの JSON。行の途中で区切る）・
途中で切る・404・リダイレクト・遅い応答（時間切れ）・中止・つながらない・正しくない要求を確かめます。
AddressSanitizer が使えればメモリ検査つきでも実行します。HTTPS は、外につながない試験では確かめていません
（TLS は OS のライブラリがそのまま行います）。
