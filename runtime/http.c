// HTTP(S) の組み込み（httpStart / httpState / httpStatus / httpRead / httpReadLine / httpHeaders / httpError /
// httpWait / httpCancel / httpClose）の実装。プログラムが HTTP の組み込みを使ったときだけ、ドライバが native.c と一緒にリンクする。
//
// - Windows は OS の WinHTTP（winhttp.dll）を使う。macOS と Linux は OS の libcurl を実行時に読み込む
//   （macOS は /usr/lib/libcurl.4.dylib、Linux は libcurl.so.4 など）。見つからなければ、要求は失敗の状態になり、
//   理由（httpError）に分かりやすい説明を入れる。新しいライブラリは要らない。
// - 要求ごとに裏のスレッドを1つ起こし、そこで送受信する。届いた本文は要求の入れ物に溜め、プログラムは
//   httpRead / httpReadLine で少しずつ取り出す（画面を止めない。SSE や改行区切りの JSON を行ごとに読める）。
// - 要求は 64 ビットの番号で表す（表の場所と世代を詰めてある）。閉じた番号を渡すと停止する（22）。
// - TLS の検証・リダイレクト（10 回まで。https から http へは移らない）は OS のライブラリの既定のまま有効。
// - timeoutMs は「待ちの上限」: 名前の解決・接続・送信と、受信で次のデータが届くまでの間隔に効く（0 以下なら 60 秒）。
//   全体の時間には上限を付けない（AI の長い応答を、届くかぎり受け取り続けるため）。
// - httpClose は待たない。裏のスレッドがまだ動いていれば中止を伝え、スレッドは終わるときに自分で後始末する。

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <strings.h>
#include <time.h>
#endif

#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Noreturn void um_panic(int32_t code, int32_t line);
void *um_alloc(size_t count, size_t elem_size);

// um_panic の原因コード（native.c の表と src/codegen.ts の並びと揃えること）。
#define UM_PANIC_HTTP_HANDLE 22

// 要求の状態（runtime/http.um の HttpPhase と揃える）。
#define ST_WAITING 0    // 接続・送信・応答の頭を待っている
#define ST_RECEIVING 1  // 状態コードと頭が届き、本文を受け取っている
#define ST_DONE 2       // 本文をすべて受け取った（HTTP の状態コードが 404 などでも、受け取れたなら DONE）
#define ST_FAILED 3     // 接続できない・途中で切れた・時間切れなど（理由は httpError）
#define ST_CANCELLED 4  // httpCancel か httpClose で中止した

#define KIND_REQUEST 3
#define MAX_SLOTS 4096
#define DEFAULT_TIMEOUT_MS 60000

// ---- 鍵と条件変数 ----

#ifdef _WIN32
typedef SRWLOCK um_lock;
typedef CONDITION_VARIABLE um_cond;
#define UM_LOCK_INIT SRWLOCK_INIT
static void lock_init(um_lock *m) { InitializeSRWLock(m); }
static void lock(um_lock *m) { AcquireSRWLockExclusive(m); }
static void unlock(um_lock *m) { ReleaseSRWLockExclusive(m); }
static void cond_init(um_cond *c) { InitializeConditionVariable(c); }
static void cond_broadcast(um_cond *c) { WakeAllConditionVariable(c); }
// ミリ秒まで待つ（負なら来るまで）。
static void cond_wait_ms(um_cond *c, um_lock *m, int64_t ms) { SleepConditionVariableSRW(c, m, ms < 0 ? INFINITE : (DWORD)ms, 0); }
static int64_t now_ms(void) { return (int64_t)GetTickCount64(); }
#else
typedef pthread_mutex_t um_lock;
typedef pthread_cond_t um_cond;
#define UM_LOCK_INIT PTHREAD_MUTEX_INITIALIZER
static void lock_init(um_lock *m) { pthread_mutex_init(m, NULL); }
static void lock(um_lock *m) { pthread_mutex_lock(m); }
static void unlock(um_lock *m) { pthread_mutex_unlock(m); }
static void cond_init(um_cond *c) { pthread_cond_init(c, NULL); }
static void cond_broadcast(um_cond *c) { pthread_cond_broadcast(c); }
static void cond_wait_ms(um_cond *c, um_lock *m, int64_t ms) {
  if (ms < 0) {
    pthread_cond_wait(c, m);
    return;
  }
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  t.tv_sec += ms / 1000;
  t.tv_nsec += (long)(ms % 1000) * 1000000L;
  if (t.tv_nsec >= 1000000000L) {
    t.tv_sec++;
    t.tv_nsec -= 1000000000L;
  }
  pthread_cond_timedwait(c, m, &t);
}
static int64_t now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
#endif

// ---- 要求 ----

typedef struct {
  _Atomic int refs;   // 表と裏のスレッドがそれぞれ1つ持つ。0 になったら解放する
  _Atomic int cancel; // 中止を頼まれた
  um_lock lock;       // 下の欄を守る
  um_cond cond;       // 状態か本文が変わった
  int state, status;
  unsigned char *buf; // まだ取り出していない本文 buf[start .. start + len)
  size_t start, len, cap;
  char *headers;      // 最後の応答の頭（"名前: 値" を "\n" でつないだもの。状態の行は含まない）
  char *error;
  uint64_t seq, seen; // 変わった回数と、httpWait が最後に見た回数
  // 要求の中身（裏のスレッドだけが読む）。
  char *method, *url, *req_headers;
  unsigned char *body;
  size_t body_len;
  int timeout_ms;
#ifdef _WIN32
  HINTERNET hreq;     // 中止のときにほかのスレッドから閉じる（lock で守る）
#endif
} req;

static char *dup_n(const char *s, size_t n) {
  char *p = malloc(n + 1);
  if (!p) um_panic(5, 0);
  if (n) memcpy(p, s, n);
  p[n] = 0;
  return p;
}

static void req_unref(req *r) {
  if (atomic_fetch_sub(&r->refs, 1) != 1) return;
  free(r->buf);
  free(r->headers);
  free(r->error);
  free(r->method);
  free(r->url);
  free(r->req_headers);
  free(r->body);
  free(r);
}

// 状態を変える（lock を持って呼ぶ）。
static void bump(req *r) {
  r->seq++;
  cond_broadcast(&r->cond);
}

// 失敗にする（中止を頼まれていれば中止）。理由は printf の形。
static void finish_failed(req *r, const char *fmt, ...) {
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  lock(&r->lock);
  if (atomic_load(&r->cancel)) {
    r->state = ST_CANCELLED;
    free(r->error);
    r->error = dup_n("the request was cancelled", strlen("the request was cancelled"));
  } else {
    r->state = ST_FAILED;
    free(r->error);
    r->error = dup_n(msg, strlen(msg));
  }
  bump(r);
  unlock(&r->lock);
}

static void finish_done(req *r) {
  lock(&r->lock);
  if (atomic_load(&r->cancel)) {
    r->state = ST_CANCELLED;
    free(r->error);
    r->error = dup_n("the request was cancelled", strlen("the request was cancelled"));
  } else {
    r->state = ST_DONE;
  }
  bump(r);
  unlock(&r->lock);
}

// 状態コードと頭が届いた。
static void set_response(req *r, int status, const char *headers, size_t hlen) {
  lock(&r->lock);
  r->status = status;
  free(r->headers);
  r->headers = dup_n(headers, hlen);
  if (r->state == ST_WAITING) r->state = ST_RECEIVING;
  bump(r);
  unlock(&r->lock);
}

// 本文を足す。
static void append_body(req *r, const void *data, size_t n) {
  if (n == 0) return;
  lock(&r->lock);
  if (r->start > 0 && r->start + r->len + n > r->cap) {
    memmove(r->buf, r->buf + r->start, r->len);
    r->start = 0;
  }
  if (r->len + n > r->cap) {
    size_t cap = r->cap ? r->cap : 4096;
    while (cap < r->len + n) cap *= 2;
    unsigned char *b = realloc(r->buf, cap);
    if (!b) um_panic(5, 0);
    r->buf = b;
    r->cap = cap;
  }
  memcpy(r->buf + r->start + r->len, data, n);
  r->len += n;
  if (r->state == ST_WAITING) r->state = ST_RECEIVING;
  bump(r);
  unlock(&r->lock);
}

// ---- 表 ----

static req *slots[MAX_SLOTS];
static uint32_t gens[MAX_SLOTS];
static um_lock table_lock = UM_LOCK_INIT;

static int64_t table_add(req *r) {
  lock(&table_lock);
  for (int i = 0; i < MAX_SLOTS; i++) {
    if (!slots[i]) {
      slots[i] = r;
      int64_t h = ((int64_t)KIND_REQUEST << 56) | ((int64_t)(gens[i] & 0xffffffffu) << 24) | (int64_t)(i + 1);
      unlock(&table_lock);
      return h;
    }
  }
  unlock(&table_lock);
  return 0;
}

// 番号から場所を求める（table_lock を持って呼ぶ）。正しくなければ停止。
static int slot_of(int64_t h, int32_t line) {
  int64_t i = (h & 0xffffff) - 1;
  uint32_t gen = (uint32_t)((h >> 24) & 0xffffffff);
  if (h <= 0 || (h >> 56) != KIND_REQUEST || i < 0 || i >= MAX_SLOTS || !slots[i] || gens[i] != gen) {
    unlock(&table_lock);
    um_panic(UM_PANIC_HTTP_HANDLE, line);
  }
  return (int)i;
}

// 要求を借りる（返すときは req_unref）。
static req *acquire(int64_t h, int32_t line) {
  lock(&table_lock);
  req *r = slots[slot_of(h, line)];
  atomic_fetch_add(&r->refs, 1);
  unlock(&table_lock);
  return r;
}

// ---- UM の値を返す ----

static void give(const void *s, size_t n, unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (n == 0) return;
  unsigned char *p = um_alloc(n, 1);
  memcpy(p, s, n);
  *out = p;
  *len = n;
}

// ---- 要求の検査 ----

static int is_token_char(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c && strchr("!#$%&'*+-.^_`|~", c));
}

// 要求の頭（"名前: 値" を "\n" でつないだもの。行末の "\r" と空の行は無視）を確かめる。正しくなければ理由を返す。
static const char *check_headers(const char *h) {
  const char *p = h;
  while (*p) {
    const char *end = strchr(p, '\n');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n > 0 && p[n - 1] == '\r') n--;
    if (n > 0) {
      size_t k = 0;
      while (k < n && is_token_char((unsigned char)p[k])) k++;
      if (k == 0 || k >= n || p[k] != ':') return "each header must be one line of the form \"Name: value\"";
      for (size_t j = k + 1; j < n; j++) {
        unsigned char c = (unsigned char)p[j];
        if ((c < 0x20 && c != '\t') || c == 0x7f) return "a header value contains a control character";
      }
    }
    if (!end) break;
    p = end + 1;
  }
  return NULL;
}

static const char *check_request(req *r) {
  if (!r->method[0]) return "the method is empty";
  for (const char *p = r->method; *p; p++) {
    if (!is_token_char((unsigned char)*p)) return "the method contains a character that is not allowed";
  }
  const char *u = r->url;
  size_t skip;
  if (strncmp(u, "http://", 7) == 0 || strncmp(u, "HTTP://", 7) == 0) skip = 7;
  else if (strncmp(u, "https://", 8) == 0 || strncmp(u, "HTTPS://", 8) == 0) skip = 8;
  else return "only http:// and https:// URLs are supported";
  if (!u[skip] || u[skip] == '/') return "the URL has no host";
  for (const unsigned char *p = (const unsigned char *)u; *p; p++) {
    if (*p <= 0x20 || *p == 0x7f) return "the URL contains a space or a control character (percent-encode it)";
  }
  return check_headers(r->req_headers);
}

// ---- 裏のスレッド: Windows（WinHTTP） ----

#ifdef _WIN32

static WCHAR *wide(const char *s) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
  if (n <= 0) return NULL;
  WCHAR *w = malloc((size_t)n * sizeof(WCHAR));
  if (!w) um_panic(5, 0);
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n);
  return w;
}

static char *narrow(const WCHAR *w, size_t *len) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
  if (n <= 0) {
    *len = 0;
    return dup_n("", 0);
  }
  char *s = malloc((size_t)n);
  if (!s) um_panic(5, 0);
  WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL);
  *len = (size_t)n - 1;
  return s;
}

// WinHTTP の誤りの番号の説明（winhttp.dll の文の表から。なければ番号だけ）。
static void win_error(req *r, const char *what, DWORD code) {
  WCHAR wbuf[512];
  DWORD n = FormatMessageW(FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS, GetModuleHandleW(L"winhttp.dll"), code, 0, wbuf,
                           (DWORD)(sizeof wbuf / sizeof wbuf[0]), NULL);
  if (n == 0) n = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, wbuf,
                                 (DWORD)(sizeof wbuf / sizeof wbuf[0]), NULL);
  while (n > 0 && (wbuf[n - 1] == L'\r' || wbuf[n - 1] == L'\n' || wbuf[n - 1] == L' ' || wbuf[n - 1] == L'.')) wbuf[--n] = 0;
  if (n > 0) {
    size_t len;
    char *msg = narrow(wbuf, &len);
    finish_failed(r, "%s: %s (WinHTTP error %lu)", what, msg, (unsigned long)code);
    free(msg);
  } else {
    finish_failed(r, "%s: WinHTTP error %lu", what, (unsigned long)code);
  }
}

static DWORD WINAPI worker(void *arg) {
  req *r = arg;
  HINTERNET hs = NULL, hc = NULL, hr = NULL;
  WCHAR *wurl = NULL, *wmethod = NULL, *wpath = NULL, *whead = NULL, *raw = NULL;
  if (atomic_load(&r->cancel)) {
    finish_failed(r, "cancelled");
    goto out;
  }
  wurl = wide(r->url);
  wmethod = wide(r->method);
  if (!wurl || !wmethod) {
    finish_failed(r, "the URL is not valid UTF-8");
    goto out;
  }
  URL_COMPONENTS uc;
  memset(&uc, 0, sizeof uc);
  uc.dwStructSize = sizeof uc;
  uc.dwHostNameLength = (DWORD)-1;
  uc.dwUrlPathLength = (DWORD)-1;
  uc.dwExtraInfoLength = (DWORD)-1;
  uc.dwSchemeLength = (DWORD)-1;
  if (!WinHttpCrackUrl(wurl, 0, 0, &uc)) {
    win_error(r, "the URL could not be parsed", GetLastError());
    goto out;
  }
  WCHAR *host = malloc(((size_t)uc.dwHostNameLength + 1) * sizeof(WCHAR));
  if (!host) um_panic(5, 0);
  memcpy(host, uc.lpszHostName, uc.dwHostNameLength * sizeof(WCHAR));
  host[uc.dwHostNameLength] = 0;
  // 道と問い合わせ（# から後ろは送らない）。
  size_t extra = uc.dwExtraInfoLength;
  for (size_t i = 0; i < extra; i++) {
    if (uc.lpszExtraInfo[i] == L'#') {
      extra = i;
      break;
    }
  }
  size_t plen = uc.dwUrlPathLength + extra;
  wpath = malloc((plen + 2) * sizeof(WCHAR));
  if (!wpath) um_panic(5, 0);
  if (uc.dwUrlPathLength == 0) {
    wpath[0] = L'/';
    memcpy(wpath + 1, uc.lpszExtraInfo, extra * sizeof(WCHAR));
    wpath[1 + extra] = 0;
  } else {
    memcpy(wpath, uc.lpszUrlPath, uc.dwUrlPathLength * sizeof(WCHAR));
    memcpy(wpath + uc.dwUrlPathLength, uc.lpszExtraInfo, extra * sizeof(WCHAR));
    wpath[plen] = 0;
  }
  // 代理（プロキシ）は OS の設定に従う（Windows 8.1 より前は既定の設定）。
  hs = WinHttpOpen(L"UnifiedMozaic", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!hs) hs = WinHttpOpen(L"UnifiedMozaic", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!hs) {
    free(host);
    win_error(r, "WinHTTP could not start", GetLastError());
    goto out;
  }
  int t = r->timeout_ms;
  WinHttpSetTimeouts(hs, t, t, t, t);
  hc = WinHttpConnect(hs, host, uc.nPort, 0);
  free(host);
  if (!hc) {
    win_error(r, "could not connect", GetLastError());
    goto out;
  }
  hr = WinHttpOpenRequest(hc, wmethod, wpath, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                          uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0);
  if (!hr) {
    win_error(r, "could not open the request", GetLastError());
    goto out;
  }
  // 中止のとき、ほかのスレッドがこの handle を閉じて、待っている呼び出しを終わらせる。
  lock(&r->lock);
  r->hreq = hr;
  int cancelled = atomic_load(&r->cancel);
  unlock(&r->lock);
  if (cancelled) {
    finish_failed(r, "cancelled");
    goto out;
  }
  // 要求の頭: "\n" を "\r\n" にして渡す。
  if (r->req_headers[0]) {
    size_t n = strlen(r->req_headers);
    char *crlf = malloc(n * 2 + 1);
    if (!crlf) um_panic(5, 0);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
      if (r->req_headers[i] == '\r') continue;
      if (r->req_headers[i] == '\n') crlf[k++] = '\r';
      crlf[k++] = r->req_headers[i];
    }
    crlf[k] = 0;
    whead = wide(crlf);
    free(crlf);
    if (!whead) {
      finish_failed(r, "the headers are not valid UTF-8");
      goto out;
    }
    if (!WinHttpAddRequestHeaders(hr, whead, (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
      win_error(r, "the headers were not accepted", GetLastError());
      goto out;
    }
  }
  if (r->body_len > 0xffffffffu) {
    finish_failed(r, "the body is too large (4 GiB or more)");
    goto out;
  }
  if (!WinHttpSendRequest(hr, WINHTTP_NO_ADDITIONAL_HEADERS, 0, r->body_len ? r->body : WINHTTP_NO_REQUEST_DATA, (DWORD)r->body_len,
                          (DWORD)r->body_len, 0)) {
    win_error(r, "could not send the request", GetLastError());
    goto out;
  }
  if (!WinHttpReceiveResponse(hr, NULL)) {
    win_error(r, "no response", GetLastError());
    goto out;
  }
  DWORD status = 0, sz = sizeof status;
  WinHttpQueryHeaders(hr, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
                      WINHTTP_NO_HEADER_INDEX);
  // 頭: 状態の行を除き、"\r\n" を "\n" にする。
  DWORD rawlen = 0;
  WinHttpQueryHeaders(hr, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, WINHTTP_NO_OUTPUT_BUFFER, &rawlen,
                      WINHTTP_NO_HEADER_INDEX);
  char *heads = NULL;
  size_t hlen = 0;
  if (rawlen > 0) {
    raw = malloc(rawlen + sizeof(WCHAR));
    if (!raw) um_panic(5, 0);
    if (WinHttpQueryHeaders(hr, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw, &rawlen, WINHTTP_NO_HEADER_INDEX)) {
      raw[rawlen / sizeof(WCHAR)] = 0;
      size_t n8;
      char *s = narrow(raw, &n8);
      heads = malloc(n8 + 1);
      if (!heads) um_panic(5, 0);
      const char *p = strchr(s, '\n');
      p = p ? p + 1 : s + n8;
      for (; *p; p++) {
        if (*p != '\r') heads[hlen++] = *p;
      }
      while (hlen > 0 && heads[hlen - 1] == '\n') hlen--;
      free(s);
    }
  }
  set_response(r, (int)status, heads ? heads : "", hlen);
  free(heads);
  // 頭の Content-Length（あれば。途中で切れたことに気づくため）。
  uint64_t expect = UINT64_MAX, got = 0;
  {
    WCHAR cl[32];
    DWORD clsz = sizeof cl;
    if (WinHttpQueryHeaders(hr, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, cl, &clsz, WINHTTP_NO_HEADER_INDEX)) {
      expect = wcstoull(cl, NULL, 10);
    }
  }
  unsigned char chunk[65536];
  for (;;) {
    DWORD avail = 0;
    if (!WinHttpQueryDataAvailable(hr, &avail)) {
      win_error(r, "the connection failed while receiving", GetLastError());
      goto out;
    }
    if (avail == 0) break;
    while (avail > 0) {
      DWORD want = avail < sizeof chunk ? avail : (DWORD)sizeof chunk, n = 0;
      if (!WinHttpReadData(hr, chunk, want, &n)) {
        win_error(r, "the connection failed while receiving", GetLastError());
        goto out;
      }
      if (n == 0) break;
      append_body(r, chunk, n);
      got += n;
      avail -= n;
    }
  }
  if (expect != UINT64_MAX && got < expect && _wcsicmp(wmethod, L"HEAD") != 0 && status != 204 && status != 304) {
    finish_failed(r, "the connection closed before the whole body arrived (%llu of %llu bytes)", (unsigned long long)got,
                  (unsigned long long)expect);
    goto out;
  }
  finish_done(r);
out:
  lock(&r->lock);
  if (r->hreq) WinHttpCloseHandle(r->hreq);
  r->hreq = NULL;
  unlock(&r->lock);
  if (hc) WinHttpCloseHandle(hc);
  if (hs) WinHttpCloseHandle(hs);
  free(wurl);
  free(wmethod);
  free(wpath);
  free(whead);
  free(raw);
  req_unref(r);
  return 0;
}

// 中止: 待っている WinHTTP の呼び出しを、要求の handle を閉じて終わらせる。
static void abort_transfer(req *r) {
  lock(&r->lock);
  if (r->hreq) {
    WinHttpCloseHandle(r->hreq);
    r->hreq = NULL;
  }
  unlock(&r->lock);
}

static int start_thread(req *r) {
  HANDLE t = CreateThread(NULL, 1024 * 1024, worker, r, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
  if (!t) return 0;
  CloseHandle(t);
  return 1;
}

#else

// ---- 裏のスレッド: macOS と Linux（libcurl を実行時に読み込む） ----

typedef void CURL;
struct curl_slist;
typedef int CURLcode;
#define CURLOPT_WRITEDATA 10001
#define CURLOPT_URL 10002
#define CURLOPT_ERRORBUFFER 10010
#define CURLOPT_POSTFIELDS 10015
#define CURLOPT_HTTPHEADER 10023
#define CURLOPT_HEADERDATA 10029
#define CURLOPT_CUSTOMREQUEST 10036
#define CURLOPT_XFERINFODATA 10057
#define CURLOPT_USERAGENT 10018
#define CURLOPT_WRITEFUNCTION 20011
#define CURLOPT_HEADERFUNCTION 20079
#define CURLOPT_XFERINFOFUNCTION 20219
#define CURLOPT_POSTFIELDSIZE_LARGE 30120
#define CURLOPT_LOW_SPEED_LIMIT 19
#define CURLOPT_LOW_SPEED_TIME 20
#define CURLOPT_NOPROGRESS 43
#define CURLOPT_NOBODY 44
#define CURLOPT_FOLLOWLOCATION 52
#define CURLOPT_MAXREDIRS 68
#define CURLOPT_NOSIGNAL 99
#define CURLOPT_CONNECTTIMEOUT_MS 156
#define CURLOPT_PROTOCOLS 181
#define CURLOPT_REDIR_PROTOCOLS 182
#define CURLPROTO_HTTP 1L
#define CURLPROTO_HTTPS 2L
#define CURLINFO_RESPONSE_CODE 0x200002
#define CURL_GLOBAL_DEFAULT 3L
#define CURL_ERROR_SIZE 256
#define CURLE_WRITE_ERROR 23
#define CURLE_ABORTED_BY_CALLBACK 42

static struct {
  CURLcode (*global_init)(long);
  CURL *(*easy_init)(void);
  CURLcode (*easy_setopt)(CURL *, int, ...);
  CURLcode (*easy_perform)(CURL *);
  void (*easy_cleanup)(CURL *);
  CURLcode (*easy_getinfo)(CURL *, int, ...);
  const char *(*easy_strerror)(CURLcode);
  struct curl_slist *(*slist_append)(struct curl_slist *, const char *);
  void (*slist_free_all)(struct curl_slist *);
} curl;
static const char *curl_missing; // 読み込めなかった理由（読めたら NULL）
static pthread_once_t curl_once = PTHREAD_ONCE_INIT;

static void curl_load(void) {
#ifdef __APPLE__
  static const char *const names[] = {"/usr/lib/libcurl.4.dylib", "libcurl.4.dylib", "libcurl.dylib", NULL};
#else
  static const char *const names[] = {"libcurl.so.4", "libcurl-gnutls.so.4", "libcurl-nss.so.4", "libcurl.so", NULL};
#endif
  void *lib = NULL;
  for (int i = 0; names[i] && !lib; i++) lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
#ifdef __APPLE__
    curl_missing = "HTTP needs the system libcurl (/usr/lib/libcurl.4.dylib), but it could not be loaded";
#else
    curl_missing = "HTTP needs libcurl (libcurl.so.4), but it could not be loaded; install it with your package manager "
                   "(for example `sudo apt install libcurl4` or `sudo dnf install libcurl`)";
#endif
    return;
  }
#define LOAD(field, sym)                                         \
  do {                                                           \
    *(void **)&curl.field = dlsym(lib, sym);                     \
    if (!curl.field) {                                           \
      curl_missing = "the libcurl that was found is too old (" sym " is missing)"; \
      return;                                                    \
    }                                                            \
  } while (0)
  LOAD(global_init, "curl_global_init");
  LOAD(easy_init, "curl_easy_init");
  LOAD(easy_setopt, "curl_easy_setopt");
  LOAD(easy_perform, "curl_easy_perform");
  LOAD(easy_cleanup, "curl_easy_cleanup");
  LOAD(easy_getinfo, "curl_easy_getinfo");
  LOAD(easy_strerror, "curl_easy_strerror");
  LOAD(slist_append, "curl_slist_append");
  LOAD(slist_free_all, "curl_slist_free_all");
#undef LOAD
  if (curl.global_init(CURL_GLOBAL_DEFAULT) != 0) curl_missing = "libcurl could not be initialized";
}

// 受け取りの途中の頭（応答ごとに作り直す。リダイレクトの応答の頭は残さない）。
typedef struct {
  req *r;
  char *heads;
  size_t hlen, hcap;
  int code;          // 今の応答の状態コード
  int has_location;  // 今の応答に Location がある
} xfer;

static size_t on_header(char *data, size_t size, size_t nitems, void *arg) {
  xfer *x = arg;
  size_t n = size * nitems;
  if (atomic_load(&x->r->cancel)) return 0;
  size_t k = n;
  while (k > 0 && (data[k - 1] == '\n' || data[k - 1] == '\r')) k--;
  if (n >= 5 && memcmp(data, "HTTP/", 5) == 0) {
    // 新しい応答の始まり（状態の行）。
    x->hlen = 0;
    x->has_location = 0;
    const char *sp = memchr(data, ' ', k);
    x->code = sp ? atoi(sp + 1) : 0;
    return n;
  }
  if (k == 0) {
    // 頭の終わり。1xx と、たどるリダイレクトの頭はプログラムに見せない。
    int redirect = x->code >= 300 && x->code < 400 && x->has_location;
    if (x->code >= 200 && !redirect) set_response(x->r, x->code, x->heads ? x->heads : "", x->hlen);
    return n;
  }
  if (k >= 9 && strncasecmp(data, "location:", 9) == 0) x->has_location = 1;
  if (x->hlen + k + 1 > x->hcap) {
    size_t cap = x->hcap ? x->hcap * 2 : 1024;
    while (cap < x->hlen + k + 1) cap *= 2;
    char *h = realloc(x->heads, cap);
    if (!h) um_panic(5, 0);
    x->heads = h;
    x->hcap = cap;
  }
  if (x->hlen > 0) x->heads[x->hlen++] = '\n';
  memcpy(x->heads + x->hlen, data, k);
  x->hlen += k;
  return n;
}

static size_t on_body(char *data, size_t size, size_t nmemb, void *arg) {
  xfer *x = arg;
  if (atomic_load(&x->r->cancel)) return 0;
  append_body(x->r, data, size * nmemb);
  return size * nmemb;
}

static int on_progress(void *arg, int64_t dt, int64_t dn, int64_t ut, int64_t un) {
  (void)dt, (void)dn, (void)ut, (void)un;
  xfer *x = arg;
  return atomic_load(&x->r->cancel) ? 1 : 0;
}

static void *worker(void *arg) {
  req *r = arg;
  pthread_once(&curl_once, curl_load);
  if (curl_missing) {
    finish_failed(r, "%s", curl_missing);
    req_unref(r);
    return NULL;
  }
  CURL *c = curl.easy_init();
  if (!c) {
    finish_failed(r, "libcurl could not create a request");
    req_unref(r);
    return NULL;
  }
  xfer x = {r, NULL, 0, 0, 0, 0};
  char errbuf[CURL_ERROR_SIZE];
  errbuf[0] = 0;
  struct curl_slist *list = NULL;
  // 大きな本文で "Expect: 100-continue" を待たない。
  list = curl.slist_append(list, "Expect:");
  {
    char *h = r->req_headers;
    while (*h) {
      char *end = strchr(h, '\n');
      size_t n = end ? (size_t)(end - h) : strlen(h);
      size_t m = n;
      if (m > 0 && h[m - 1] == '\r') m--;
      if (m > 0) {
        char *line = dup_n(h, m);
        list = curl.slist_append(list, line);
        free(line);
      }
      if (!end) break;
      h = end + 1;
    }
  }
  long tsec = (r->timeout_ms + 999) / 1000;
  curl.easy_setopt(c, CURLOPT_URL, r->url);
  curl.easy_setopt(c, CURLOPT_ERRORBUFFER, errbuf);
  curl.easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
  curl.easy_setopt(c, CURLOPT_USERAGENT, "UnifiedMozaic");
  curl.easy_setopt(c, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
  // https から http へのリダイレクトはたどらない（WinHTTP の既定と同じ）。
  int secure = strncasecmp(r->url, "https:", 6) == 0;
  curl.easy_setopt(c, CURLOPT_REDIR_PROTOCOLS, secure ? CURLPROTO_HTTPS : CURLPROTO_HTTP | CURLPROTO_HTTPS);
  curl.easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl.easy_setopt(c, CURLOPT_MAXREDIRS, 10L);
  curl.easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, (long)r->timeout_ms);
  // 受信で、1 バイトも届かない時間が timeoutMs（秒に切り上げ）続いたら時間切れ。
  curl.easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L);
  curl.easy_setopt(c, CURLOPT_LOW_SPEED_TIME, tsec);
  curl.easy_setopt(c, CURLOPT_HTTPHEADER, list);
  curl.easy_setopt(c, CURLOPT_HEADERFUNCTION, on_header);
  curl.easy_setopt(c, CURLOPT_HEADERDATA, &x);
  curl.easy_setopt(c, CURLOPT_WRITEFUNCTION, on_body);
  curl.easy_setopt(c, CURLOPT_WRITEDATA, &x);
  curl.easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
  curl.easy_setopt(c, CURLOPT_XFERINFOFUNCTION, on_progress);
  curl.easy_setopt(c, CURLOPT_XFERINFODATA, &x);
  int head = strcmp(r->method, "HEAD") == 0;
  if (head) {
    curl.easy_setopt(c, CURLOPT_NOBODY, 1L);
  } else if (r->body_len > 0 || strcmp(r->method, "POST") == 0) {
    curl.easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, (int64_t)r->body_len);
    curl.easy_setopt(c, CURLOPT_POSTFIELDS, r->body_len ? (char *)r->body : "");
  }
  if (!head && strcmp(r->method, "GET") != 0 && strcmp(r->method, "POST") != 0) curl.easy_setopt(c, CURLOPT_CUSTOMREQUEST, r->method);
  CURLcode rc = curl.easy_perform(c);
  if (rc == 0) {
    long code = 0;
    curl.easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    lock(&r->lock);
    r->status = (int)code;
    if (!r->headers && x.heads) r->headers = dup_n(x.heads, x.hlen);
    unlock(&r->lock);
    finish_done(r);
  } else if (rc == CURLE_ABORTED_BY_CALLBACK || (rc == CURLE_WRITE_ERROR && atomic_load(&r->cancel))) {
    finish_failed(r, "cancelled");
  } else {
    finish_failed(r, "%s", errbuf[0] ? errbuf : curl.easy_strerror(rc));
  }
  curl.easy_cleanup(c);
  curl.slist_free_all(list);
  free(x.heads);
  req_unref(r);
  return NULL;
}

// libcurl は、呼び出し（on_body / on_progress）で中止に気づく（待ちの最中でもおよそ 1 秒ごとに呼ばれる）。
static void abort_transfer(req *r) { (void)r; }

static int start_thread(req *r) {
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setstacksize(&a, 1024 * 1024);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_t t;
  int rc = pthread_create(&t, &a, worker, r);
  pthread_attr_destroy(&a);
  return rc == 0;
}

#endif

// ---- 組み込み ----

// 要求を始めて番号を返す。始められなくても番号を返し、要求は失敗の状態になる（理由は httpError）。
int64_t um_http_start(const char *method, size_t mlen, const char *url, size_t ulen, const char *headers, size_t hlen,
                      const unsigned char *body, size_t blen, int32_t timeout_ms, int32_t line) {
  req *r = calloc(1, sizeof *r);
  if (!r) um_panic(5, 0);
  atomic_init(&r->refs, 1);
  atomic_init(&r->cancel, 0);
  lock_init(&r->lock);
  cond_init(&r->cond);
  r->method = dup_n(method, mlen);
  r->url = dup_n(url, ulen);
  r->req_headers = dup_n(headers, hlen);
  if (blen) {
    r->body = malloc(blen);
    if (!r->body) um_panic(5, 0);
    memcpy(r->body, body, blen);
  }
  r->body_len = blen;
  r->timeout_ms = timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS;
  int64_t h = table_add(r);
  if (!h) {
    req_unref(r);
    um_panic(5, line);  // 同時に 4096 を超える要求（閉じ忘れ）
  }
  const char *bad = NULL;
  if (memchr(method, 0, mlen) || memchr(url, 0, ulen) || memchr(headers, 0, hlen)) bad = "the method, URL or headers contain a zero byte";
  if (!bad) bad = check_request(r);
  if (bad) {
    finish_failed(r, "%s", bad);
    return h;
  }
  atomic_fetch_add(&r->refs, 1);
  if (!start_thread(r)) {
    atomic_fetch_sub(&r->refs, 1);
    finish_failed(r, "could not start a thread for the request");
  }
  return h;
}

int32_t um_http_state(int64_t h, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  int s = r->state;
  unlock(&r->lock);
  req_unref(r);
  return s;
}

int32_t um_http_status(int64_t h, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  int s = r->status;
  unlock(&r->lock);
  req_unref(r);
  return s;
}

// 届いて、まだ取り出していない本文をすべて取り出す。
void um_http_read(int64_t h, unsigned char **out, size_t *len, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  give(r->buf + r->start, r->len, out, len);
  r->start = 0;
  r->len = 0;
  r->seen = r->seq;
  unlock(&r->lock);
  req_unref(r);
}

// 次の1行（"\n" と、その前の "\r" を除く）を取り出す。1行そろっていなければ 0。
// 受け取りが終わっていれば、改行で終わらない最後の部分も1行として返す。
int32_t um_http_read_line(int64_t h, unsigned char **out, size_t *len, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  *out = NULL;
  *len = 0;
  int ok = 0;
  unsigned char *p = r->buf + r->start;
  unsigned char *nl = r->len ? memchr(p, '\n', r->len) : NULL;
  if (nl) {
    size_t n = (size_t)(nl - p);
    size_t m = n > 0 && p[n - 1] == '\r' ? n - 1 : n;
    give(p, m, out, len);
    r->start += n + 1;
    r->len -= n + 1;
    ok = 1;
  } else if (r->len > 0 && r->state >= ST_DONE) {
    give(p, r->len, out, len);
    r->start = 0;
    r->len = 0;
    ok = 1;
  } else {
    r->seen = r->seq;
  }
  if (r->len == 0) r->start = 0;
  unlock(&r->lock);
  req_unref(r);
  return ok;
}

void um_http_headers(int64_t h, unsigned char **out, size_t *len, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  give(r->headers, r->headers ? strlen(r->headers) : 0, out, len);
  unlock(&r->lock);
  req_unref(r);
}

void um_http_error(int64_t h, unsigned char **out, size_t *len, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  give(r->error, r->error ? strlen(r->error) : 0, out, len);
  unlock(&r->lock);
  req_unref(r);
}

// 前に httpWait か httpRead（か null を返した httpReadLine）を呼んでから、本文が届くか状態が変わるまで、
// timeoutMs ミリ秒まで待つ（0 は待たない、負は来るまで）。変わったら true。
int32_t um_http_wait(int64_t h, int32_t timeout_ms, int32_t line) {
  req *r = acquire(h, line);
  lock(&r->lock);
  int64_t deadline = timeout_ms >= 0 ? now_ms() + timeout_ms : -1;
  while (r->seq == r->seen) {
    if (r->state >= ST_DONE) break;
    int64_t left = -1;
    if (deadline >= 0) {
      left = deadline - now_ms();
      if (left <= 0) break;
    }
    cond_wait_ms(&r->cond, &r->lock, left);
  }
  int changed = r->seq != r->seen;
  r->seen = r->seq;
  unlock(&r->lock);
  req_unref(r);
  return changed;
}

// 中止する（番号はまだ使える。状態は少しあとで CANCELLED になる）。終わった要求には何もしない。
void um_http_cancel(int64_t h, int32_t line) {
  req *r = acquire(h, line);
  atomic_store(&r->cancel, 1);
  abort_transfer(r);
  req_unref(r);
}

// 番号を手放す（動いていれば中止する。待たない）。
void um_http_close(int64_t h, int32_t line) {
  lock(&table_lock);
  int i = slot_of(h, line);
  req *r = slots[i];
  slots[i] = NULL;
  gens[i]++;
  unlock(&table_lock);
  atomic_store(&r->cancel, 1);
  abort_transfer(r);
  req_unref(r);
}
