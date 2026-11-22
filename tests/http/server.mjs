// tests/http/*.um の相手になる、手元だけの HTTP サーバー（127.0.0.1 の空いている番号で待つ）。
// 最初の行に "port <番号>" を書く。親（tests/run.ts）が標準入力を閉じたら終わる。
// 環境変数 UM_TEST_TLS_KEY / UM_TEST_TLS_CERT（PEM のファイル）があれば、同じ応答を返す HTTPS のサーバーも立て、
// 最初の行を "port <番号> tlsport <番号>" にする（自分で署名した証明書なので、検証が有効なら接続は失敗する）。
// 外へはつながない。応答はどれも、AI の API（SSE / 改行区切りの JSON）の届き方をまねた決まった中身。
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createServer as createTlsServer } from 'node:https';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const handler = async (req, res) => {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const body = Buffer.concat(chunks).toString('utf8');
  const url = new URL(req.url, 'http://localhost');
  switch (url.pathname) {
    case '/hello': {
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'X-Um-Test': 'yes' });
      res.end('hello\nこんにちは\n');
      return;
    }
    case '/echo': {
      // 方法・頭・本文をそのまま返す（要求が正しく届いたかを確かめる）。
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end(`${req.method} ${url.search}\nx-api-key=${req.headers['x-api-key'] ?? ''}\ncontent-type=${req.headers['content-type'] ?? ''}\n${body}`);
      return;
    }
    case '/sse': {
      // Claude の Messages API のストリーミング（SSE）の形。少しずつ、行の途中でも区切って送る。
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      const words = ['Hel', 'lo', ',', 'wor', 'ld'];
      res.write('event: message_start\ndata: {"type":"message_start"}\n\n');
      for (const w of words) {
        await sleep(40);
        const ev = `event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"${w}"}}\n\n`;
        const half = Math.floor(ev.length / 2);
        res.write(ev.slice(0, half));
        await sleep(10);
        res.write(ev.slice(half));
      }
      await sleep(40);
      res.end('event: message_stop\ndata: {"type":"message_stop"}\n\n');
      return;
    }
    case '/ndjson': {
      // Ollama の /api/chat の形（改行区切りの JSON）。行の途中で区切り、最後の行は改行で終わらない。
      res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
      const lines = ['{"message":{"content":"A"},"done":false}', '{"message":{"content":"B"},"done":false}', '{"done":true}'];
      const all = lines.join('\n');
      for (let i = 0; i < all.length; i += 7) {
        res.write(all.slice(i, i + 7));
        await sleep(5);
      }
      res.end();
      return;
    }
    case '/notfound': {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end('{"error":"not found"}');
      return;
    }
    case '/redirect': {
      res.writeHead(302, { Location: '/hello' });
      res.end();
      return;
    }
    case '/cut': {
      // Content-Length より短いところで接続を切る。
      res.writeHead(200, { 'Content-Type': 'text/plain', 'Content-Length': '1000' });
      res.write('partial');
      await sleep(50);
      res.socket.destroy();
      return;
    }
    case '/cut-chunked': {
      // チャンクの途中（終わりの印の前）で接続を切る。
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write('data: one\n\n');
      await sleep(50);
      res.socket.destroy();
      return;
    }
    case '/slow': {
      // 応答の頭を遅らせる（時間切れを確かめる）。
      await sleep(Number(url.searchParams.get('ms') ?? '3000'));
      if (!res.destroyed) {
        res.writeHead(200);
        res.end('late');
      }
      return;
    }
    case '/hang': {
      // 頭と少しの本文のあと、何も送らない（中止を確かめる）。
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write('data: first\n\n');
      req.socket.on('close', () => {});
      return;
    }
    default:
      res.writeHead(400);
      res.end('unknown path');
  }
};

const listen = (srv) => new Promise((resolve) => {
  srv.keepAliveTimeout = 1000;
  srv.listen(0, '127.0.0.1', () => resolve(srv.address().port));
});
let line = `port ${await listen(createServer(handler))}`;
if (process.env.UM_TEST_TLS_KEY && process.env.UM_TEST_TLS_CERT) {
  const tls = createTlsServer({ key: readFileSync(process.env.UM_TEST_TLS_KEY), cert: readFileSync(process.env.UM_TEST_TLS_CERT) }, handler);
  // 検証で断られた接続の誤りは、標準エラー出力に出さない。
  tls.on('tlsClientError', () => {});
  line += ` tlsport ${await listen(tls)}`;
}
process.stdout.write(line + '\n');
// 親が標準入力を閉じたら（親が終わったときも）終わる。
process.stdin.on('end', () => process.exit(0));
process.stdin.on('error', () => process.exit(0));
process.stdin.resume();
