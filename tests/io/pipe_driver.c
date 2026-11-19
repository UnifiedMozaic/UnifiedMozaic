// tests/io/*.um を、標準入力をパイプにつないで動かす（node tests/run.ts が使う）。
//   pipe_driver <やり方> <プログラム>
//     nonblocking  読む側を「待たない」設定（O_NONBLOCK）にしたパイプへ、間を空けて2行書く。
//     partial      1行を2回に分けて書く（"sto"、1秒おいて "p\n"）。
// プログラムの標準出力はそのまま引き継ぐ。終了コードはプログラムのもの。
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void pause_ms(long ms) {
  struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
  while (nanosleep(&t, &t) != 0) {
  }
}

static void put(int fd, const char *s) {
  size_t n = strlen(s);
  while (n > 0) {
    ssize_t w = write(fd, s, n);
    if (w <= 0) return;
    s += w;
    n -= (size_t)w;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: pipe_driver <nonblocking|partial> <program>\n");
    return 2;
  }
  int nonblocking = strcmp(argv[1], "nonblocking") == 0;
  if (!nonblocking && strcmp(argv[1], "partial") != 0) return 2;
  int fds[2];
  if (pipe(fds) != 0) return 2;
  if (nonblocking && fcntl(fds[0], F_SETFL, fcntl(fds[0], F_GETFL) | O_NONBLOCK) != 0) return 2;
  pid_t pid = fork();
  if (pid < 0) return 2;
  if (pid == 0) {
    dup2(fds[0], 0);
    close(fds[0]);
    close(fds[1]);
    execl(argv[2], argv[2], (char *)NULL);
    _exit(127);
  }
  close(fds[0]);
  if (nonblocking) {
    pause_ms(300);
    put(fds[1], "first\n");
    pause_ms(300);
    put(fds[1], "second\n");
  } else {
    put(fds[1], "sto");
    pause_ms(1000);
    put(fds[1], "p\n");
  }
  close(fds[1]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 3;
}
