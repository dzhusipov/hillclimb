/*
 * quickread — fast one-shot memory reads to bypass anti-cheat.
 * Opens /proc/PID/mem, reads N bytes, closes fd — in a loop.
 * Unlike memserver which keeps fd open (triggers crash in ~0.4s),
 * this mimics sigscan's safe pattern of open→read→close.
 *
 * Usage: quickread <pid> <addr_hex> <size> <interval_ms> [count]
 *   pid          — target process PID
 *   addr_hex     — hex address to read from (e.g. 0x7a12345678)
 *   size         — bytes to read (max 4096)
 *   interval_ms  — delay between reads in milliseconds
 *   count        — number of reads (0 = infinite, default=100)
 *
 * Output: one line per read with timestamp and float values
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

static void print_floats(const unsigned char *buf, int size, double elapsed) {
    printf("t=%.3f ", elapsed);
    int nfloats = size / 4;
    if (nfloats > 32) nfloats = 32;  /* limit output */
    for (int i = 0; i < nfloats; i++) {
        float f;
        memcpy(&f, buf + i * 4, 4);
        printf("%.4f ", f);
    }
    printf("\n");
    fflush(stdout);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: quickread <pid> <addr_hex> <size> <interval_ms> [count]\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    unsigned long addr = strtoul(argv[2], NULL, 16);
    int size = atoi(argv[3]);
    int interval_ms = atoi(argv[4]);
    int count = (argc > 5) ? atoi(argv[5]) : 100;

    if (size < 4 || size > 4096) {
        fprintf(stderr, "Error: size must be 4..4096\n");
        return 1;
    }

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);

    unsigned char buf[4096];
    double t0 = now_sec();
    int ok_count = 0, err_count = 0;

    struct timespec sleep_ts;
    sleep_ts.tv_sec = interval_ms / 1000;
    sleep_ts.tv_nsec = (interval_ms % 1000) * 1000000L;

    fprintf(stderr, "quickread: pid=%d addr=0x%lx size=%d interval=%dms count=%d\n",
            pid, addr, size, interval_ms, count);

    for (int i = 0; count == 0 || i < count; i++) {
        /* Open */
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "read %d: open failed: %s\n", i, strerror(errno));
            err_count++;
            if (err_count > 5) {
                fprintf(stderr, "Too many errors, stopping\n");
                break;
            }
            nanosleep(&sleep_ts, NULL);
            continue;
        }

        /* Read */
        ssize_t nr = pread(fd, buf, size, addr);

        /* Close immediately */
        close(fd);

        if (nr == size) {
            print_floats(buf, size, now_sec() - t0);
            ok_count++;
        } else {
            fprintf(stderr, "read %d: pread returned %zd (expected %d): %s\n",
                    i, nr, size, strerror(errno));
            err_count++;
            if (err_count > 5) {
                fprintf(stderr, "Too many errors, stopping\n");
                break;
            }
        }

        nanosleep(&sleep_ts, NULL);
    }

    fprintf(stderr, "Done: %d ok, %d errors, %.1f seconds\n",
            ok_count, err_count, now_sec() - t0);
    return 0;
}
