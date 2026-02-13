/*
 * pvmread — memory reader using process_vm_readv.
 * Does NOT open /proc/PID/mem at all.
 * Reads memory via process_vm_readv syscall — invisible to anti-cheat.
 *
 * Usage: pvmread <pid> <addr_hex> <size> <interval_ms> [count]
 * Output: one line per read with timestamp and float values
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/uio.h>

static void print_floats(const unsigned char *buf, int size, double elapsed) {
    printf("t=%.3f ", elapsed);
    int nfloats = size / 4;
    if (nfloats > 20) nfloats = 20;
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
        fprintf(stderr, "Usage: pvmread <pid> <addr_hex> <size> <interval_ms> [count]\n");
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

    unsigned char buf[4096];
    double t0 = now_sec();
    int ok_count = 0, err_count = 0;

    struct timespec sleep_ts;
    sleep_ts.tv_sec = interval_ms / 1000;
    sleep_ts.tv_nsec = (interval_ms % 1000) * 1000000L;

    fprintf(stderr, "pvmread: pid=%d addr=0x%lx size=%d interval=%dms count=%d\n",
            pid, addr, size, interval_ms, count);

    for (int i = 0; count == 0 || i < count; i++) {
        struct iovec local = { .iov_base = buf, .iov_len = size };
        struct iovec remote = { .iov_base = (void *)addr, .iov_len = size };
        ssize_t nr = process_vm_readv(pid, &local, 1, &remote, 1, 0);

        if (nr == size) {
            print_floats(buf, size, now_sec() - t0);
            ok_count++;
        } else {
            fprintf(stderr, "read %d: process_vm_readv returned %zd: %s\n",
                    i, nr, strerror(errno));
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
