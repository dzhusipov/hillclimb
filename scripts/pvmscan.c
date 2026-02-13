/*
 * pvmscan — signature scanner using process_vm_readv.
 * Does NOT open /proc/PID/mem at all.
 * Reads memory regions via process_vm_readv syscall,
 * which should be invisible to anti-cheat that monitors /proc/PID/mem.
 *
 * Usage: pvmscan <pid>
 * Output: body_x address(es) on stdout, details on stderr
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/uio.h>

/* The signature: 6 consecutive floats = 24 bytes */
static const float SIG[] = { 70.0f, 35.5f, 0.5f, 0.5f, 140.0f, 71.0f };
#define SIG_LEN 6
#define SIG_BYTES (SIG_LEN * 4)

#define BUF_SIZE (4 * 1024 * 1024)
static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: pvmscan <pid>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    char maps_path[256];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);

    FILE *maps = fopen(maps_path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    unsigned char sig_bytes[SIG_BYTES];
    memcpy(sig_bytes, SIG, SIG_BYTES);

    char line[512];
    long total_scanned = 0;
    unsigned long found_body_x = 0;
    int found_count = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        if (perms[1] != 'w') continue;
        long size = end - start;
        if (size < 4096 || size > 50000000) continue;

        char *name = strrchr(line, ' ');
        if (name) name++; else name = "";
        if (strstr(name, "/dev/") || strstr(name, "/system/")) continue;

        /* heap-only filter */
        int ok = 0;
        if (strstr(name, "scudo")) ok = 1;
        if (strstr(name, "cocos2d")) ok = 1;
        if (strstr(name, "fingersoft")) ok = 1;
        if (strstr(name, "[heap]")) ok = 1;
        if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0') ok = 1;
        if (!ok) continue;

        long offset = 0;
        while (offset < size) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;

            /* Use process_vm_readv instead of open/pread/close */
            struct iovec local = { .iov_base = buf, .iov_len = to_read };
            struct iovec remote = { .iov_base = (void *)(start + offset), .iov_len = to_read };
            ssize_t nread = process_vm_readv(pid, &local, 1, &remote, 1, 0);

            if (nread < SIG_BYTES + 28) {
                offset += to_read; /* skip this chunk */
                continue;
            }
            total_scanned += nread;

            for (long i = 0; i <= nread - SIG_BYTES; i += 4) {
                if (memcmp(buf + i, sig_bytes, SIG_BYTES) == 0) {
                    unsigned long sig_addr = start + offset + i;
                    unsigned long body_x_addr = sig_addr - 28;

                    float pos_x = 0, pos_y = 0, vel_x = 0, vel_y = 0;
                    if (i >= 32) {
                        memcpy(&pos_x, buf + i - 28, 4);
                        memcpy(&pos_y, buf + i - 32, 4);
                    }
                    if ((long)(i + SIG_BYTES + 32) <= nread) {
                        memcpy(&vel_x, buf + i + SIG_BYTES + 4, 4);
                        memcpy(&vel_y, buf + i + SIG_BYTES + 8, 4);
                    }

                    ++found_count;
                    fprintf(stderr, "FOUND #%d sig=0x%lx body_x=0x%lx pos=(%.2f,%.2f) vel=(%.3f,%.3f)\n",
                            found_count, sig_addr, body_x_addr, pos_x, pos_y, vel_x, vel_y);
                    printf("0x%lx\n", body_x_addr);
                    found_body_x = body_x_addr;
                }
            }
            offset += nread;
        }
    }

    fclose(maps);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "Scanned %ld MB in %.2fs, %d matches\n", total_scanned / (1024*1024), elapsed, found_count);

    return found_body_x ? 0 : 1;
}
