/*
 * safescan — anti-cheat safe signature scanner.
 * Opens /proc/PID/mem briefly per region (open→pread→close),
 * never holds the fd for more than a few milliseconds.
 * This avoids the anti-cheat that detects persistent /proc/PID/mem fd.
 *
 * Usage: safescan <pid>
 * Output: body_x address on stdout (hex), details on stderr
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define BUF_SIZE (4 * 1024 * 1024)

/* The signature: 6 consecutive floats = 24 bytes */
static const float SIG[] = { 70.0f, 35.5f, 0.5f, 0.5f, 140.0f, 71.0f };
#define SIG_LEN 6
#define SIG_BYTES (SIG_LEN * 4)

static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: safescan <pid>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    char mem_path[256], maps_path[256];
    snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", pid);
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

        /* Open fd for THIS region only */
        int fd = open(mem_path, O_RDONLY);
        if (fd < 0) continue;

        long offset = 0;
        while (offset < size) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;
            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < SIG_BYTES + 28) break;
            total_scanned += nread;

            for (long i = 0; i <= nread - SIG_BYTES; i += 4) {
                if (memcmp(buf + i, sig_bytes, SIG_BYTES) == 0) {
                    unsigned long sig_addr = start + offset + i;
                    unsigned long body_x_addr = sig_addr - 28;

                    float pos_x = 0, pos_y = 0, vel_x = 0, vel_y = 0;
                    if (i >= 28) {
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
                    /* Print each match to stdout */
                    printf("0x%lx\n", body_x_addr);
                    found_body_x = body_x_addr;
                }
            }
            offset += nread;
        }

        /* Close fd immediately after each region */
        close(fd);
    }

    fclose(maps);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "Scanned %ld MB in %.2fs\n", total_scanned / (1024*1024), elapsed);

    fprintf(stderr, "Found %d matches\n", found_count);
    if (found_body_x) {
        return 0;
    }

    fprintf(stderr, "Signature not found\n");
    return 1;
}
