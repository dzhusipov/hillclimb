/*
 * sigwatch — find car body by signature, then continuously monitor it.
 * Combines sigscan + memwatch in one tool (no gap between finding and reading).
 *
 * Usage: sigwatch <pid> <duration_sec> <interval_ms>
 *
 * Signature: 70.0, 35.5, 0.5, 0.5, 140.0, 71.0
 * Body X position is at sig_addr - 28.
 *
 * Reads:
 *   off  0: pos_x
 *   off -4: pos_y
 *   off 56: vel_x (two floats after signature end: sig+24+4+4 = +56 from body_x)
 *   off 60: vel_y
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define BUF_SIZE (4 * 1024 * 1024)

static const float SIG[] = { 70.0f, 35.5f, 0.5f, 0.5f, 140.0f, 71.0f };
#define SIG_LEN 6
#define SIG_BYTES (SIG_LEN * 4)

static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: sigwatch <pid> <duration_sec> <interval_ms>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int interval_ms = atoi(argv[3]);

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *maps = fopen(path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    unsigned char sig_bytes[SIG_BYTES];
    memcpy(sig_bytes, SIG, SIG_BYTES);

    /* Phase 1: Find signature */
    char line[512];
    unsigned long body_x_addr = 0;
    long total_scanned = 0;

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

        int ok = 0;
        if (strstr(name, "scudo")) ok = 1;
        if (strstr(name, "cocos2d")) ok = 1;
        if (strstr(name, "fingersoft")) ok = 1;
        if (strstr(name, "[heap]")) ok = 1;
        if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0') ok = 1;
        if (!ok) continue;

        long offset = 0;
        while (offset < size && !body_x_addr) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;
            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < SIG_BYTES + 28) break;
            total_scanned += nread;

            for (long i = 28; i <= nread - SIG_BYTES; i += 4) {
                if (memcmp(buf + i, sig_bytes, SIG_BYTES) == 0) {
                    body_x_addr = start + offset + i - 28;
                    break;
                }
            }
            offset += nread;
        }
        if (body_x_addr) break;
    }
    fclose(maps);

    if (!body_x_addr) {
        fprintf(stderr, "Signature not found after scanning %ld MB\n", total_scanned / (1024*1024));
        close(fd);
        return 1;
    }

    fprintf(stderr, "Found body at 0x%016lx (scanned %ld MB)\n", body_x_addr, total_scanned / (1024*1024));

    /* Phase 2: Dump initial struct (512 bytes from body_x - 64) */
    fprintf(stderr, "=== Initial struct dump ===\n");
    unsigned char dump[512];
    unsigned long dump_base = body_x_addr - 64;
    if (pread(fd, dump, 512, dump_base) == 512) {
        for (int off = 0; off < 512; off += 4) {
            float fval;
            unsigned int ival;
            memcpy(&fval, dump + off, 4);
            memcpy(&ival, dump + off, 4);
            char marker = ' ';
            if (dump_base + off == body_x_addr) marker = 'X';
            if (dump_base + off == body_x_addr - 4) marker = 'Y';
            fprintf(stderr, "%c [%+4d]  0x%08x  %14.6f\n",
                    marker, off - 64, ival, fval);
        }
    }

    /* Phase 3: Continuous monitoring */
    fprintf(stderr, "\n=== Monitoring for %d seconds ===\n", duration);
    printf("time_ms\tpos_x\tpos_y\tvel_x\tvel_y\n");
    fflush(stdout);

    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    long elapsed_ms = 0;
    while (elapsed_ms < duration * 1000L) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed_ms = (now.tv_sec - t0.tv_sec) * 1000 + (now.tv_nsec - t0.tv_nsec) / 1000000;

        float pos_x, pos_y, vel_x, vel_y;
        int ok = 1;
        if (pread(fd, &pos_x, 4, body_x_addr) != 4) ok = 0;
        if (pread(fd, &pos_y, 4, body_x_addr - 4) != 4) ok = 0;
        if (pread(fd, &vel_x, 4, body_x_addr + 56) != 4) ok = 0;
        if (pread(fd, &vel_y, 4, body_x_addr + 60) != 4) ok = 0;

        if (ok) {
            printf("%ld\t%.4f\t%.4f\t%.4f\t%.4f\n", elapsed_ms, pos_x, pos_y, vel_x, vel_y);
        } else {
            printf("%ld\tERR\tERR\tERR\tERR\n", elapsed_ms);
            break;
        }
        fflush(stdout);
        usleep(interval_ms * 1000);
    }

    close(fd);
    return 0;
}
