/*
 * sigscan — find a specific float signature pattern in process memory.
 * Searches for a sequence of known float constants to locate the car body struct.
 *
 * Pattern: 70.0, 35.5, 0.5, 0.5, 140.0, 71.0 (24 bytes)
 * These are AABB/dimension constants unique to the car body.
 * The car's X position is at -28 bytes before the start of this pattern.
 *
 * Usage: sigscan <pid>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (4 * 1024 * 1024)
#define MAX_RESULTS 100

/* The signature: 6 consecutive floats */
static const float SIG[] = { 70.0f, 35.5f, 0.5f, 0.5f, 140.0f, 71.0f };
#define SIG_LEN 6
#define SIG_BYTES (SIG_LEN * 4)  /* 24 bytes */

static char buf[BUF_SIZE];

struct result {
    unsigned long sig_addr;  /* where signature starts */
    unsigned long pos_x_addr; /* sig_addr - 28 = position X */
    float pos_x, pos_y;
    float vel_x, vel_y;
};

static struct result results[MAX_RESULTS];
static int n_results = 0;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: sigscan <pid>\n");
        return 1;
    }

    int pid = atoi(argv[1]);

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *maps = fopen(path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    /* Build byte signature */
    unsigned char sig_bytes[SIG_BYTES];
    memcpy(sig_bytes, SIG, SIG_BYTES);

    char line[512];
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

        /* heap-only */
        int ok = 0;
        if (strstr(name, "scudo")) ok = 1;
        if (strstr(name, "cocos2d")) ok = 1;
        if (strstr(name, "fingersoft")) ok = 1;
        if (strstr(name, "[heap]")) ok = 1;
        if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0') ok = 1;
        if (!ok) continue;

        long offset = 0;
        while (offset < size && n_results < MAX_RESULTS) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;
            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < SIG_BYTES + 28) break;
            total_scanned += nread;

            /* Scan for signature (4-byte aligned) */
            for (long i = 0; i <= nread - SIG_BYTES; i += 4) {
                if (memcmp(buf + i, sig_bytes, SIG_BYTES) == 0) {
                    unsigned long sig_addr = start + offset + i;
                    unsigned long body_addr = sig_addr - 28;

                    /* Read surrounding data: pos_x at -28, pos_y at -32, vel_x at +28, vel_y at +32 */
                    float pos_x = 0, pos_y = 0, vel_x = 0, vel_y = 0;

                    /* Position X is at sig - 28 = i - 28 */
                    if (i >= 28) {
                        memcpy(&pos_x, buf + i - 28, 4);
                        memcpy(&pos_y, buf + i - 32, 4);
                    }
                    /* Velocity X is at sig + 28 (offset from sig start) */
                    if (i + SIG_BYTES + 8 <= nread) {
                        memcpy(&vel_x, buf + i + SIG_BYTES, 4);
                        memcpy(&vel_y, buf + i + SIG_BYTES + 4, 4);
                    }

                    if (n_results < MAX_RESULTS) {
                        results[n_results].sig_addr = sig_addr;
                        results[n_results].pos_x_addr = body_addr;
                        results[n_results].pos_x = pos_x;
                        results[n_results].pos_y = pos_y;
                        results[n_results].vel_x = vel_x;
                        results[n_results].vel_y = vel_y;
                        n_results++;
                    }
                }
            }
            offset += nread;
        }
    }

    fclose(maps);
    close(fd);

    fprintf(stderr, "Scanned %ld MB, found %d signature matches\n",
            total_scanned / (1024*1024), n_results);

    for (int i = 0; i < n_results; i++) {
        printf("sig=0x%016lx  body_x=0x%016lx  pos=(%.3f, %.3f)  vel=(%.4f, %.4f)\n",
               results[i].sig_addr, results[i].pos_x_addr,
               results[i].pos_x, results[i].pos_y,
               results[i].vel_x, results[i].vel_y);
    }

    /* If we found exactly 1 match, also dump full struct */
    if (n_results > 0) {
        fprintf(stderr, "\n=== Full dump around first match ===\n");
        unsigned long base = results[0].pos_x_addr - 64;
        unsigned char dump[512];
        if (pread(fd, dump, 512, base) == 512) {
            for (int off = 0; off < 512; off += 4) {
                float fval;
                memcpy(&fval, dump + off, 4);
                unsigned int ival;
                memcpy(&ival, dump + off, 4);
                char marker = ' ';
                if (base + off == results[0].pos_x_addr) marker = 'X';
                if (base + off == results[0].pos_x_addr - 4) marker = 'Y';
                printf("%c [%+4d]  0x%08x  %14.6f\n",
                       marker, off - 64, ival, fval);
            }
        }
    }

    return 0;
}
