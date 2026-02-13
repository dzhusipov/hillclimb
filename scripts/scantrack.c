/*
 * scantrack — scan for float, wait, rescan same addresses, dump changed.
 * Usage: scantrack <pid> <target> <tolerance> <wait_seconds> <new_target> <new_tolerance>
 *
 * 1. Scan heap for float ≈ target ± tolerance
 * 2. Sleep wait_seconds (external gas tap happens)
 * 3. Re-read ALL found addresses
 * 4. Filter: value changed to ≈ new_target ± new_tolerance
 * 5. Dump 256 bytes around each changed address as hex+float
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (4 * 1024 * 1024)
#define MAX_MATCHES 100000

struct match {
    unsigned long addr;
    float val;
};

static struct match matches[MAX_MATCHES];
static int n_matches = 0;
static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: scantrack <pid> <target> <tol> <wait_sec> <new_target> <new_tol>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    float target = atof(argv[2]);
    float tol = atof(argv[3]);
    int wait_sec = atoi(argv[4]);
    float new_target = atof(argv[5]);
    float new_tol = atof(argv[6]);

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *maps = fopen(path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    /* Phase 1: Scan for target */
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

        /* heap-only filter */
        int dominated = 0;
        if (strstr(name, "scudo")) dominated = 1;
        if (strstr(name, "cocos2d")) dominated = 1;
        if (strstr(name, "fingersoft")) dominated = 1;
        if (strstr(name, "[heap]")) dominated = 1;
        if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0') dominated = 1;
        if (!dominated) continue;

        long offset = 0;
        while (offset < size && n_matches < MAX_MATCHES) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;
            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < 4) break;
            total_scanned += nread;

            for (long i = 0; i <= nread - 4; i += 4) {
                float val;
                memcpy(&val, buf + i, 4);
                if (val != val) continue;
                if (fabsf(val - target) < tol) {
                    matches[n_matches].addr = start + offset + i;
                    matches[n_matches].val = val;
                    n_matches++;
                    if (n_matches >= MAX_MATCHES) break;
                }
            }
            offset += nread;
        }
    }
    fclose(maps);

    fprintf(stderr, "Phase1: scanned %ld MB, found %d matches for %.1f±%.1f\n",
            total_scanned / (1024*1024), n_matches, target, tol);

    /* Phase 2: Wait for external action */
    fprintf(stderr, "Waiting %d seconds...\n", wait_sec);
    sleep(wait_sec);

    /* Phase 3: Re-read all found addresses, find changed ones */
    fprintf(stderr, "Phase3: re-reading %d addresses...\n", n_matches);

    struct match changed[1000];
    int n_changed = 0;

    for (int i = 0; i < n_matches && n_changed < 1000; i++) {
        float new_val;
        if (pread(fd, &new_val, 4, matches[i].addr) != 4) continue;
        if (new_val != new_val) continue;

        /* Value must have changed AND be near new_target */
        float delta = new_val - matches[i].val;
        if (fabsf(delta) < 0.1f) continue; /* didn't change */
        if (fabsf(new_val - new_target) < new_tol) {
            changed[n_changed].addr = matches[i].addr;
            changed[n_changed].val = new_val;
            n_changed++;
            fprintf(stderr, "  CHANGED: 0x%016lx  %.4f -> %.4f (delta=%.4f)\n",
                    matches[i].addr, matches[i].val, new_val, delta);
        }
    }

    fprintf(stderr, "Phase3: found %d changed addresses\n", n_changed);

    /* Phase 4: Dump 256 bytes around each changed address */
    for (int i = 0; i < n_changed; i++) {
        unsigned long base = changed[i].addr - 64;
        unsigned char dump[256];
        ssize_t nr = pread(fd, dump, 256, base);
        if (nr < 256) {
            fprintf(stderr, "Short read at 0x%lx\n", base);
            continue;
        }

        printf("\n=== 0x%016lx (value=%.4f) context [-64..+192] ===\n", changed[i].addr, changed[i].val);
        for (int off = 0; off < 256; off += 4) {
            float fval;
            memcpy(&fval, dump + off, 4);
            unsigned int ival;
            memcpy(&ival, dump + off, 4);
            char marker = ' ';
            if (base + off == changed[i].addr) marker = '*';
            printf("%c 0x%016lx  [%+4d]  0x%08x  %14.6f\n",
                   marker, base + off, off - 64, ival, fval);
        }
    }

    close(fd);
    return 0;
}
