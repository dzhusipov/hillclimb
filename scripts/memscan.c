/*
 * memscan — fast float memory scanner for /proc/PID/mem
 * Usage: memscan <pid> <float_value> [tolerance] [--heap]
 * --heap: only scan scudo heap + cocos2d regions (5x faster)
 * Output: hex_address float_value (one per line)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (4 * 1024 * 1024)  /* 4 MB read buffer */
#define MAX_MATCHES 100000

struct match {
    unsigned long addr;
    float val;
};

static struct match matches[MAX_MATCHES];
static int n_matches = 0;
static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: memscan <pid> <value> [tolerance] [--heap]\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    float target = atof(argv[2]);
    float tol = argc > 3 && argv[3][0] != '-' ? atof(argv[3]) : 1.0f;

    int heap_only = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--heap") == 0) heap_only = 1;
    }

    /* Open maps */
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *maps = fopen(path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    /* Open mem */
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    char line[512];
    long total_scanned = 0;
    int regions = 0;

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3)
            continue;

        /* Only writable regions */
        if (perms[1] != 'w') continue;

        long size = end - start;
        /* Skip tiny and huge regions */
        if (size < 4096 || size > 50000000) continue;

        /* Region name filtering */
        char *name = strrchr(line, ' ');
        if (name) name++; else name = "";

        /* Always skip device and system mappings */
        if (strstr(name, "/dev/") || strstr(name, "/system/"))
            continue;

        if (heap_only) {
            /* Only scan: scudo heap, cocos2d lib, fingersoft lib, [heap] */
            int dominated = 0;
            if (strstr(name, "scudo")) dominated = 1;
            if (strstr(name, "cocos2d")) dominated = 1;
            if (strstr(name, "fingersoft")) dominated = 1;
            if (strstr(name, "[heap]")) dominated = 1;
            if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0')
                dominated = 1;  /* unnamed anonymous regions */
            if (!dominated) continue;
        }

        regions++;

        /* Read in chunks */
        long offset = 0;
        while (offset < size && n_matches < MAX_MATCHES) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;

            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < 4) break;

            total_scanned += nread;

            /* Scan for float matches */
            for (long i = 0; i <= nread - 4; i += 4) {
                float val;
                memcpy(&val, buf + i, 4);
                /* Skip NaN/Inf */
                if (val != val || val == 1.0f/0.0f || val == -1.0f/0.0f)
                    continue;
                if (fabsf(val - target) < tol) {
                    if (n_matches < MAX_MATCHES) {
                        matches[n_matches].addr = start + offset + i;
                        matches[n_matches].val = val;
                        n_matches++;
                    }
                }
            }
            offset += nread;
        }
    }

    fclose(maps);
    close(fd);

    fprintf(stderr, "Scanned %ld MB in %d regions, found %d matches\n",
            total_scanned / (1024*1024), regions, n_matches);

    for (int i = 0; i < n_matches; i++) {
        printf("0x%016lx %.4f\n", matches[i].addr, matches[i].val);
    }

    return 0;
}
