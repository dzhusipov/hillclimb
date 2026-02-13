/*
 * ptrscan — find pointers to a target address range.
 * Scans all readable memory for 8-byte values pointing into [target-range, target+range].
 * Groups results by region type (static lib vs heap).
 *
 * Usage: ptrscan <pid> <target_hex> [range] [--depth N]
 *   range: how far around target to search (default 0x1000)
 *   --depth: chain depth (default 1, max 3)
 *
 * Output: found_addr -> points_to [region_name]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (4 * 1024 * 1024)
#define MAX_RESULTS 50000
#define MAX_REGIONS 4096

struct region {
    unsigned long start, end;
    char perms[8];
    char name[256];
    int is_static;  /* 1 if belongs to a loaded library (not heap) */
};

struct result {
    unsigned long found_at;
    unsigned long points_to;
    int region_idx;
};

static struct region regions[MAX_REGIONS];
static int n_regions = 0;
static struct result results[MAX_RESULTS];
static int n_results = 0;
static char buf[BUF_SIZE];

int find_region(unsigned long addr) {
    for (int i = 0; i < n_regions; i++) {
        if (addr >= regions[i].start && addr < regions[i].end)
            return i;
    }
    return -1;
}

void load_maps(int pid) {
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen maps"); return; }

    char line[512];
    while (fgets(line, sizeof(line), f) && n_regions < MAX_REGIONS) {
        struct region *r = &regions[n_regions];
        char perms[8];
        unsigned long offset;
        int dev_major, dev_minor;
        unsigned long inode;
        char name[256] = "";

        int n = sscanf(line, "%lx-%lx %4s %lx %x:%x %lu %255[^\n]",
                        &r->start, &r->end, perms, &offset,
                        &dev_major, &dev_minor, &inode, name);
        if (n < 7) continue;

        strncpy(r->perms, perms, 7);
        /* Trim leading spaces from name */
        char *p = name;
        while (*p == ' ') p++;
        strncpy(r->name, p, 255);

        /* Static = belongs to a .so file with rw permissions */
        r->is_static = 0;
        if (strstr(r->name, ".so") && perms[1] == 'w')
            r->is_static = 1;
        /* Also mark [stack], dalvik as non-static */

        n_regions++;
    }
    fclose(f);
}

void scan_for_pointers(int fd, unsigned long target_lo, unsigned long target_hi) {
    for (int ri = 0; ri < n_regions; ri++) {
        struct region *r = &regions[ri];

        /* Only scan readable regions */
        if (r->perms[0] != 'r') continue;

        long size = r->end - r->start;
        if (size < 8 || size > 50000000) continue;

        /* Skip /dev/ */
        if (strstr(r->name, "/dev/")) continue;

        long offset = 0;
        while (offset < size && n_results < MAX_RESULTS) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;

            ssize_t nread = pread(fd, buf, to_read, r->start + offset);
            if (nread < 8) break;

            for (long i = 0; i <= nread - 8; i += 8) {
                unsigned long val;
                memcpy(&val, buf + i, 8);

                if (val >= target_lo && val <= target_hi) {
                    if (n_results < MAX_RESULTS) {
                        results[n_results].found_at = r->start + offset + i;
                        results[n_results].points_to = val;
                        results[n_results].region_idx = ri;
                        n_results++;
                    }
                }
            }
            offset += nread;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ptrscan <pid> <target_hex> [range_hex]\n");
        fprintf(stderr, "  Finds all pointers to [target-range, target+range]\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    unsigned long target = strtoul(argv[2], NULL, 16);
    unsigned long range = argc > 3 ? strtoul(argv[3], NULL, 16) : 0x1000;

    unsigned long target_lo = target > range ? target - range : 0;
    unsigned long target_hi = target + range;

    fprintf(stderr, "Scanning for pointers to [0x%lx — 0x%lx]\n", target_lo, target_hi);

    load_maps(pid);
    fprintf(stderr, "Loaded %d memory regions\n", n_regions);

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    scan_for_pointers(fd, target_lo, target_hi);
    close(fd);

    fprintf(stderr, "Found %d pointers\n", n_results);

    /* Print results grouped by static/heap */
    printf("=== STATIC (library .data/.bss) ===\n");
    for (int i = 0; i < n_results; i++) {
        int ri = results[i].region_idx;
        if (regions[ri].is_static) {
            unsigned long off_in_region = results[i].found_at - regions[ri].start;
            printf("0x%016lx -> 0x%016lx  [%s +0x%lx]\n",
                   results[i].found_at, results[i].points_to,
                   regions[ri].name, off_in_region);
        }
    }

    printf("\n=== HEAP / ANON ===\n");
    int heap_count = 0;
    for (int i = 0; i < n_results; i++) {
        int ri = results[i].region_idx;
        if (!regions[ri].is_static) {
            if (heap_count < 100) {  /* limit output */
                printf("0x%016lx -> 0x%016lx  [%s]\n",
                       results[i].found_at, results[i].points_to,
                       regions[ri].name);
            }
            heap_count++;
        }
    }
    if (heap_count > 100)
        printf("... and %d more heap pointers\n", heap_count - 100);

    return 0;
}
