/*
 * patternscan — find b2Body-like structures in memory.
 * Searches for b2Transform pattern: (pos_x, pos_y, rot_cos, rot_sin)
 * where sin²+cos² ≈ 1 and pos_x is near expected distance.
 *
 * Usage: patternscan <pid> <expected_x> [tolerance]
 *   expected_x: approximate X position of car (distance + starting offset)
 *   tolerance: how far pos_x can be from expected (default 30.0)
 *
 * Looks for pattern at every 4-byte aligned address:
 *   float pos_x ≈ expected_x ± tolerance
 *   float pos_y in [-100, 100]
 *   float rot_s in [-1.1, 1.1]  (sin)
 *   float rot_c in [-1.1, 1.1]  (cos)
 *   sin²+cos² in [0.9, 1.1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (4 * 1024 * 1024)
#define MAX_RESULTS 5000

struct result {
    unsigned long addr;
    float pos_x, pos_y, rot_s, rot_c;
};

static struct result results[MAX_RESULTS];
static int n_results = 0;
static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: patternscan <pid> <expected_x> [tolerance]\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    float expected_x = atof(argv[2]);
    float tol = argc > 3 ? atof(argv[3]) : 30.0f;

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *maps = fopen(path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    char line[512];
    long total_scanned = 0;

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3)
            continue;
        if (perms[1] != 'w') continue;

        long size = end - start;
        if (size < 4096 || size > 50000000) continue;

        char *name = strrchr(line, ' ');
        if (name) { name++; } else name = "";
        if (strstr(name, "/dev/") || strstr(name, "/system/")) continue;

        /* Only scan heap-like regions */
        if (!strstr(name, "scudo") && !strstr(name, "cocos2d") &&
            !strstr(name, "fingersoft") && !strstr(name, "[heap]"))
            continue;

        long offset = 0;
        while (offset < size && n_results < MAX_RESULTS) {
            long to_read = size - offset;
            if (to_read > BUF_SIZE) to_read = BUF_SIZE;

            ssize_t nread = pread(fd, buf, to_read, start + offset);
            if (nread < 16) break;
            total_scanned += nread;

            for (long i = 0; i <= nread - 16; i += 4) {
                float px, py, rs, rc;
                memcpy(&px, buf + i, 4);
                memcpy(&py, buf + i + 4, 4);
                memcpy(&rs, buf + i + 8, 4);
                memcpy(&rc, buf + i + 12, 4);

                /* Check NaN */
                if (px != px || py != py || rs != rs || rc != rc) continue;

                /* pos_x near expected */
                if (fabsf(px - expected_x) > tol) continue;

                /* pos_y reasonable */
                if (py < -100.0f || py > 100.0f) continue;

                /* rotation sin/cos in range */
                if (rs < -1.1f || rs > 1.1f) continue;
                if (rc < -1.1f || rc > 1.1f) continue;

                /* sin²+cos² ≈ 1 */
                float mag = rs * rs + rc * rc;
                if (mag < 0.9f || mag > 1.1f) continue;

                if (n_results < MAX_RESULTS) {
                    results[n_results].addr = start + offset + i;
                    results[n_results].pos_x = px;
                    results[n_results].pos_y = py;
                    results[n_results].rot_s = rs;
                    results[n_results].rot_c = rc;
                    n_results++;
                }
            }
            offset += nread;
        }
    }

    fclose(maps);
    close(fd);

    fprintf(stderr, "Scanned %ld MB, found %d b2Transform matches\n",
            total_scanned / (1024*1024), n_results);

    for (int i = 0; i < n_results; i++) {
        printf("0x%016lx  x=%8.3f y=%8.3f sin=%7.4f cos=%7.4f\n",
               results[i].addr, results[i].pos_x, results[i].pos_y,
               results[i].rot_s, results[i].rot_c);
    }

    return 0;
}
