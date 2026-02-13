/*
 * memrescan — re-read specific addresses and filter by new value.
 * Reads addresses from stdin (hex format, one per line).
 * Usage: memrescan <pid> <float_value> [tolerance]
 * Output: hex_address float_value (matches only)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: memrescan <pid> <value> [tolerance]\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    float target = atof(argv[2]);
    float tol = argc > 3 ? atof(argv[3]) : 1.0f;

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    char line[128];
    int total = 0, found = 0;
    float val;

    while (fgets(line, sizeof(line), stdin)) {
        unsigned long addr;
        if (sscanf(line, "0x%lx", &addr) != 1 &&
            sscanf(line, "%lx", &addr) != 1)
            continue;
        total++;

        if (pread(fd, &val, 4, addr) != 4) continue;
        if (val != val) continue;  /* skip NaN */

        if (fabsf(val - target) < tol) {
            printf("0x%016lx %.4f\n", addr, val);
            found++;
        }
    }

    close(fd);
    fprintf(stderr, "Rescan: %d/%d match target=%.2f tol=%.2f\n",
            found, total, target, tol);
    return 0;
}
