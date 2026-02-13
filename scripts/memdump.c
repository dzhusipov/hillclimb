/*
 * memdump — read float values at specific addresses.
 * Reads addresses from stdin (hex, one per line).
 * Outputs: hex_address float_value (ALL addresses, even unreadable → NaN)
 * Usage: memdump <pid>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: memdump <pid>\n");
        return 1;
    }
    int pid = atoi(argv[1]);
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    char line[128];
    int total = 0;
    float val;
    while (fgets(line, sizeof(line), stdin)) {
        unsigned long addr;
        if (sscanf(line, "0x%lx", &addr) != 1 &&
            sscanf(line, "%lx", &addr) != 1)
            continue;
        total++;
        if (pread(fd, &val, 4, addr) == 4) {
            printf("0x%016lx %.6f\n", addr, val);
        } else {
            printf("0x%016lx NaN\n", addr);
        }
    }
    close(fd);
    fprintf(stderr, "Dumped %d addresses\n", total);
    return 0;
}
