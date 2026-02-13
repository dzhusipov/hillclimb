/*
 * memwatch — continuously read float values at given offsets from a base address.
 * Usage: memwatch <pid> <base_hex> <duration_sec> <interval_ms> <off1> [off2] ...
 *
 * Prints timestamp + float values at each offset, every interval_ms.
 * Use to verify that a memory address tracks car position/velocity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define MAX_OFFSETS 32

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: memwatch <pid> <base_hex> <duration_sec> <interval_ms> <off1> [off2...]\n");
        fprintf(stderr, "  Offsets are decimal, can be negative\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    unsigned long base = strtoul(argv[2], NULL, 16);
    int duration = atoi(argv[3]);
    int interval_ms = atoi(argv[4]);

    int offsets[MAX_OFFSETS];
    int n_offsets = 0;
    for (int i = 5; i < argc && n_offsets < MAX_OFFSETS; i++) {
        offsets[n_offsets++] = atoi(argv[i]);
    }

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    /* Print header */
    printf("time_ms");
    for (int i = 0; i < n_offsets; i++) {
        printf("\toff_%d", offsets[i]);
    }
    printf("\n");
    fflush(stdout);

    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    long elapsed_ms = 0;
    while (elapsed_ms < duration * 1000L) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed_ms = (now.tv_sec - t0.tv_sec) * 1000 + (now.tv_nsec - t0.tv_nsec) / 1000000;

        printf("%ld", elapsed_ms);
        for (int i = 0; i < n_offsets; i++) {
            float val;
            ssize_t nr = pread(fd, &val, 4, base + offsets[i]);
            if (nr == 4) {
                printf("\t%.4f", val);
            } else {
                printf("\tERR");
            }
        }
        printf("\n");
        fflush(stdout);

        usleep(interval_ms * 1000);
    }

    close(fd);
    return 0;
}
