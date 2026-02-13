/*
 * safememserver — anti-cheat safe memory reader server.
 * Same protocol as memserver, but opens/closes /proc/PID/mem
 * for EACH read request (never holds fd for more than ~1ms).
 *
 * Protocol:
 *   Request (stdin):  "R <addr_hex> <size>\n"  — read <size> bytes at <addr>
 *                     "Q\n"                     — quit
 *   Response (stdout): binary bytes (exactly <size> bytes), or "E\n" on error
 *
 * Usage: safememserver <pid>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: safememserver <pid>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);

    /* Verify we can access the process */
    int test_fd = open(path, O_RDONLY);
    if (test_fd < 0) {
        perror("open mem");
        return 1;
    }
    close(test_fd);

    fprintf(stderr, "safememserver: attached to PID %d (safe mode)\n", pid);

    /* Disable buffering on stdout for immediate responses */
    setbuf(stdout, NULL);

    char line[256];
    char buf[4096];

    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == 'Q' || line[0] == 'q') break;

        if (line[0] == 'R' || line[0] == 'r') {
            unsigned long addr;
            int size;
            if (sscanf(line + 1, "%lx %d", &addr, &size) != 2) {
                write(1, "E\n", 2);
                continue;
            }
            if (size < 0 || size > 4096) {
                write(1, "E\n", 2);
                continue;
            }

            /* Open fd for this read only */
            int fd = open(path, O_RDONLY);
            if (fd < 0) {
                write(1, "E\n", 2);
                continue;
            }

            ssize_t nr = pread(fd, buf, size, addr);

            /* Close immediately */
            close(fd);

            if (nr == size) {
                write(1, buf, size);
            } else {
                write(1, "E\n", 2);
            }
        }
    }

    return 0;
}
