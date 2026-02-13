/*
 * static_ptrscan — scan specific memory ranges for pointers to a target range.
 * Usage: static_ptrscan <pid> <scan_start> <scan_end> <target_start> <target_end>
 * All addresses in hex.
 * Scans [scan_start, scan_end) for 8-byte values in [target_start, target_end).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define BUF_SIZE (1024 * 1024)
static char buf[BUF_SIZE];

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: static_ptrscan <pid> <scan_start> <scan_end> <target_start> <target_end>\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    unsigned long scan_start = strtoul(argv[2], NULL, 16);
    unsigned long scan_end = strtoul(argv[3], NULL, 16);
    unsigned long tgt_start = strtoul(argv[4], NULL, 16);
    unsigned long tgt_end = strtoul(argv[5], NULL, 16);

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open mem"); return 1; }

    long total = scan_end - scan_start;
    int found = 0;

    fprintf(stderr, "Scanning 0x%lx-0x%lx (%ld KB) for ptrs to 0x%lx-0x%lx\n",
            scan_start, scan_end, total/1024, tgt_start, tgt_end);

    long offset = 0;
    while (offset < total) {
        long to_read = total - offset;
        if (to_read > BUF_SIZE) to_read = BUF_SIZE;

        ssize_t nread = pread(fd, buf, to_read, scan_start + offset);
        if (nread < 8) break;

        for (long i = 0; i <= nread - 8; i += 8) {
            unsigned long val;
            memcpy(&val, buf + i, 8);
            if (val >= tgt_start && val < tgt_end) {
                unsigned long addr = scan_start + offset + i;
                unsigned long off_from_start = addr - scan_start;
                printf("0x%016lx (+0x%06lx) -> 0x%016lx (tgt+0x%lx)\n",
                       addr, off_from_start, val, val - tgt_start);
                found++;
            }
        }
        offset += nread;
    }

    close(fd);
    fprintf(stderr, "Found %d pointers\n", found);
    return 0;
}
