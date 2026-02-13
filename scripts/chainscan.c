/*
 * chainscan — find 2-level pointer chains: static → heap → target
 *
 * For each pointer in [scan_start, scan_end):
 *   if it points to readable memory (heap), read 4KB from there
 *   check if any 8-byte value in that 4KB points to [target_start, target_end)
 *
 * Usage: chainscan <pid> <scan_start> <scan_end> <target_start> <target_end>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define FOLLOW_SIZE 4096  /* bytes to read at each intermediate pointer */
#define MAX_CHAINS 10000

static char scan_buf[1024 * 1024];  /* for reading scan region */
static char follow_buf[FOLLOW_SIZE];

/* Check if address is in a reasonable range (not NULL, not kernel) */
int is_valid_ptr(unsigned long val) {
    return val > 0x10000 && val < 0x800000000000UL;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: chainscan <pid> <scan_start> <scan_end> <tgt_start> <tgt_end>\n");
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

    long scan_size = scan_end - scan_start;
    fprintf(stderr, "Scanning %ld KB static, target 0x%lx-0x%lx\n",
            scan_size / 1024, tgt_start, tgt_end);

    int chains_found = 0;
    int ptrs_checked = 0;
    long offset = 0;

    while (offset < scan_size) {
        long to_read = scan_size - offset;
        if (to_read > (long)sizeof(scan_buf)) to_read = sizeof(scan_buf);

        ssize_t nread = pread(fd, scan_buf, to_read, scan_start + offset);
        if (nread < 8) break;

        for (long i = 0; i <= nread - 8; i += 8) {
            unsigned long ptr1;
            memcpy(&ptr1, scan_buf + i, 8);

            if (!is_valid_ptr(ptr1)) continue;
            ptrs_checked++;

            /* Try to read 4KB from the intermediate address */
            ssize_t fread = pread(fd, follow_buf, FOLLOW_SIZE, ptr1);
            if (fread < 8) continue;

            /* Scan follow_buf for pointers to target */
            for (long j = 0; j <= fread - 8; j += 8) {
                unsigned long ptr2;
                memcpy(&ptr2, follow_buf + j, 8);

                if (ptr2 >= tgt_start && ptr2 < tgt_end) {
                    unsigned long static_off = (scan_start + offset + i) - scan_start;
                    printf("CHAIN: [base+0x%06lx] -> 0x%lx [+0x%lx] -> 0x%lx (tgt+0x%lx)\n",
                           static_off, ptr1, j, ptr2, ptr2 - tgt_start);
                    chains_found++;
                    if (chains_found >= MAX_CHAINS) goto done;
                }
            }
        }
        offset += nread;
    }

done:
    close(fd);
    fprintf(stderr, "Checked %d pointers, found %d chains\n", ptrs_checked, chains_found);
    return 0;
}
