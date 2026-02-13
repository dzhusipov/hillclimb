/*
 * pvmtest — minimal process_vm_readv test.
 * Tests which operation triggers anti-cheat:
 *   mode 0: only read /proc/PID/maps (no process_vm_readv)
 *   mode 1: one process_vm_readv call (4 bytes from stack)
 *   mode 2: 10 process_vm_readv calls (4 bytes each, 100ms apart)
 *   mode 3: scan ~10MB with process_vm_readv (like mini pvmscan)
 *   mode 4: scan full heap (~600MB) with process_vm_readv
 *
 * Usage: pvmtest <pid> <mode>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/uio.h>
#include <unistd.h>

static void msleep(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: pvmtest <pid> <mode>\n");
        fprintf(stderr, "  mode 0: only read /proc/PID/maps\n");
        fprintf(stderr, "  mode 1: one process_vm_readv (4 bytes)\n");
        fprintf(stderr, "  mode 2: 10x process_vm_readv (4 bytes, 100ms apart)\n");
        fprintf(stderr, "  mode 3: scan ~10MB with process_vm_readv\n");
        fprintf(stderr, "  mode 4: scan full heap (~600MB) with process_vm_readv\n");
        return 1;
    }

    int pid = atoi(argv[1]);
    int mode = atoi(argv[2]);

    char maps_path[256];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);

    /* Always parse maps to get a valid address */
    FILE *maps = fopen(maps_path, "r");
    if (!maps) { perror("fopen maps"); return 1; }

    char line[512];
    unsigned long first_rw_start = 0;
    unsigned long first_rw_end = 0;

    /* Collect heap-like regions */
    typedef struct { unsigned long start, end; } Region;
    Region regions[2000];
    int nregions = 0;

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        if (perms[0] != 'r' || perms[1] != 'w') continue;
        long size = end - start;
        if (size < 4096 || size > 50000000) continue;

        char *name = strrchr(line, ' ');
        if (name) name++; else name = "";
        if (strstr(name, "/dev/") || strstr(name, "/system/")) continue;

        int ok = 0;
        if (strstr(name, "scudo")) ok = 1;
        if (strstr(name, "cocos2d")) ok = 1;
        if (strstr(name, "fingersoft")) ok = 1;
        if (strstr(name, "[heap]")) ok = 1;
        if (strstr(name, "[anon]") || name[0] == '\n' || name[0] == '\0') ok = 1;
        if (!ok) continue;

        if (first_rw_start == 0) {
            first_rw_start = start;
            first_rw_end = end;
        }
        if (nregions < 2000) {
            regions[nregions].start = start;
            regions[nregions].end = end;
            nregions++;
        }
    }
    fclose(maps);

    fprintf(stderr, "maps parsed: %d heap regions, first=0x%lx-0x%lx\n",
            nregions, first_rw_start, first_rw_end);

    if (mode == 0) {
        fprintf(stderr, "mode 0: maps read only, done\n");
        return 0;
    }

    if (first_rw_start == 0) {
        fprintf(stderr, "No rw region found\n");
        return 1;
    }

    char buf[4 * 1024 * 1024];

    if (mode == 1) {
        /* One single process_vm_readv call */
        struct iovec local = { .iov_base = buf, .iov_len = 4 };
        struct iovec remote = { .iov_base = (void *)first_rw_start, .iov_len = 4 };
        ssize_t nr = process_vm_readv(pid, &local, 1, &remote, 1, 0);
        fprintf(stderr, "mode 1: single read returned %zd\n", nr);
        return nr == 4 ? 0 : 1;
    }

    if (mode == 2) {
        /* 10 small reads, 100ms apart */
        for (int i = 0; i < 10; i++) {
            struct iovec local = { .iov_base = buf, .iov_len = 4 };
            struct iovec remote = { .iov_base = (void *)(first_rw_start + i * 4096), .iov_len = 4 };
            ssize_t nr = process_vm_readv(pid, &local, 1, &remote, 1, 0);
            fprintf(stderr, "mode 2: read %d returned %zd\n", i, nr);
            if (nr != 4) break;
            msleep(100);
        }
        fprintf(stderr, "mode 2: done\n");
        return 0;
    }

    if (mode == 3) {
        /* Scan ~10MB */
        long total = 0;
        for (int r = 0; r < nregions && total < 10 * 1024 * 1024; r++) {
            long size = regions[r].end - regions[r].start;
            long to_read = size;
            if (to_read > 4 * 1024 * 1024) to_read = 4 * 1024 * 1024;
            if (total + to_read > 10 * 1024 * 1024) to_read = 10 * 1024 * 1024 - total;

            struct iovec local = { .iov_base = buf, .iov_len = to_read };
            struct iovec remote = { .iov_base = (void *)regions[r].start, .iov_len = to_read };
            ssize_t nr = process_vm_readv(pid, &local, 1, &remote, 1, 0);
            if (nr > 0) total += nr;
            else break;
        }
        fprintf(stderr, "mode 3: scanned %ld KB\n", total / 1024);
        return 0;
    }

    if (mode == 4) {
        /* Full heap scan */
        long total = 0;
        for (int r = 0; r < nregions; r++) {
            long size = regions[r].end - regions[r].start;
            long offset = 0;
            while (offset < size) {
                long to_read = size - offset;
                if (to_read > 4 * 1024 * 1024) to_read = 4 * 1024 * 1024;
                struct iovec local = { .iov_base = buf, .iov_len = to_read };
                struct iovec remote = { .iov_base = (void *)(regions[r].start + offset), .iov_len = to_read };
                ssize_t nr = process_vm_readv(pid, &local, 1, &remote, 1, 0);
                if (nr > 0) { total += nr; offset += nr; }
                else { offset += to_read; }
            }
        }
        fprintf(stderr, "mode 4: scanned %ld MB\n", total / (1024*1024));
        return 0;
    }

    fprintf(stderr, "Unknown mode %d\n", mode);
    return 1;
}
