/*
 * nodefinder — find car position via Cocos2d-x Node structural pattern.
 *
 * Strategy:
 *   1. Find the LARGEST scudo:primary rw region (typically ~22MB)
 *   2. Structural scan: scale [1,1,1] + rot sin²+cos²=1 + copy at +108
 *   3. Wait for car to move, delta filter → pick the live (moving) Node
 *   4. Stream pos_x/pos_y with stale detection + rescan
 *
 * Cocos2d-x Node layout (offsets from pos_x at +0):
 *   [-36,-32]: pos_Y (duplicate)
 *   [-20]: sin(rotation), [-16]: cos(rotation)
 *   [-12,-8,-4]: scale (1.0, 1.0, 1.0)
 *   [+0]: pos_X
 *   [+4]: vel?
 *   [+60]: cos(tilt), [+64]: sin(tilt)
 *   [+96]: -0.707, [+100]: 0.707
 *   [+108]: pos_X copy
 *
 * Protocol v2 (stdout, binary):
 *   Header: "OK2\n" + initial_x(f32) + initial_y(f32)
 *   Normal: 7 floats [pos_x, pos_y, sin_rot, cos_rot, vel_raw, cos_tilt, sin_tilt] = 28 bytes
 *   Switch: NaN + 6 zeros (28 bytes), then new_initial_x + new_initial_y + 5 zeros (28 bytes)
 *
 * Usage: nodefinder <pid> [interval_ms] [max_sec] [--wait=N]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/uio.h>

/* The largest scudo:primary region is ~22MB — safe for process_vm_readv */
#define MAX_REGION (30 * 1024 * 1024)
static char buf1[MAX_REGION];

#define STALE_MS 2000
#define MAX_RESCANS 50
#define SCAN_RETRY_MS 2000
#define MAX_SCAN_RETRIES 3

/* Cocos2d-x Node offsets from pos_x */
#define NODE_POSY_OFF   (-32)
#define NODE_SINROT_OFF (-20)
#define NODE_COSROT_OFF (-16)
#define NODE_VEL_OFF      4
#define NODE_COSTILT_OFF 60
#define NODE_SINTILT_OFF 64
#define NODE_COPY_OFF   108
#define NODE_COS45_OFF   96
#define NODE_SIN45_OFF  100

#define FRAME_FLOATS 7

static const float SCALE_ONE = 1.0f;
static const float COS45 = 0.70710678f;  /* cos(45°) = sin(45°) */

static int pid_g;

/* We scan ONE region: the largest scudo:primary */
static unsigned long region_start, region_end;
static long region_size;

/* Structural candidates (Nodes found by pattern) */
#define MAX_NODES 8192
static unsigned long node_addrs[MAX_NODES];
static float node_vals[MAX_NODES];
static int num_nodes;

static ssize_t pvm_read(int pid, void *buf, size_t len, unsigned long addr) {
    struct iovec local = { .iov_base = buf, .iov_len = len };
    struct iovec remote = { .iov_base = (void *)addr, .iov_len = len };
    return process_vm_readv(pid, &local, 1, &remote, 1, 0);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Find the LARGEST scudo:primary rw region */
static int find_region(int pid) {
    char maps_path[256];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
    FILE *maps = fopen(maps_path, "r");
    if (!maps) return -1;

    char line[512];
    region_start = 0;
    region_end = 0;
    region_size = 0;

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        if (perms[1] != 'w') continue;
        if (strstr(line, "reserve")) continue;
        if (!strstr(line, "scudo:primary")) continue;

        long size = end - start;
        if (size > region_size && size <= MAX_REGION) {
            region_start = start;
            region_end = end;
            region_size = size;
        }
    }
    fclose(maps);

    if (region_size > 0) {
        fprintf(stderr, "  largest region: 0x%lx-0x%lx (%.1f MB)\n",
                region_start, region_end, region_size / (1024.0 * 1024.0));
        return 0;
    }
    return -1;
}

/* Structural scan: find all Cocos2d-x Node patterns in the region.
 * Pattern: scale xyz [1.0,1.0,1.0] at [-12,-8,-4], rotation sin²+cos²=1,
 * copy at +108 ≈ pos_x */
static int scan_structural(void) {
    num_nodes = 0;
    unsigned char one_bytes[4];
    memcpy(one_bytes, &SCALE_ONE, 4);

    ssize_t n = pvm_read(pid_g, buf1, region_size, region_start);
    if (n < region_size) {
        fprintf(stderr, "  read error: %zd / %ld\n", n, region_size);
        return 0;
    }

    long lo = 20;  /* need [-20] for rotation check */
    long hi = region_size - 112;
    if (hi <= lo) return 0;

    for (long i = lo; i < hi; i += 4) {
        /* Quick check: scale_z at [i-4] == 1.0 */
        if (memcmp(buf1 + i - 4, one_bytes, 4) != 0) continue;
        /* scale_y at [i-8] == 1.0 */
        if (memcmp(buf1 + i - 8, one_bytes, 4) != 0) continue;
        /* scale_x at [i-12] == 1.0 */
        if (memcmp(buf1 + i - 12, one_bytes, 4) != 0) continue;

        /* Check rotation at [-20,-16]: sin²+cos² ≈ 1.0 */
        float sin_rot, cos_rot;
        memcpy(&sin_rot, buf1 + i - 20, 4);
        memcpy(&cos_rot, buf1 + i - 16, 4);
        if (isnan(sin_rot) || isnan(cos_rot)) continue;
        float rot_sum = sin_rot * sin_rot + cos_rot * cos_rot;
        if (fabsf(rot_sum - 1.0f) > 0.1f) continue;

        /* Read pos_x at [i] */
        float pos_x;
        memcpy(&pos_x, buf1 + i, 4);
        if (isnan(pos_x) || isinf(pos_x)) continue;

        /* Check pos_x copy at [i+108] */
        float copy;
        memcpy(&copy, buf1 + i + NODE_COPY_OFF, 4);
        if (fabsf(copy - pos_x) > 5.0f) continue;

        if (fabsf(pos_x) > 100000.0f) continue;

        /* Check ±0.707 markers at [+96,+100] — car body signature */
        float cos45, sin45;
        memcpy(&cos45, buf1 + i + NODE_COS45_OFF, 4);
        memcpy(&sin45, buf1 + i + NODE_SIN45_OFF, 4);
        if (fabsf(cos45 + COS45) > 0.01f) continue;  /* should be -0.707 */
        if (fabsf(sin45 - COS45) > 0.01f) continue;  /* should be +0.707 */

        /* Check pos_Y duplicate: [-36] ≈ [-32] */
        float py1, py2;
        memcpy(&py1, buf1 + i - 36, 4);
        memcpy(&py2, buf1 + i - 32, 4);
        if (isnan(py1) || isnan(py2)) continue;
        if (fabsf(py1 - py2) > 0.01f) continue;

        if (num_nodes < MAX_NODES) {
            node_addrs[num_nodes] = region_start + i;
            node_vals[num_nodes] = pos_x;
            num_nodes++;
        }
    }

    return num_nodes;
}

/* Find the LIVE node via consensus: majority of car-part Nodes share the same
 * pos_x (they all belong to the same car). Templates have different pos_x.
 * No movement required — works even when car is stationary. */
static unsigned long find_consensus_node(void) {
    if (num_nodes == 0) return 0;
    if (num_nodes == 1) return node_addrs[0];

    /* Re-read current pos_x for all nodes */
    float vals[MAX_NODES];
    for (int i = 0; i < num_nodes; i++) {
        pvm_read(pid_g, &vals[i], 4, node_addrs[i]);
    }

    /* Find the pos_x value with most neighbors within tolerance */
    float tolerance = 2.0f;
    int best_count = 0;
    int best_idx = 0;

    for (int i = 0; i < num_nodes; i++) {
        int count = 0;
        for (int j = 0; j < num_nodes; j++) {
            if (fabsf(vals[i] - vals[j]) < tolerance) count++;
        }
        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    fprintf(stderr, "  consensus: %d/%d nodes at pos_x≈%.1f\n",
            best_count, num_nodes, vals[best_idx]);

    /* Need at least 3 nodes to agree */
    if (best_count < 3) {
        fprintf(stderr, "  no consensus (best cluster = %d)\n", best_count);
        return 0;
    }

    return node_addrs[best_idx];
}

/* Dump context around pos_x for debugging */
static void dump_context(unsigned long addr) {
    unsigned char ctx[160];
    unsigned long start = addr - 40;
    if (pvm_read(pid_g, ctx, 160, start) != 160) return;

    fprintf(stderr, "Context around 0x%lx:\n", addr);
    for (int i = 0; i < 160; i += 4) {
        float f;
        memcpy(&f, ctx + i, 4);
        long off = (long)(i - 40);
        const char *label = "";
        if (off == -32) label = " ← pos_Y";
        if (off == -20) label = " ← sin(rot)";
        if (off == -16) label = " ← cos(rot)";
        if (off == -12) label = " ← scale_x";
        if (off ==  -8) label = " ← scale_y";
        if (off ==  -4) label = " ← scale_z";
        if (off ==   0) label = " ★ pos_X";
        if (off ==   4) label = " ← vel?";
        if (off == 108) label = " ← pos_X copy";
        fprintf(stderr, "  [%+4ld] %12.4f%s\n", off, f, label);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stdout, "ERR:usage: nodefinder <pid> [interval_ms] [max_sec] [--wait=N]\n");
        fflush(stdout);
        return 1;
    }

    pid_g = atoi(argv[1]);
    int interval_ms = (argc > 2) ? atoi(argv[2]) : 20;
    int max_sec = (argc > 3) ? atoi(argv[3]) : 120;
    int wait_sec = 2;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--wait=", 7) == 0)
            wait_sec = atoi(argv[i] + 7);
    }

    fprintf(stderr, "nodefinder: PID=%d interval=%dms max=%ds wait=%ds\n",
            pid_g, interval_ms, max_sec, wait_sec);

    if (find_region(pid_g) < 0) {
        fprintf(stdout, "ERR:no scudo:primary region found\n");
        fflush(stdout);
        return 1;
    }

    /* Find car Node: structural scan + delta filter */
    unsigned long best_addr = 0;

    for (int attempt = 0; attempt < MAX_SCAN_RETRIES; attempt++) {
        if (attempt > 0) {
            fprintf(stderr, "retry #%d (waiting %dms)\n", attempt, SCAN_RETRY_MS);
            usleep(SCAN_RETRY_MS * 1000);
            find_region(pid_g);
        }

        fprintf(stderr, "Structural scan (attempt %d)...\n", attempt + 1);
        int n = scan_structural();
        fprintf(stderr, "  %d structural matches\n", n);

        if (n > 0) {
            best_addr = find_consensus_node();
            if (best_addr) break;
        }
    }

    if (!best_addr) {
        fprintf(stdout, "ERR:node not found after %d retries\n", MAX_SCAN_RETRIES);
        fflush(stdout);
        return 1;
    }

    dump_context(best_addr);

    /* Read initial pos_x, pos_y */
    float initial_x, initial_y;
    if (pvm_read(pid_g, &initial_x, 4, best_addr) != 4 ||
        pvm_read(pid_g, &initial_y, 4, best_addr + NODE_POSY_OFF) != 4) {
        fprintf(stdout, "ERR:read initial\n");
        fflush(stdout);
        return 1;
    }

    fprintf(stderr, "Streaming: pos_x=%.2f pos_y=%.2f addr=0x%lx\n",
            initial_x, initial_y, best_addr);

    /* Output header (protocol v2) */
    fprintf(stdout, "OK2\n");
    fwrite(&initial_x, sizeof(float), 1, stdout);
    fwrite(&initial_y, sizeof(float), 1, stdout);
    fflush(stdout);

    /* Stream with stale detection */
    double t_start = now_ms();
    float last_x = initial_x;
    double last_change_ms = t_start;
    int rescan_count = 0;
    int consecutive_stale = 0;

    while (1) {
        float pos_x, pos_y;
        if (pvm_read(pid_g, &pos_x, 4, best_addr) != 4)
            break;
        if (pvm_read(pid_g, &pos_y, 4, best_addr + NODE_POSY_OFF) != 4)
            break;

        /* Read extended physics data */
        float sin_rot, cos_rot, vel_raw, cos_tilt, sin_tilt;
        pvm_read(pid_g, &sin_rot,  4, best_addr + NODE_SINROT_OFF);
        pvm_read(pid_g, &cos_rot,  4, best_addr + NODE_COSROT_OFF);
        pvm_read(pid_g, &vel_raw,  4, best_addr + NODE_VEL_OFF);
        pvm_read(pid_g, &cos_tilt, 4, best_addr + NODE_COSTILT_OFF);
        pvm_read(pid_g, &sin_tilt, 4, best_addr + NODE_SINTILT_OFF);

        double now = now_ms();

        if (fabsf(pos_x - last_x) > 0.05f) {
            last_x = pos_x;
            last_change_ms = now;
            consecutive_stale = 0;
        }

        /* Stale detection → rescan */
        if ((now - last_change_ms) > STALE_MS && rescan_count < MAX_RESCANS) {
            rescan_count++;
            fprintf(stderr, "RESCAN #%d (stale %.0fms, pos_x=%.1f)\n",
                    rescan_count, now - last_change_ms, pos_x);

            if (++consecutive_stale >= 5) {
                fprintf(stderr, "  5 consecutive stales, exiting\n");
                break;
            }

            find_region(pid_g);
            int n = scan_structural();
            fprintf(stderr, "  rescan: %d structural matches\n", n);

            if (n > 0) {
                /* Quick liveness: snapshot + 500ms + check */
                float *live_vals = (float *)malloc(num_nodes * sizeof(float));
                for (int i = 0; i < num_nodes; i++)
                    pvm_read(pid_g, &live_vals[i], 4, node_addrs[i]);
                usleep(500000);

                unsigned long new_addr = 0;
                float new_best_delta = 0;
                for (int i = 0; i < num_nodes; i++) {
                    float cur;
                    pvm_read(pid_g, &cur, 4, node_addrs[i]);
                    float d = cur - live_vals[i];
                    if (d > 0.1f && d > new_best_delta) {
                        new_addr = node_addrs[i];
                        new_best_delta = d;
                    }
                }
                free(live_vals);

                if (new_addr && new_addr != best_addr) {
                    float new_x, new_y;
                    if (pvm_read(pid_g, &new_x, 4, new_addr) == 4 &&
                        pvm_read(pid_g, &new_y, 4, new_addr + NODE_POSY_OFF) == 4) {
                        fprintf(stderr, "  switch: 0x%lx→0x%lx pos_x=%.1f→%.1f\n",
                                best_addr, new_addr, pos_x, new_x);
                        best_addr = new_addr;
                        last_x = new_x;
                        last_change_ms = now;
                        consecutive_stale = 0;

                        /* Switch marker: NaN + 6 zeros (28 bytes) */
                        float switch_frame[FRAME_FLOATS] = {0};
                        switch_frame[0] = 0.0f / 0.0f;  /* NaN */
                        fwrite(switch_frame, sizeof(float), FRAME_FLOATS, stdout);

                        /* New initial: new_x, new_y, 5 zeros (28 bytes) */
                        float init_frame[FRAME_FLOATS] = {0};
                        init_frame[0] = new_x;
                        init_frame[1] = new_y;
                        fwrite(init_frame, sizeof(float), FRAME_FLOATS, stdout);
                        fflush(stdout);

                        /* Re-read physics for new addr */
                        pvm_read(pid_g, &sin_rot,  4, best_addr + NODE_SINROT_OFF);
                        pvm_read(pid_g, &cos_rot,  4, best_addr + NODE_COSROT_OFF);
                        pvm_read(pid_g, &vel_raw,  4, best_addr + NODE_VEL_OFF);
                        pvm_read(pid_g, &cos_tilt, 4, best_addr + NODE_COSTILT_OFF);
                        pvm_read(pid_g, &sin_tilt, 4, best_addr + NODE_SINTILT_OFF);
                        pvm_read(pid_g, &pos_x, 4, best_addr);
                        pvm_read(pid_g, &pos_y, 4, best_addr + NODE_POSY_OFF);
                    }
                }
            }
        }

        float frame[FRAME_FLOATS] = {pos_x, pos_y, sin_rot, cos_rot, vel_raw, cos_tilt, sin_tilt};
        fwrite(frame, sizeof(float), FRAME_FLOATS, stdout);
        fflush(stdout);

        if ((now - t_start) > max_sec * 1000.0)
            break;

        usleep(interval_ms * 1000);
    }

    fprintf(stderr, "exit (%.1fs, %d rescans)\n",
            (now_ms() - t_start) / 1000.0, rescan_count);
    return 0;
}
