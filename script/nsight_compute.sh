# install nvidia-modprobe
sudo apt install nvidia-modprobe
# reboot
# run sudo nvidia-modprobe -u -c=0 once per boot
sudo nvidia-modprobe -u -c=0

ncu ./main.out
: <<'OUTPUT_OF_NSIGHT_COMPUTE'
==PROF== Connected to process 11966 (/home/tz/Documents/programming/SuperCuda/main.out)
==PROF== Profiling "vector_sum3" - 0: 0%....50%....100% - 9 passes
time elapsed = 1.33607 seconds
sum of vector = 1e+08
==PROF== Disconnected from process 11966
[11966] main.out@127.0.0.1
  vector_sum3(const float *, unsigned long, double *) (390625, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz         13.79
    SM Frequency                    Ghz          2.41
    Elapsed Cycles                cycle    14,159,394
    Memory Throughput                 %         33.44
    DRAM Throughput                   %         17.71
    Duration                         ms          5.88
    L1/TEX Cache Throughput           %         33.45
    L2 Cache Throughput               %          6.99
    SM Active Cycles              cycle 14,143,131.61
    Compute (SM) Throughput           %         88.35
    ----------------------- ----------- -------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                390,625
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            2.05
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              36
    Stack Size                                                 1,024
    Threads                                   thread     100,000,000
    # TPCs                                                        18
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            1,808.45
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           21
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        95.02
    Achieved Active Warps Per SM           warp        45.61
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    14,364,016
    Total DRAM Elapsed Cycles        cycle   324,399,104
    Average L1 Active Cycles         cycle 14,143,131.61
    Total L1 Elapsed Cycles          cycle   509,329,116
    Average L2 Active Cycles         cycle 12,462,271.62
    Total L2 Elapsed Cycles          cycle   202,908,768
    Average SM Active Cycles         cycle 14,143,131.61
    Total SM Elapsed Cycles          cycle   509,329,116
    Average SMSP Active Cycles       cycle 14,143,029.11
    Total SMSP Elapsed Cycles        cycle 2,037,316,464
    -------------------------- ----------- -------------
OUTPUT_OF_NSIGHT_COMPUTE
