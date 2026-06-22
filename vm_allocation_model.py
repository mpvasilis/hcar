"""PM/VM data for the ``vm_allocation`` benchmark (paper scale: 10 PMs, 40 VMs).

``main.py`` imports ``PM_DATA`` and ``VM_DATA`` from this module to build the
Cloud VM Allocation benchmark. The instance follows the paper's specification
(Appendix B): 40 VMs are assigned to 10 PMs subject to CPU/memory/disk capacity
(``Sum``), GPU capacity on the GPU-equipped PMs (``Count``), and high-availability
separation for priority-1 VMs per availability zone.

The original experiment-machine data file was not preserved in version control;
this module reconstructs an equivalent instance of the same problem. The
configuration below is verified satisfiable for both the binary
(``benchmarks/vm_allocation.py``) and global
(``benchmarks_global/vm_allocation.py``) encodings.

Each PM exposes ``capacity_cpu/memory/disk`` (plus ``capacity_gpu`` on the GPU
PMs); each VM exposes ``demand_cpu/memory/disk``, ``priority`` (1-3),
``availability_zone`` (``AZ1``/``AZ2``) and, for GPU VMs, ``demand_gpu``.

Note: the binary encoding co-locates VMs cautiously (every cross-AZ pair and
every pair of large-CPU/large-memory VMs must land on distinct PMs), so most VMs
are kept small and only a few are "large" per zone to keep the instance feasible
on 10 PMs.
"""

N_PMS = 10
N_VMS = 40
N_GPU_PMS = 6              # PM_1..PM_6 are GPU-equipped

N_LARGE_PER_AZ = 4        # "large" VMs per availability zone (must be spread out)
N_PRIO1_PER_AZ = 3        # priority-1 VMs per zone (drive the HA constraints)
N_PRIO3_PER_AZ = 2        # priority-3 VMs per zone

_GPU_DEMANDS = [1, 2, 4]  # three distinct GPU demands -> 6 GPU PMs x 3 = 18 Count


def _build_pm_data():
    pm_data = {}
    for i in range(1, N_PMS + 1):
        pm = {"capacity_cpu": 128, "capacity_memory": 512, "capacity_disk": 8192}
        if i <= N_GPU_PMS:
            pm["capacity_gpu"] = 16
        pm_data[f"PM_{i}"] = pm
    return pm_data


def _build_vm_data():
    vm_data = {}
    half = N_VMS // 2
    gpu_idx = 0
    for i in range(1, N_VMS + 1):
        az = "AZ1" if i <= half else "AZ2"
        pos = (i - 1) % half                       # index within the zone block
        if pos < N_LARGE_PER_AZ:                    # a few large VMs per zone
            vm = {"demand_cpu": 8, "demand_memory": 16, "demand_disk": 256, "priority": 2}
        else:                                       # the rest are small
            vm = {"demand_cpu": 2, "demand_memory": 4, "demand_disk": 64, "priority": 2}
            k = pos - N_LARGE_PER_AZ
            if k < N_PRIO1_PER_AZ:
                vm["priority"] = 1
            elif k < N_PRIO1_PER_AZ + N_PRIO3_PER_AZ:
                vm["priority"] = 3
        vm["availability_zone"] = az
        if pos % 5 == 0:                            # scatter GPU demands {1,2,4}
            vm["demand_gpu"] = _GPU_DEMANDS[gpu_idx % len(_GPU_DEMANDS)]
            gpu_idx += 1
        vm_data[f"VM_{i}"] = vm
    return vm_data


PM_DATA = _build_pm_data()
VM_DATA = _build_vm_data()
