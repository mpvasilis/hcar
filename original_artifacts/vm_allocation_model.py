from cpmpy import Model, AllDifferent, intvar
import numpy as np
import json
from datetime import datetime

# --- Cloud Infrastructure Data ----------------------------------------------------
# Define PM types
PM_TYPES = [
    {"capacity_cpu": 32, "capacity_memory": 128, "capacity_disk": 2048, "capacity_gpu": 4, "power_efficiency": 85},
    {"capacity_cpu": 64, "capacity_memory": 256, "capacity_disk": 4096, "capacity_gpu": 8, "power_efficiency": 90},
    {"capacity_cpu": 48, "capacity_memory": 192, "capacity_disk": 3072, "capacity_gpu": 6, "power_efficiency": 88}
]

PM_DATA = {}
for i in range(1, 10): 
    pm_type = PM_TYPES[(i-1) % len(PM_TYPES)]  
    PM_DATA[f"PM_{i}"] = pm_type.copy()

VM_TYPES = [
    {"demand_cpu": 4, "demand_memory": 16, "demand_disk": 256, "demand_gpu": 0},  # No GPU
    {"demand_cpu": 8, "demand_memory": 32, "demand_disk": 512, "demand_gpu": 2},  # 2 GPUs
    {"demand_cpu": 16, "demand_memory": 64, "demand_disk": 1024, "demand_gpu": 4}  # 4 GPUs
]

VM_DATA = {}
for i in range(1, 20): 
    vm_type_idx = (i-1) % len(VM_TYPES)
    vm_data = VM_TYPES[vm_type_idx].copy()
    
    vm_data["priority"] = (i % 3) + 1  
    vm_data["availability_zone"] = "AZ1" if i % 2 == 1 else "AZ2"  
    
    VM_DATA[f"VM_{i}"] = vm_data

class VMAllocationModel:
    def __init__(self, time_period):
        self.time_period = time_period
        self.model = Model()
        self.pms = list(PM_DATA.keys())
        self.vms = list(VM_DATA.keys())
        self.setup_variables()
        self.add_constraints()
        self.solution = None
    
    def setup_variables(self):
        self.vm_assignments = {
            vm: intvar(0, len(self.pms), name=f"assign_{vm}") 
            for vm in self.vms
        }
        
        self.pm_active = {
            pm: intvar(0, 1, name=f"active_{pm}")
            for pm in self.pms
        }
        
        self.cpu_utilization = {
            pm: intvar(0, PM_DATA[pm]["capacity_cpu"], name=f"cpu_{pm}")
            for pm in self.pms
        }
        
        self.memory_utilization = {
            pm: intvar(0, PM_DATA[pm]["capacity_memory"], name=f"mem_{pm}")
            for pm in self.pms
        }
        
        self.disk_utilization = {
            pm: intvar(0, PM_DATA[pm]["capacity_disk"], name=f"disk_{pm}")
            for pm in self.pms
        }
    
    def add_constraints(self):
        print("Adding resource capacity constraints...")
        # 1. Resource Capacity Constraints
        for i, pm in enumerate(self.pms):
            # CPU Capacity
            cpu_sum = sum(VM_DATA[vm]["demand_cpu"] * (self.vm_assignments[vm] == i + 1) for vm in self.vms)
            self.model += [self.cpu_utilization[pm] == cpu_sum]
            self.model += [cpu_sum <= PM_DATA[pm]["capacity_cpu"]]
            
            # Memory Capacity
            mem_sum = sum(VM_DATA[vm]["demand_memory"] * (self.vm_assignments[vm] == i + 1) for vm in self.vms)
            self.model += [self.memory_utilization[pm] == mem_sum]
            self.model += [mem_sum <= PM_DATA[pm]["capacity_memory"]]
            
            # Disk Capacity
            disk_sum = sum(VM_DATA[vm]["demand_disk"] * (self.vm_assignments[vm] == i + 1) for vm in self.vms)
            self.model += [self.disk_utilization[pm] == disk_sum]
            self.model += [disk_sum <= PM_DATA[pm]["capacity_disk"]]
        
        print("Adding availability zone constraints...")
        # 2. Availability Zone Constraints - Only for high priority VMs
        for az in ["AZ1", "AZ2"]:
            high_priority_vms = [
                vm for vm in self.vms 
                if VM_DATA[vm]["availability_zone"] == az and VM_DATA[vm]["priority"] == 1
            ]
            if len(high_priority_vms) > 1:
                print(f"Adding AllDifferent constraint for {len(high_priority_vms)} high priority VMs in {az}")
                self.model += [AllDifferent([self.vm_assignments[vm] for vm in high_priority_vms])]
        
        print("Adding PM active status constraints...")
        # 3. PM Active Status Constraints
        for i, pm in enumerate(self.pms):
            vm_count = sum(self.vm_assignments[vm] == i + 1 for vm in self.vms)
            self.model += [self.pm_active[pm] == (vm_count > 0)]
        
        print("Adding VM assignment constraints...")
        # 4. Each VM must be assigned to exactly one PM
        for vm in self.vms:
            self.model += [self.vm_assignments[vm] >= 1]  # Must be assigned
            self.model += [self.vm_assignments[vm] <= len(self.pms)]  # Valid PM index
        
        print("All constraints added successfully.")
    
    def solve(self):
        if self.model.solve():
            self.create_solution_json()
            return True
        return False
    
    def create_solution_json(self):
        solution = {
            "time_period": self.time_period,
            "vm_assignments": {},
            "pm_utilization": {},
            "metadata": {
                "total_vms": len(self.vms),
                "total_pms": len(self.pms),
                "active_pms": sum(self.pm_active[pm].value() for pm in self.pms),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        # VM assignments
        for vm in self.vms:
            assigned_pm = self.vm_assignments[vm].value()
            if assigned_pm > 0:
                solution["vm_assignments"][vm] = {
                    "assigned_to": self.pms[assigned_pm - 1],
                    "resources": {
                        "cpu": VM_DATA[vm]["demand_cpu"],
                        "memory": VM_DATA[vm]["demand_memory"],
                        "disk": VM_DATA[vm]["demand_disk"]
                    },
                    "priority": VM_DATA[vm]["priority"],
                    "availability_zone": VM_DATA[vm]["availability_zone"]
                }
        
        # PM utilization
        for pm in self.pms:
            if self.pm_active[pm].value():
                solution["pm_utilization"][pm] = {
                    "active": True,
                    "utilization": {
                        "cpu": self.cpu_utilization[pm].value(),
                        "memory": self.memory_utilization[pm].value(),
                        "disk": self.disk_utilization[pm].value()
                    },
                    "capacity": {
                        "cpu": PM_DATA[pm]["capacity_cpu"],
                        "memory": PM_DATA[pm]["capacity_memory"],
                        "disk": PM_DATA[pm]["capacity_disk"]
                    },
                    "efficiency": PM_DATA[pm]["power_efficiency"]
                }
        
        self.solution = solution

def generate_multiple_solutions(num_solutions=5):
    """Generate multiple VM allocation solutions for different time periods."""
    all_solutions = []
    
    for time_period in range(num_solutions):
        print(f"\nGenerating solution for time period {time_period + 1}...")
        print("Creating model...")
        model = VMAllocationModel(time_period + 1)
        
        print("Attempting to solve model...")
        if model.solve():
            print("Solution found!")
            all_solutions.append(model.solution)
        else:
            print("Failed to find solution. Model is infeasible.")
            # Print resource requirements vs capacity
            total_cpu = sum(VM_DATA[vm]["demand_cpu"] for vm in VM_DATA)
            total_mem = sum(VM_DATA[vm]["demand_memory"] for vm in VM_DATA)
            total_disk = sum(VM_DATA[vm]["demand_disk"] for vm in VM_DATA)
            total_cpu_capacity = sum(PM_DATA[pm]["capacity_cpu"] for pm in PM_DATA)
            total_mem_capacity = sum(PM_DATA[pm]["capacity_memory"] for pm in PM_DATA)
            total_disk_capacity = sum(PM_DATA[pm]["capacity_disk"] for pm in PM_DATA)
            print(f"\nResource Requirements vs Capacity:")
            print(f"CPU: {total_cpu} / {total_cpu_capacity}")
            print(f"Memory: {total_mem} / {total_mem_capacity}")
            print(f"Disk: {total_disk} / {total_disk_capacity}")
    
    # Save solutions to JSON file
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_solutions": len(all_solutions),
            "total_vms": len(VM_DATA),
            "total_pms": len(PM_DATA)
        },
        "solutions": all_solutions
    }
    
    with open("vm_allocation_solutions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(all_solutions)} solutions saved to 'vm_allocation_solutions.json'")
    return all_solutions

if __name__ == "__main__":
    solutions = generate_multiple_solutions(5)
