#!/bin/bash

# Fully automatic parallel TAMP testing
# No manual configuration needed - just run this script!

NUM_PROCESSES=${1:-4}  # Default 4 processes, can be overridden via argument

echo "=========================================="
echo "Parallel TAMP Testing (Fully Automatic)"
echo "=========================================="
echo "Number of processes: $NUM_PROCESSES"
echo "Lock timeout: 15 minutes"
echo "Logs: tamp_process_*.log"
echo ""

# Start processes
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    log_file="tamp_process_${i}.log"
    python omnigibson/examples/learning/tamp.py > "$log_file" 2>&1 &
    pid=$!
    echo "Started process $i (PID: $pid, log: $log_file)"
    sleep 2  # Stagger startup to reduce conflicts
done

echo ""
echo "All processes started!"
echo ""
echo "Monitor progress:"
echo "  - View logs: tail -f tamp_process_*.log"
echo "  - Count completed: ls -1 tamp_results/*.result 2>/dev/null | wc -l"
echo "  - Watch progress: watch -n 5 'ls -1 tamp_results/*.result 2>/dev/null | wc -l'"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
wait

echo ""
echo "=========================================="
echo "All processes completed!"
echo "=========================================="

# Merge results
if [ -d "tamp_results" ]; then
    result_count=$(ls -1 tamp_results/*.result 2>/dev/null | wc -l)
    echo "Total results: $result_count"
    
    # Create merged result file using Python
    python3 << 'EOF'
import os

RESULT_DIR = 'tamp_results'
TASK_PLANNING_MODE = 'exhaustive'  # or read from config

all_results = []
for filename in os.listdir(RESULT_DIR):
    if filename.endswith('.result'):
        filepath = os.path.join(RESULT_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    all_results.append(content)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

all_results.sort()

output_file = f'tamp_valid_results_{TASK_PLANNING_MODE}.txt'
with open(output_file, 'w') as f:
    for result in all_results:
        f.write(result + '\n')

print(f"Merged {len(all_results)} results to {output_file}")
EOF
fi

echo "Done!"
