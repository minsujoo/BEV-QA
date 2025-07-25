#!/bin/bash -l
#SBATCH --job-name=collect_carla_data          
#SBATCH --nodes=1                     
#SBATCH --ntasks=1         
#SBATCH --export=NONE            
#SBATCH --time=24:00:00    
#SBATCH --gres=gpu:4         
#SBATCH --output=carla_data_output.txt
#SBATCH --error=carla_data_error.txt


ALL_ROUTES=(
    "run_route_routes_town01_tiny.sh" 
    "run_route_routes_town02_tiny.sh" 
    "run_route_routes_town03_tiny.sh" 
    "run_route_routes_town04_tiny.sh"
    #"run_route_routes_town05_tiny.sh"
    #"run_route_routes_town06_tiny.sh"
    #"run_route_routes_town07_tiny.sh"
    #"run_route_routes_town10_tiny.sh"
    #"run_route_routes_town01_short.sh" 
    #"run_route_routes_town02_short.sh" 
    #"run_route_routes_town03_short.sh" 
    #"run_route_routes_town04_short.sh"
    #"run_route_routes_town05_short.sh"
    #"run_route_routes_town06_short.sh"
    #"run_route_routes_town07_short.sh"
    #"run_route_routes_town10_short.sh"
    #"run_route_routes_town01_long.sh" 
    #"run_route_routes_town02_long.sh" 
    #"run_route_routes_town03_long.sh" 
    #"run_route_routes_town04_long.sh"
    #"run_route_routes_town05_long.sh"
    #"run_route_routes_town06_long.sh"
    )

PORTS=(2000 2002 2004 2006)
CARLA_PIDS=()  # Array to store CARLA process IDs

conda activate bevdriver

# Function to kill CARLA servers
cleanup_carla() {
    echo "Cleaning up CARLA servers..."
    for pid in "${CARLA_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            echo "Killed CARLA process $pid"
        fi
    done
    
    killall -9 CarlaUE4-Linux-* 2>/dev/null
    killall -9 CarlaUE4.sh 2>/dev/null
    
    for port in "${PORTS[@]}"; do
        fuser -k ${port}/tcp 2>/dev/null
    done
}

trap cleanup_carla EXIT

start_carla_servers() {
    CARLA_PIDS=()  
    for gpu in {0..3}; do
        export CUDA_VISIBLE_DEVICES=$gpu
        carla/CarlaUE4.sh --world-port=${PORTS[$gpu]} -RenderOffScreen -opengl &
        CARLA_PIDS+=($!)  
        echo "Started CARLA server on GPU $gpu with PID ${CARLA_PIDS[-1]}"
    done
    # Wait for servers to initialize
    sleep 30
    echo "All CARLA servers started"
}

run_route() {
    local gpu=$1
    local route=$2
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Running $route on GPU $gpu with port ${PORTS[$gpu]}"
    bash "data_collection/batch_run/$route"
}

start_carla_servers

current_index=0
while [ $current_index -lt ${#ALL_ROUTES[@]} ]; do
    echo "Starting batch beginning at route $current_index"
    
    for gpu in {0..3}; do
        route_index=$((current_index + gpu))
        if [ $route_index -lt ${#ALL_ROUTES[@]} ]; then
            echo "Starting ${ALL_ROUTES[$route_index]} on GPU $gpu"
            ( 
            set -x
            bash data_collection/batch_run/"${ALL_ROUTES[$route_index]}" > logs/route_gpu${gpu}.log 2>&1 
            ) &

        fi
    done

    wait
    
    current_index=$((current_index + 4))
    echo "Batch complete. Moving to route $current_index"
done

echo "All routes completed"

conda deactivate