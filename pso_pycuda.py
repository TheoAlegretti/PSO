import numpy as np
import time
import pycuda.autoinit
from config_pso import config_pso
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.driver import cudaMemcpy, cudaMemcpyHostToDevice


mod = SourceModule("""
    __global__ void update_particles(float *particles, float *velocities, float *best_positions, float *global_best_position, float w, float c1, float c2, int nb_part, int dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nb_part * dim) {
            int i = idx / dim;
            int j = idx % dim;
            float rp = (float)rand() / (float)RAND_MAX;
            float rg = (float)rand() / (float)RAND_MAX;
            velocities[idx] = w * velocities[idx] + c1 * rp * (best_positions[idx] - particles[idx]) + c2 * rg * (global_best_position[j] - particles[idx]);
            particles[idx] += velocities[idx];
        }
    }
""")

update_particles = mod.get_function("update_particles")


def pso_opti_gpu(fct, params):
    begin = time.time()

    nb_part = params["nb_part"]
    dim = params["Dim"]
    nb_iter = params["max_ite"]
    nb_block = 128
    nb_thread = 512

    particles = np.random.uniform(params["min_x"], params["max_x"], (nb_part, dim)).astype(np.float32)
    velocities = np.random.uniform(-params["vit"], params["vit"], (nb_part, dim)).astype(np.float32)
    best_positions = np.copy(particles)
    costs = np.array([fct(particles[i]) for i in range(nb_part)], dtype=np.float32)
    best_costs = np.copy(costs)
    global_best_index = np.argmax(costs)
    global_best_cost = costs[global_best_index]
    global_best_position = particles[global_best_index].astype(np.float32)

    particles_gpu = cuda.to_device(particles)
    velocities_gpu = cuda.to_device(velocities)
    best_positions_gpu = cuda.to_device(best_positions)
    global_best_position_gpu = cuda.to_device(global_best_position)

    for _ in range(nb_iter):
        update_particles(
            particles_gpu, velocities_gpu, best_positions_gpu, global_best_position_gpu,
            np.float32(params["w"]), np.float32(params["c1"]), np.float32(params["c2"]),
            np.int32(nb_part), np.int32(dim),
            block=(nb_block, 1, 1), grid=(nb_thread, 1, 1)
        )

        cuda.memcpy_dtoh(particles, particles_gpu)

        costs = np.array([fct(particles[i]) for i in range(nb_part)], dtype=np.float32)
        improved_mask = costs > best_costs
        best_positions[improved_mask] = particles[improved_mask]
        best_costs[improved_mask] = costs[improved_mask]

        global_best_index = np.argmax(costs)
        new_global_best_cost = costs[global_best_index]
        if new_global_best_cost > global_best_cost:
            global_best_cost = new_global_best_cost
            global_best_position = particles[global_best_index].astype(np.float32)
            cudaMemcpy(global_best_position_gpu, global_best_position, global_best_position.nbytes, cudaMemcpyHostToDevice)

    end = time.time()
    print("Global best cost: ", global_best_cost)
    print("Global best position: ", global_best_position)
    print("Execution time: ", end - begin)


if __name__ == "__main__":
    
    params = config_pso()

    pso_opti_gpu(params['function'], params)