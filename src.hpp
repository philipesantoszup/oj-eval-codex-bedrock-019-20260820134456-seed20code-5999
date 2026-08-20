#pragma once
#include "simulator.hpp"
#include <string>
#include <vector>
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix* Q = rater.GetNextQuery(); // Shape: (N, 512) where N = i+1
    size_t num_current_keys = i + 1; // Use first i+1 keys and values

    // Step 1: Move current keys and values to shared memory
    for (size_t k = 0; k < num_current_keys; ++k) {
      if (keys[k]->GetPosition() != kInSharedMemory) {
          gpu_sim.MoveMatrixToSharedMem(keys[k]);
      }
      if (values[k]->GetPosition() != kInSharedMemory) {
          gpu_sim.MoveMatrixToSharedMem(values[k]);
      }
    }
    // Step 2: Collect current keys into K (num_current_keys, 512)
    Matrix* K = matrix_memory_allocator.Allocate("K_current_" + std::to_string(i));
    if (num_current_keys == 1) {
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
    } else {
      Matrix* temp = matrix_memory_allocator.Allocate("K_temp_0_" + std::to_string(i));
      gpu_sim.Copy(keys[0], temp, kInSharedMemory);
      for (size_t j = 1; j < num_current_keys; ++j) {
        Matrix* new_temp = matrix_memory_allocator.Allocate("K_temp_" + std::to_string(j) + "_" + std::to_string(i));
        gpu_sim.Concat(temp, keys[j], new_temp, 0, kInSharedMemory);
        temp = new_temp;
      }
      gpu_sim.Copy(temp, K, kInSharedMemory);
    }
    // Step 3: Transpose K to get K^T (512, num_current_keys)
    gpu_sim.Transpose(K, kInSharedMemory);

    // Step 4: Move Q to shared memory
    if (Q->GetPosition() != kInSharedMemory) {
        gpu_sim.MoveMatrixToSharedMem(Q);
    }

    // Step 5: Compute Q @ K^T → matmul_qk (N, num_current_keys)
    Matrix* matmul_qk = matrix_memory_allocator.Allocate("matmul_qk_" + std::to_string(i));
    gpu_sim.MatMul(Q, K, matmul_qk);

    // Step 6: Compute softmax for each row of matmul_qk
    size_t N = Q->GetRowNum();
    std::vector<Matrix*> softmaxed_rows;

    for (size_t j = 0; j < N; ++j) {
      // Step 6a: Get row j of matmul_qk (1xnum_current_keys)
      Matrix* row_j = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.GetRow(matmul_qk, j, row_j, kInSharedMemory);

      // Step 6b: Compute exp(row_j) → exp_row (1xnum_current_keys)
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatExp(row_j, exp_row);

      // Step 6c: Sum exp_row → sum_row (1x1)
      Matrix* sum_row = matrix_memory_allocator.Allocate("sum_row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Sum(exp_row, sum_row);

      // Step 6d: Compute exp_row / sum_row → softmax_row (1xnum_current_keys)
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatDiv(exp_row, sum_row, softmax_row);

      softmaxed_rows.push_back(softmax_row);
    }

    // Step 6e: Concatenate all softmaxed rows into softmaxed (N, num_current_keys)
    Matrix* softmaxed = matrix_memory_allocator.Allocate("softmaxed_" + std::to_string(i));
    if (N == 1) {
      gpu_sim.Copy(softmaxed_rows[0], softmaxed, kInSharedMemory);
    } else {
      Matrix* temp = softmaxed_rows[0];
      for (size_t j = 1; j < N; ++j) {
        Matrix* new_temp = matrix_memory_allocator.Allocate("temp_concat_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(temp, softmaxed_rows[j], new_temp, 0, kInSharedMemory);
        temp = new_temp;
      }
      gpu_sim.Copy(temp, softmaxed, kInSharedMemory);
    }

    // Step 7: Collect current values into V (num_current_keys, 512)
    Matrix* V = matrix_memory_allocator.Allocate("V_current_" + std::to_string(i));
    if (num_current_keys == 1) {
      gpu_sim.Copy(values[0], V, kInSharedMemory);
    } else {
      Matrix* temp_v = matrix_memory_allocator.Allocate("V_temp_0_" + std::to_string(i));
      gpu_sim.Copy(values[0], temp_v, kInSharedMemory);
      for (size_t j = 1; j < num_current_keys; ++j) {
        Matrix* new_temp_v = matrix_memory_allocator.Allocate("V_temp_" + std::to_string(j) + "_" + std::to_string(i));
        gpu_sim.Concat(temp_v, values[j], new_temp_v, 0, kInSharedMemory);
        temp_v = new_temp_v;
      }
      gpu_sim.Copy(temp_v, V, kInSharedMemory);
    }

    // Step 8: Compute softmaxed @ V → answer (N, 512)
    Matrix* answer = matrix_memory_allocator.Allocate("answer_" + std::to_string(i));
    gpu_sim.MatMul(softmaxed, V, answer);

    // Step 9: Move answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Run all instructions for this iteration
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit answer
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
