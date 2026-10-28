#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8GdnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR64C128,
    SimtRowViewSplit,
    CutlassSm70,
};

enum class Q8GdnInputConvScheduleId {
    DecodeFused,
    SplitKMmaFused,
    Materialized,
};

struct Q8GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8GdnInputPlan {
    Q8GdnInputScheduleId schedule;
};

struct Q8GdnInputConvPlan {
    Q8GdnInputConvScheduleId schedule;
};

const char* q8_gdn_input_schedule_name(Q8GdnInputScheduleId schedule) noexcept;
const char* q8_gdn_input_conv_schedule_name(Q8GdnInputConvScheduleId schedule) noexcept;
bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept;
Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem);
Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size);

// `workspace` may be null; it is required only by schedules that stage a
// dequantized weight (CutlassSm70). Capacity is reported by
// gdn_input_proj_workspace_capacity_bytes().
void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           WorkspaceArena* workspace, cudaStream_t stream);

// Transient bytes the Q8 GDN input projection needs across [min_cols, max_cols].
[[nodiscard]] std::size_t q8_gdn_input_workspace_bytes(std::int32_t min_cols,
                                                       std::int32_t max_cols);

} // namespace ninfer::ops::detail
