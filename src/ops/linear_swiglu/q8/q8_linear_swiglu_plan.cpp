#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8LinearSwiGluScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// The DFlash MLP is Q8-only and its tuned routes all use Ampere+ MMA past
// decode. The paired SIMT kernel streams gate/up rows together and tiles any T.
constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, Q8LinearSwiGluScheduleId::DecodePairR16},
    {2, 4, Q8LinearSwiGluScheduleId::SimtPairC4},
    {5, kAnyCols, Q8LinearSwiGluScheduleId::SimtPairC8},
}};
#else
constexpr std::array<RouteSpec, 18> kRoutes{{
    {1, 1, Q8LinearSwiGluScheduleId::DecodePairR16},
    {2, 48, Q8LinearSwiGluScheduleId::SplitKMmaExactT},
    {49, 64, Q8LinearSwiGluScheduleId::MmaR32C64},
    {65, 80, Q8LinearSwiGluScheduleId::MmaR32C80},
    {81, 96, Q8LinearSwiGluScheduleId::MmaR32C96},
    {97, 128, Q8LinearSwiGluScheduleId::MmaR64C64},
    {129, 192, Q8LinearSwiGluScheduleId::MmaR32C64},
    {193, 240, Q8LinearSwiGluScheduleId::MmaR128C80},
    {241, 255, Q8LinearSwiGluScheduleId::MmaR32C128},
    {256, 256, Q8LinearSwiGluScheduleId::MmaR64C128},
    {257, 264, Q8LinearSwiGluScheduleId::MmaR64C64},
    {265, 288, Q8LinearSwiGluScheduleId::MmaR64C96},
    {289, 320, Q8LinearSwiGluScheduleId::MmaR64C64},
    {321, 384, Q8LinearSwiGluScheduleId::MmaR64C128},
    {385, 448, Q8LinearSwiGluScheduleId::MmaR128C64},
    {449, 512, Q8LinearSwiGluScheduleId::MmaR64C128},
    {513, 560, Q8LinearSwiGluScheduleId::MmaR128C80},
    {561, kAnyCols, Q8LinearSwiGluScheduleId::MmaR64C128},
}};
#endif

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q8 LinearSwiGLU routes must be exact and closed");

constexpr bool is_dflash1_shape(const Q8LinearSwiGluProblem& p) noexcept {
    return p.gate_up_rows == 12288 && p.output_rows == 6144 && p.k == 2048 && p.padded_k == 2048;
}

constexpr bool is_dflash2_shape(const Q8LinearSwiGluProblem& p) noexcept {
    return p.gate_up_rows == 34816 && p.output_rows == 17408 && p.k == 5120 &&
           p.padded_k == 5120;
}

bool supported_shape(const Q8LinearSwiGluProblem& problem) noexcept {
    return is_dflash1_shape(problem) || is_dflash2_shape(problem);
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows,
                                       std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q8_linear_swiglu_schedule_name(Q8LinearSwiGluScheduleId schedule) noexcept {
    if (schedule == Q8LinearSwiGluScheduleId::Materialized) {
        return "linear_swiglu.q8.materialized";
    }
    switch (schedule) {
    case Q8LinearSwiGluScheduleId::DecodePairR16:
        return "linear_swiglu.q8.decode.pair.r16";
    case Q8LinearSwiGluScheduleId::SimtPairC4:
        return "linear_swiglu.q8.simt.pair.c4";
    case Q8LinearSwiGluScheduleId::SimtPairC8:
        return "linear_swiglu.q8.simt.pair.c8";
    case Q8LinearSwiGluScheduleId::VoltaQpnSplit:
        return "linear_swiglu.q8.sm70.qpn.split";
    case Q8LinearSwiGluScheduleId::SplitKMmaExactT:
        return "linear_swiglu.q8.splitk.mma.pair.exact_t";
    case Q8LinearSwiGluScheduleId::MmaR32C64:
        return "linear_swiglu.q8.mma.pair.r16.c64";
    case Q8LinearSwiGluScheduleId::MmaR32C80:
        return "linear_swiglu.q8.mma.pair.r16.c80";
    case Q8LinearSwiGluScheduleId::MmaR32C96:
        return "linear_swiglu.q8.mma.pair.r16.c96";
    case Q8LinearSwiGluScheduleId::MmaR32C128:
        return "linear_swiglu.q8.mma.pair.r16.c128";
    case Q8LinearSwiGluScheduleId::MmaR64C64:
        return "linear_swiglu.q8.mma.pair.r32.c64";
    case Q8LinearSwiGluScheduleId::MmaR64C96:
        return "linear_swiglu.q8.mma.pair.r32.c96";
    case Q8LinearSwiGluScheduleId::MmaR64C128:
        return "linear_swiglu.q8.mma.pair.r32.c128";
    case Q8LinearSwiGluScheduleId::MmaR128C64:
        return "linear_swiglu.q8.mma.pair.r64.c64";
    case Q8LinearSwiGluScheduleId::MmaR128C80:
        return "linear_swiglu.q8.mma.pair.r64.c80";
    }
    return "linear_swiglu.q8.unknown";
}

bool q8_linear_swiglu_schedule_uses_mma(Q8LinearSwiGluScheduleId schedule) noexcept {
    return schedule != Q8LinearSwiGluScheduleId::DecodePairR16 &&
           schedule != Q8LinearSwiGluScheduleId::SimtPairC4 &&
           schedule != Q8LinearSwiGluScheduleId::SimtPairC8 &&
           schedule != Q8LinearSwiGluScheduleId::Materialized;
}

std::size_t q8_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    const Q8LinearSwiGluProblem lo{gate_up_rows, output_rows, k, padded_k, min_tokens};
    const Q8LinearSwiGluProblem hi{gate_up_rows, output_rows, k, padded_k, max_tokens};
    (void)q8_linear_swiglu_resolve_plan(lo);
    (void)q8_linear_swiglu_resolve_plan(hi);
    if (!is_dflash2_shape(hi)) { return 0; }
#ifdef NINFER_VOLTA_BUILD
    if (min_tokens >= 5 && max_tokens <= 8) { return 256; }
#endif
    return materialized_workspace_bytes(gate_up_rows, max_tokens);
}

bool q8_linear_swiglu_admits(const Q8LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8LinearSwiGluPlan q8_linear_swiglu_resolve_plan(const Q8LinearSwiGluProblem& problem) {
    if (!q8_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "Q8 LinearSwiGLU: exact problem or column count is not admitted");
    }
    if (is_dflash2_shape(problem)) {
#ifdef NINFER_VOLTA_BUILD
        if (problem.cols >= 5 && problem.cols <= 8) {
            return {Q8LinearSwiGluScheduleId::VoltaQpnSplit};
        }
#endif
        return {Q8LinearSwiGluScheduleId::Materialized};
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 LinearSwiGLU: admitted problem has no route");
}

void q8_linear_swiglu_execute_plan(const Q8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearSwiGluPlan resolved = q8_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q8 LinearSwiGLU: plan does not match exact problem");
    }
    if (plan.schedule == Q8LinearSwiGluScheduleId::Materialized) {
        auto scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    if (plan.schedule == Q8LinearSwiGluScheduleId::VoltaQpnSplit) {
        auto scope = ws.scope();
        (void)ws.alloc_bytes(256);
        q8_linear_swiglu_volta_qpn_split_launch(x, w, out, stream);
        return;
    }
    switch (plan.schedule) {
    case Q8LinearSwiGluScheduleId::DecodePairR16:
        q8_linear_swiglu_decode_pair_r16_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::SimtPairC4:
        q8_linear_swiglu_simt_pair_c4_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::SimtPairC8:
        q8_linear_swiglu_simt_pair_c8_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::VoltaQpnSplit:
        break; // handled above
    case Q8LinearSwiGluScheduleId::SplitKMmaExactT:
        q8_linear_swiglu_splitk_exact_t_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR32C64:
        q8_linear_swiglu_mma_r32_c64_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR32C80:
        q8_linear_swiglu_mma_r32_c80_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR32C96:
        q8_linear_swiglu_mma_r32_c96_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR32C128:
        q8_linear_swiglu_mma_r32_c128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR64C64:
        q8_linear_swiglu_mma_r64_c64_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR64C96:
        q8_linear_swiglu_mma_r64_c96_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR64C128:
        q8_linear_swiglu_mma_r64_c128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR128C64:
        q8_linear_swiglu_mma_r128_c64_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::MmaR128C80:
        q8_linear_swiglu_mma_r128_c80_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::Materialized:
        break; // handled above
    }
    throw std::logic_error("Q8 LinearSwiGLU: unknown schedule");
}

void q8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_swiglu_execute_plan(q8_linear_swiglu_resolve_plan(problem), x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
