#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"

#include "ops/gdn_input_proj/q8/q8_gdn_input_cutlass_sm70.h"
#include "ops/linear/q8/q8_launch.h"

#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8GdnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// SplitKMmaDirect and MmaR64C128 both reach q8_small_t_mma_kernel / the
// rowsplit mma family, which trap below sm_80, and DecodeR8Direct's
// q8_k2048_decode_kernel has no token dimension at all -- it is strictly T=1.
// So unlike the other q8 sites in this port there is no already-general SIMT
// sibling to widen. SimtRowViewSplit builds one from pieces that are already
// validated: the [12288,2048] parent is exactly qkv rows [0,8192) followed by z
// rows [8192,12288), so two row views drive the general SIMT kernel straight
// into the two destinations, with no scratch and no new kernel arithmetic.
// Measured at 41% of a3b prefill on its own, so wide T goes to tensor cores; the
// SIMT row-view split stays for the narrow range where the fixed dequant cost of
// the whole [12288,2048] parent is not yet amortized. The seam is a placeholder
// until swept -- see docs/v100.md.
constexpr std::int32_t kCutlassFirstCols = 32;

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, kCutlassFirstCols - 1, Q8GdnInputScheduleId::SimtRowViewSplit},
    {kCutlassFirstCols, kAnyCols, Q8GdnInputScheduleId::CutlassSm70},
}};
#else
constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8GdnInputScheduleId::SplitKMmaDirect},
    {97, kAnyCols, Q8GdnInputScheduleId::MmaR64C128},
}};
#endif // NINFER_VOLTA_BUILD

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q8 GDN input routes must be exact and closed");

bool supported_shape(const Q8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

const char* q8_gdn_input_schedule_name(Q8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        return "gdn_input_proj.q8.decode.r8.direct.k2048.split2";
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        return "gdn_input_proj.q8.mma.splitk.direct.k2048";
    case Q8GdnInputScheduleId::MmaR64C128:
        return "gdn_input_proj.q8.mma.r64.c128.split2";
    case Q8GdnInputScheduleId::SimtRowViewSplit:
        return "gdn_input_proj.q8.simt.row_view.split2";
    case Q8GdnInputScheduleId::CutlassSm70:
        return "gdn_input_proj.q8.cutlass.sm70.split2";
    }
    return "gdn_input_proj.q8.unknown";
}

const char* q8_gdn_input_conv_schedule_name(Q8GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputConvScheduleId::DecodeFused:
        return "gdn_input_proj_conv.q8.decode.fused";
    case Q8GdnInputConvScheduleId::SplitKMmaFused:
        return "gdn_input_proj_conv.q8.mma.splitk.fused";
    case Q8GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q8.materialized";
    }
    return "gdn_input_proj_conv_snapshot.q8.unknown";
}

bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem) {
    if (!q8_gdn_input_admits(problem)) {
        throw std::invalid_argument("Q8 GDN input: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 GDN input: admitted problem has no covering route");
}

Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size) {
    if (!q8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {Q8GdnInputConvScheduleId::DecodeFused}; }
#ifdef NINFER_VOLTA_BUILD
    // The fused small-T convolution epilogue is implemented on the Ampere+
    // q8_small_t_mma family.  Volta must materialize the projection through its
    // qualified SIMT/CUTLASS route before applying the convolution separately.
    return {Q8GdnInputConvScheduleId::Materialized};
#else
    if (problem.cols <= 16) { return {Q8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {Q8GdnInputConvScheduleId::Materialized};
#endif // NINFER_VOLTA_BUILD
}

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           WorkspaceArena* workspace, cudaStream_t stream) {
    const Q8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const Q8GdnInputPlan plan = q8_gdn_input_resolve_plan(problem);
    switch (plan.schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR64C128:
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SimtRowViewSplit:
        q8_gdn_input_simt_row_view_split_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::CutlassSm70:
        // The public convenience overload documents that it needs no transient
        // workspace, so it cannot stage a dequantized parent. Degrade to the
        // in-place route rather than break that contract. Every caller that
        // matters passes an arena -- the a3b leaf and the Op's own test both use
        // the policy-bearing overload -- so this is a fallback, not the norm.
        if (workspace == nullptr) {
            q8_gdn_input_simt_row_view_split_launch(x, weight, qkv, z, stream);
            return;
        }
        q8_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
        return;
    }
    throw std::logic_error("Q8 GDN input: unknown schedule");
}

#ifdef NINFER_VOLTA_BUILD
namespace {

// Row view over a Q8_G32_FP16 RowSplit parent, matching the construction the a3b
// binder already uses for DFlash K/V views: codes are padded_k bytes per row and
// scales are padded_k/group FP16 values per row, so a row offset is exact.
Weight q8_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > parent.n ||
        parent.qtype != QType::Q8_G32_FP16 || parent.layout != QuantLayout::RowSplit ||
        parent.group <= 0) {
        throw std::logic_error("Q8 GDN input: invalid row view");
    }
    const auto code_row  = static_cast<std::uint64_t>(parent.padded_shape[1]);
    const auto scale_row = static_cast<std::uint64_t>(parent.padded_shape[1] / parent.group) *
                           sizeof(std::uint16_t);

    Weight out = parent;
    out.qdata =
        static_cast<const std::byte*>(parent.qdata) + static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(parent.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

} // namespace

void q8_gdn_input_simt_row_view_split_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, cudaStream_t stream) {
    const Weight qkv_weight = q8_row_view(weight, 0, qkv.ne[0]);
    const Weight z_weight   = q8_row_view(weight, qkv.ne[0], z.ne[0]);
    launch_q8_simt_r8_c8(x, qkv_weight, qkv, stream);
    launch_q8_simt_r8_c8(x, z_weight, z, stream);
}
#endif // NINFER_VOLTA_BUILD

std::size_t q8_gdn_input_workspace_bytes(std::int32_t min_cols, std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q8 GDN input workspace: invalid column interval");
    }
    // Only CutlassSm70 stages anything, and its need grows monotonically with the
    // column count (the dequantized parent is fixed; the activation cast is not),
    // so the widest routed column count bounds the interval.
    std::size_t required = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.schedule != Q8GdnInputScheduleId::CutlassSm70) { continue; }
        const std::int32_t first = route.first > min_cols ? route.first : min_cols;
        const std::int32_t last  = route.last < max_cols ? route.last : max_cols;
        if (first > last) { continue; }
        const std::size_t bytes = q8_gdn_input_cutlass_workspace_bytes(last);
        required                = bytes > required ? bytes : required;
    }
    return required;
}

} // namespace ninfer::ops::detail
