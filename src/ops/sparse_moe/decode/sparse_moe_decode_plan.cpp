#include "core/weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops::detail {

std::size_t sparse_moe_decode_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_decode_workspace(layout);
    return layout.peak_bytes(1);
}

SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up, QType routed_down) {
    const bool main_profile =
        routed_gate_up == QType::Q4_G64_FP16 &&
        (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16);
    const bool mtp_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    if (!main_profile && !mtp_profile) {
        throw std::invalid_argument("sparse_moe: unsupported routed codec profile");
    }

    SparseMoeDecodePlan plan;
    plan.workspace_bytes = sparse_moe_decode_workspace_bytes();
    return plan;
}

} // namespace ninfer::ops::detail
