#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
// The Volta port restores the dedicated sm_70 target-verify kernels, so MTP keeps the full [1,7]
// window (bit-exact against --spec none through K=7; see docs/v100.md).
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
