#pragma once

#ifdef NINFER_VOLTA_BUILD

#include "artifact/materializer.h"
#include "core/device.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/weights.h"

#include <span>

namespace ninfer::models::qwen3_5::loading {

// Volta port: permute the device payload of every row-scaled FP8 projection parent and every
// NVFP4 dense MLP parent in place into the QPN fragment order read by the sm_70 decode kernels.
// The artifact bytes are unchanged; device allocations keep their size, and transient repack
// storage is released before this returns. Runs once, before any native Weight is prepared.
void prepack_volta_qpn(const ModelWeights& weights, std::span<const PendingWeight> pending,
                       artifact::MaterializedArtifact& backing, DeviceContext& device);

} // namespace ninfer::models::qwen3_5::loading

#endif // NINFER_VOLTA_BUILD
