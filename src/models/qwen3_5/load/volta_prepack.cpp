#include "models/qwen3_5/load/volta_prepack.h"

#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"

#include <map>
#include <set>
#include <variant>

namespace ninfer::models::qwen3_5::loading {
namespace {

// Complete-parent linear-family consumers whose sm_70 kernels read the QPN fragment order.
// Parents that also back a row view (Dense MTP Attention, draft context K/V) or a row gather
// (token embedding) are never selected, so those consumers keep the checkpoint layout.
struct Selection {
    std::set<std::size_t> fp8;
    std::set<std::size_t> nvfp4;

    void dense(const DenseWeights& w) {
        for (const WeightId id : {w.gate, w.up, w.down}) {
            fp8.insert(id.index);
            nvfp4.insert(id.index);
        }
    }

    void projection(WeightId id) { fp8.insert(id.index); }
};

Selection select(const ModelWeights& weights) {
    Selection out;
    for (const auto& layer : weights.text.layers) {
        if (const auto* attention = std::get_if<AttentionWeights>(&layer.mixer)) {
            for (const WeightId id : {attention->query, attention->key, attention->gate,
                                      attention->value, attention->output}) {
                out.projection(id);
            }
        } else {
            const auto& gdn = std::get<GdnWeights>(layer.mixer);
            for (const WeightId id : {gdn.query, gdn.key, gdn.value, gdn.z, gdn.output}) {
                out.projection(id);
            }
        }
        if (const auto* dense = std::get_if<DenseWeights>(&layer.ffn)) { out.dense(*dense); }
    }
    out.projection(weights.text.output_head);
    if (weights.mtp) {
        if (const auto* dense = std::get_if<DenseWeights>(&weights.mtp->layer.ffn)) {
            out.dense(*dense);
        }
    }
    if (weights.draft) {
        out.projection(weights.draft->feature_projection);
        for (const auto& layer : weights.draft->layers) {
            out.projection(layer.attention.output);
            for (const WeightId id : {layer.mlp.gate, layer.mlp.up, layer.mlp.down}) {
                out.projection(id);
            }
            for (const auto& conv : {layer.attention_conv, layer.mlp_conv}) {
                if (conv) { out.projection(conv->kernel_projection); }
            }
        }
        if (weights.draft->selector) { out.projection(weights.draft->selector->hidden_projection); }
    }
    return out;
}

} // namespace

void prepack_volta_qpn(const ModelWeights& weights, std::span<const PendingWeight> pending,
                       artifact::MaterializedArtifact& backing, DeviceContext& device) {
    const Selection selection = select(weights);

    // Every logical parameter that borrows a device parent, so a shared parent is permuted only
    // when all of its consumers read the prepacked order.
    std::map<std::size_t, std::vector<std::size_t>> consumers;
    for (std::size_t parameter = 0; parameter < pending.size(); ++parameter) {
        const auto& reference = pending[parameter].reference;
        if (reference.residency != artifact::Residency::Device) { continue; }
        for (const auto& part : reference.binding.parts) {
            auto& users = consumers[part.object.index];
            if (users.empty() || users.back() != parameter) { users.push_back(parameter); }
        }
    }

    for (const auto& [object, users] : consumers) {
        const artifact::ObjectHandle handle{object};
        if (!backing.has_device(handle)) { continue; }
        const WeightParent& parent = backing.device_parent(handle);
        const auto& geometry       = parent.geometry;
        const bool fp8             = geometry.format == QType::FP8_E4M3FN_ROW_BF16 &&
                         geometry.layout == QuantLayout::RowScale;
        const bool nvfp4 =
            geometry.format == QType::NVFP4 && geometry.layout == QuantLayout::BlockScaleK16M128x4;
        if (!fp8 && !nvfp4) { continue; }
        const auto& eligible = fp8 ? selection.fp8 : selection.nvfp4;
        bool selected        = true;
        for (const std::size_t user : users) { selected = selected && eligible.contains(user); }
        if (!selected || geometry.shape.size() != 2 || geometry.shape[0] % 32 != 0 ||
            (fp8 && geometry.shape[1] % 128 != 0)) {
            continue;
        }

        const WeightView whole{geometry.shape, {{&parent, 0, geometry.elements}}};
        Weight weight = native_weight(whole, 1.0F);
        if (fp8) {
            ops::detail::fp8_prepack_qpn_sm70(weight, device.stream);
        } else {
            ops::detail::nvfp4_prepack_qpn_sm70(weight, device.stream);
        }
        backing.mark_volta_qpn_prepacked(handle);
    }
}

} // namespace ninfer::models::qwen3_5::loading
