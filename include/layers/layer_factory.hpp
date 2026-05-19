#ifndef MINI_INFERENCE_ENGINE_LAYERS_LAYER_FACTORY_HPP_
#define MINI_INFERENCE_ENGINE_LAYERS_LAYER_FACTORY_HPP_

#include "ops/op.hpp"
#include "layers/layer.hpp"
#include <memory>
#include <map>

namespace mini_inference_engine {

class LayerFactory {
public:
    typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Op>& op);
    typedef std::map<OpType, Creator> CreatorRegistry;

    static void RegisterCreator(OpType op_type, const Creator& creator);

    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Op>& op);

    static CreatorRegistry& GetRegistry();
};

class LayerRegistrar {
public:
    LayerRegistrar(OpType op_type, const LayerFactory::Creator& creator) {
        LayerFactory::RegisterCreator(op_type, creator);
    }
};

}

#endif
