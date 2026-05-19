#include "layers/layer_factory.hpp"
#include <glog/logging.h>

namespace mini_inference_engine {

void LayerFactory::RegisterCreator(OpType op_type, const Creator& creator) {
    CHECK(creator != nullptr) << "Layer creator is empty";
    
    CreatorRegistry& registry = GetRegistry();

    CHECK_EQ(registry.count(op_type), 0)
        << "Layer type: " << int(op_type) << " has already been registered!";
    
    registry.insert({op_type, creator});
}

std::shared_ptr<Layer> LayerFactory::CreateLayer(const std::shared_ptr<Op>& op) {
    CreatorRegistry& registry = GetRegistry();
    const OpType op_type = op->op_type_;

    auto it = registry.find(op_type);
    LOG_IF(FATAL, it == registry.end()) 
        << "Can not find the layer type: " << int(op_type);

    const auto& creator = it->second;
    LOG_IF(FATAL, !creator) << "Layer creator is empty!";

    std::shared_ptr<Layer> layer = creator(op);
    LOG_IF(FATAL, !layer) << "Layer initialization failed!";
    
    return layer;
}

LayerFactory::CreatorRegistry& LayerFactory::GetRegistry() {
    static CreatorRegistry* registry = new CreatorRegistry();
    CHECK(registry != nullptr) << "Global layer registry initialization failed!";

    return *registry;
}

}
