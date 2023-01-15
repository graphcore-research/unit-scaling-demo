#include <cmath>
#include <sstream>

#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Util.hpp>

namespace {
std::istream& operator>>(std::istream& in, poplar::QuarterMetadata::Format& metadata) {
    std::string name;
    in >> name;
    if (name == "1.4.3") {
        metadata = poplar::QuarterMetadata::Format::F143;
    } else if (name == "1.5.2") {
        metadata = poplar::QuarterMetadata::Format::F152;
    } else {
        in.setstate(std::ios::badbit);
    }
    return in;
}

float maxRepresentableValue(poplar::QuarterMetadata::Format format, int bias) {
    auto expBits = (format == poplar::QuarterMetadata::Format::F143 ? 4 : 5);
    auto maxMantissaValue = 2.f - std::pow(2.f, expBits - 7);
    auto maxExponentValue = std::pow(2.f, (1 << (expBits - 1)) - 1 + bias);
    return maxExponentValue * maxMantissaValue;
}
}  // namespace

extern "C" {
int32_t custom_op_api_level = 5;
}

extern "C" void Build_metadata(
    std::vector<std::int64_t>& /*allocating_indices*/,
    std::vector<std::int64_t>& /*replica_identical_output_indices*/,
    std::map<std::int64_t, std::int64_t>& /*input_to_output_tensor_aliasing*/,
    bool& /*is_elementwise*/,
    bool& is_stateless,
    bool& is_hashable,
    std::uint32_t /*num_inputs*/
) {
    is_stateless = true;
    is_hashable = true;
}

extern "C" poplar::program::Program Build(poplar::Graph& graph,
                                          const std::vector<poplar::Tensor>& inputs,
                                          std::vector<poplar::Tensor>& outputs,
                                          const std::string& attributes,
                                          const std::string& debugPrefix) {
    poplar::program::Sequence program;
    poplar::DebugContext di(debugPrefix);

    poplar::QuarterMetadata::Format format(poplar::QuarterMetadata::Format::F143);
    int exponentBias;
    std::istringstream in(attributes);
    in >> format >> exponentBias;
    if (!in.eof()) {
        std::ostringstream msg;
        msg << "Cannot read attribute '" << attributes << "'";
        throw std::invalid_argument(msg.str());
    }

    auto input = inputs.at(0);
    auto casted = input;

    // 1. Cast to float16
    if (input.elementType() == poplar::FLOAT) {
        casted = popops::cast(graph, casted, poplar::HALF, program, {di, "fp32_to_fp16"});
    }

    // 2. Clamp
    auto limit = maxRepresentableValue(format, exponentBias);
    auto minLimit = graph.addConstant(poplar::HALF, {}, -limit, {di, "min"});
    graph.setTileMapping(minLimit, 0);
    auto maxLimit = graph.addConstant(poplar::HALF, {}, limit, {di, "max"});
    graph.setTileMapping(maxLimit, 0);
    casted = popops::clamp(graph, casted, minLimit, maxLimit, program, {di, "clamp"});

    // 3. Cast to float8
    auto metadata = poputil::createConstantMetadataTensor(graph, format, exponentBias);
    casted = popops::cast(graph, casted, poplar::QUARTER, metadata, program, {di, "fp16_to_fp8"});

    // 4. Cast to float16
    casted = popops::cast(graph, casted, poplar::HALF, program, {di, "fp8_to_fp16"});

    // 5. Cast to float32
    if (input.elementType() == poplar::FLOAT) {
        casted = popops::cast(graph, casted, poplar::FLOAT, program, {di, "fp16_to_fp32"});
    }

    outputs = {casted};
    return program;
}
