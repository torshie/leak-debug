#include <cstdio>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>


int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: leak_debug <model> <number>\n");
        return 1;
    }

    Ort::Env env{ORT_LOGGING_LEVEL_INFO, ""};

    Ort::SessionOptions options;
    Ort::ThrowOnError(
         OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
    options.SetGraphOptimizationLevel(
         GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(1);
    options.DisableCpuMemArena();
    options.DisableMemPattern();

    Ort::Session session{env, argv[1], options};

    std::array<std::int64_t, 4> shape{1, 1080, 1920, 3};
    std::vector<std::uint8_t> input_values(1080 * 1920 * 3);

    const char* input_names[] = {"COALER_raw_input"};
    const char* output_names[] = {"COALER_bbox", "COALER_label",
         "COALER_confidence"};

    int count = std::atoi(argv[2]);

    for (int i = 0; i < count; ++i) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
             OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<uint8_t>(memory_info,
             input_values.data(), input_values.size(), shape.data(), 4);
        session.Run(Ort::RunOptions{nullptr}, input_names, &input,
                1, output_names, 3);
    }

    return 0;
}
