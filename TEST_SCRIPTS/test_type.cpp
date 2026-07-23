#include <torch/torch.h>
#include <iostream>

int main() {
    auto tensor_int = torch::ones({2, 2}, torch::kInt);
    double scale = 304.8;
    auto result = scale * tensor_int;
    std::cout << "Result dtype: " << result.dtype() << std::endl;
    return 0;
}
