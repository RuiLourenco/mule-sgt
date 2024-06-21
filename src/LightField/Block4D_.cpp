#include "Block4D_.h"

Block4D_::Block4D(std::array<int,4> size) {
    data = torch::zeros({size[0], size[1], size[2], size[3]});
}
Block4D_::Block4D(const Block4D_& B00, const Block4D_& B01, const Block4D_& B10, const Block4D_& B11) {
    data = torch::cat({torch::cat({B00.data, B01.data}, 0),torch::cat({B10.data, B11.data}, 0)},1);
}