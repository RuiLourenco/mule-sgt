#include "LightField/BlockCollage.h"

void BlockCollage::logBlockSizes() const {
    // for(size_t i = 0; i < blocks.size(); ++i) {
    //     const auto& block = blocks[i];
    //     ABSL_VLOG(3) << "Includes Invalid Corners: " << block.includesInvalidCorners;
    //     ABSL_VLOG(3) << "Valid Positions: " << block.validPositions.size(0);

    //     ABSL_VLOG(3) << "[Debug] Block " << i << " transform size: " 
    //                 << block.transformSize[0] << "x" 
    //                 << block.transformSize[1] << "x" 
    //                 << block.transformSize[2] << "x" 
    //                 << block.transformSize[3];
    //     ABSL_VLOG(3) << "[Debug] Block"<< i <<" Real Size: "
    //                 << block.data.size(0) << "x"
    //                 << block.data.size(1) << "x"
    //                 << block.data.size(2) << "x"
    //                 << block.data.size(3);
    // }
}

void BlockCollage::verifyBlockSize(const std::array<BlockCollage,4>& inputCollages) const{
    int x1 = 2; int x2 = 3;
    if (inputCollages[0].size[2] == inputCollages[3].size[2] && inputCollages[0].size[3] == inputCollages[1].size[3] &&
        (inputCollages[0].size[0] != inputCollages[3].size[0] || inputCollages[0].size[1] != inputCollages[1].size[1])) {
        x1 = 0; x2 = 1;
    }
    assert(inputCollages[0].size[x1]+inputCollages[3].size[x1] == inputCollages[1].size[x1]+inputCollages[2].size[x1] && "dimension x1 doesn't match" );
    assert(inputCollages[0].size[x2]+inputCollages[1].size[x2] == inputCollages[3].size[x2]+inputCollages[2].size[x2] && "dimension x2 doesn't match");
}

void BlockCollage::getSizeFromBlocks(const std::array<BlockCollage, 4>& inputCollages) {
    verifyBlockSize(inputCollages);
    
    int x1 = 2; 
    int x2 = 3;
    if (inputCollages[0].size[2] == inputCollages[3].size[2] && inputCollages[0].size[3] == inputCollages[1].size[3] &&
        (inputCollages[0].size[0] != inputCollages[3].size[0] || inputCollages[0].size[1] != inputCollages[1].size[1])) {
        x1 = 0; x2 = 1;
    }

    // Start with the base size from the top-left collage
    this->size = inputCollages[0].size;
    
    // Add height/depth from the block below (Index 3 is Bottom-Left)
    this->size[x1] += inputCollages[3].size[x1];
    
    // Add width from the block to the right (Index 1 is Top-Right)
    this->size[x2] += inputCollages[1].size[x2];
}

BlockCollage::BlockCollage(std::array<BlockCollage, 4>&& inputCollages) {
    // 1. Compute spatial metadata FIRST (before we gut the input objects)
    // Note: getSizeFromBlocks takes a const reference, which safely binds to our rvalue
    this->getSizeFromBlocks(inputCollages);
    
    // Position of the collage is the position of the Top-Left corner
    this->lightFieldPosition = inputCollages[0].lightFieldPosition;

    // 2. Efficiency: Reserve memory once to prevent multiple reallocations
    size_t totalBlocks = 0;
    for (const auto& c : inputCollages) {
        totalBlocks += c.blocks.size();
    }
    this->blocks.reserve(totalBlocks);

    // 3. Flatten blocks into the master list via MOVE semantics
    for (auto& c : inputCollages) { // CRITICAL: 'auto&', not 'const auto&'
        this->blocks.insert(
            this->blocks.end(), 
            std::make_move_iterator(c.blocks.begin()), 
            std::make_move_iterator(c.blocks.end())
        );
    }
}

// Change signature to take an rvalue reference (&&)
BlockCollage::BlockCollage(Block4D_&& block) {
    this->size = block.size;
    this->lightFieldPosition = block.lightFieldPosition;
    
    // std::move "steals" the internal pointers from the block
    this->blocks.push_back(std::move(block));
}

const Block4D_& BlockCollage::getBlock(size_t index) const {
    return blocks.at(index); // .at() provides bounds checking for safety
}
int BlockCollage::getNumBlocks() const {
    return blocks.size();
}
