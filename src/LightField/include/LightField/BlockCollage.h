#ifndef __BLOCK_COLLAGE_H__
#define __BLOCK_COLLAGE_H__

#include <vector>
#include <array>
#include <cstdint>
#include "Block4D_.h"


class BlockCollage {
    // This class is used to store the collage of the transformed blocks. It is used to store the transformed blocks in a way that allows us to easily access them for the inverse transform.
    // It is also used to store the transformed blocks in a way that allows us to easily access them for the RD optimization.
    //This is meant to be a "dumb" class. The encoder should be responsible for the ordering of the blocks and make sure it matches decoder logic.
    // Still smarter than storing everything in a single tensor as it prevents issues with different-sized partitions.
    std::vector<Block4D_> blocks;
    std::array<int64_t,4> size;
    std::array<int64_t,4> lightFieldPosition;
    void getSizeFromBlocks(const std::array<BlockCollage,4>& blocks);
    void verifyBlockSize(const std::array<BlockCollage,4>& blocks) const;
    public:
        BlockCollage() = default;
        BlockCollage(std::array<BlockCollage, 4>&& blocks);
        BlockCollage(Block4D_&& block);
        void addCollage(const BlockCollage& collage);
        const Block4D_& getBlock(size_t index) const;
        int getNumBlocks() const;
        void logBlockSizes() const;
};

#endif