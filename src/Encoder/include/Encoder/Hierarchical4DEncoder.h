/* 
 * File:   Hierarchical4DEncoder.h
 * Author: murilo
 *
 * Created on December 28, 2017, 11:41 AM
 */
#include "LightField/Block4D.h"
#include "LightField/Block4D_.h"
#include "Encoder/ABACoder.h"
#include "ProbabilityModel/ProbabilityModel.h"



#ifndef HIERARCHICAL4DENCODER_H
#define HIERARCHICAL4DENCODER_H

//#define MAX_DEPH_CONDICIONING 9
#define BITPLANE_BYPASS -1
#define BITPLANE_BYPASS_FLAGS -1
#define SYMBOL_PROBABILITY_MODEL_INDEX 1
#define SEGMENTATION_PROB_MODEL_INDEX 32
#define NUMBER_OF_MODELS 161



// Forward declarations remain the same
class ModelBufferHandle; 

class ProbabilityModelArena {
public:
    // The constructor now takes the required size at run-time.
    explicit ProbabilityModelArena(size_t max_depth) 
        : m_active_count(0),
          m_max_depth(max_depth) // Store the max depth
    {
        // This is the single heap allocation. It happens ONCE.
        m_pool.resize(m_max_depth * NUMBER_OF_MODELS);
    }

    ModelBufferHandle get_buffer();

private:
    friend class ModelBufferHandle;

    // The pool is a dynamic vector, managing its own memory.
    std::vector<ProbabilityModel> m_pool; 
    size_t m_active_count;
    size_t m_max_depth;
};
// --- The Handle Class (RAII Guard / Lease) ---
class ModelBufferHandle {
public:
    // Destructor is the key: it automatically returns the resource.
    ~ModelBufferHandle() {
        m_arena_counter_ref--; 
    }

    // Overload -> to make it act like a pointer
    ProbabilityModel* operator->() { return m_buffer_ptr; }
    // Overload * to get the underlying array
    ProbabilityModel(&operator*())[NUMBER_OF_MODELS] { 
        return *(reinterpret_cast<ProbabilityModel (*)[NUMBER_OF_MODELS]>(m_buffer_ptr)); 
    }
    // A simple getter for the raw pointer if needed (e.g., for memcpy)
    ProbabilityModel* get() { return m_buffer_ptr; }

    // Disable copying to prevent errors. A lease should not be copied.
    ModelBufferHandle(const ModelBufferHandle&) = delete;
    ModelBufferHandle& operator=(const ModelBufferHandle&) = delete;

private:
    // Only the Arena can create a handle.
    friend class ProbabilityModelArena; 
    ModelBufferHandle(ProbabilityModel* buffer, size_t& counter) 
        : m_buffer_ptr(buffer), m_arena_counter_ref(counter) {}

    ProbabilityModel* m_buffer_ptr;
    size_t& m_arena_counter_ref;
};

inline ModelBufferHandle ProbabilityModelArena::get_buffer() {
    if (m_active_count >= m_max_depth) {
        throw std::runtime_error("ProbabilityModelArena: Exceeded pre-calculated maximum depth!");
    }
    // Get the address of the start of the next buffer slice in our 1D vector.
    ProbabilityModel* buffer_start = &m_pool[m_active_count * NUMBER_OF_MODELS];
    
    // Increment the counter and return the handle.
    m_active_count++;
    return ModelBufferHandle(buffer_start, m_active_count);
}


struct CostResults {
    double cost = 0.0;
    double signalEnergy = 0.0;};

const uint32_t NULL_NODE = static_cast<uint32_t>(-1);
struct Node {
    CostResults   costResults;          // 16 bytes
    char     decision = 'X';                      // 1 byte
    // 3 bytes of compiler padding will likely go here
    uint32_t children_idx[4] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE}; // 16 bytes
};

// This struct holds the state and the reusable memory pool.
struct ProcessingContext {
    std::vector<Node> nodePool;
    const size_t image_width;
    const size_t image_height;
    const int initial_bitdepth;
    int next_available_idx = 0; // Index for the next available node in the pool

    size_t calculate_depth(size_t dimension) {
        if (dimension == 0) return 0;
        size_t depth = 0;
        size_t dim = 1;
        while (dim < dimension) {
            dim *= 2;
            depth++;
        }
        return depth;
    }
    
    // Calculates the total nodes in a full quadtree of a given depth.
    size_t total_nodes_in_quadtree(size_t depth) {
        // Using a 128-bit integer to prevent overflow during the geometric sum calculation,
        // as 4^32 can exceed a 64-bit integer.
        unsigned __int128 total = 0;
        unsigned __int128 term = 1;
        for (size_t i = 0; i <= depth; ++i) {
            total += term;
            term *= 4;
        }
        return static_cast<size_t>(total);
    }
    void resetCounter() {
        next_available_idx = 0; // Reset the index to reuse the buffer
    }

    // The constructor takes runtime parameters and allocates the buffer ONCE.
    ProcessingContext(size_t height, size_t width, int bitdepth) : 
        image_width(width), image_height(height), initial_bitdepth(bitdepth) {

        // 1. Determine the largest dimension to find the required tree depth.
        size_t max_dimension = std::max(width, height);
        size_t tree_depth = calculate_depth(max_dimension);

        // 2. Calculate the number of nodes in the full spatial quadtree structure.
        size_t n_quad = total_nodes_in_quadtree(tree_depth);

        // 3. Get the number of actual leaf nodes (the true image size).
        size_t n_leaves_actual = width * height;

        // 4. Calculate the final, tight upper bound for the number of nodes.
        size_t max_nodes = n_quad + (n_leaves_actual * (initial_bitdepth)); // Simplified from (B-1) for safety

        std::cout << "--- Processing Context Initialized ---" << std::endl;
        std::cout << "Input: " << width << "x" << height << ", " << bitdepth+1 << " bitdepth levels" << std::endl;
        std::cout << "Required Tree Depth: " << tree_depth << std::endl;
        std::cout << "Max Nodes Required (Upper Bound): " << max_nodes << std::endl;
        std::cout << "Node size: " << sizeof(Node) << " bytes" << std::endl;
        std::cout << "Allocating reusable buffer of ~" << (static_cast<uint64_t>(max_nodes) * sizeof(Node)) / (1024*1024) << " MB..." << std::endl;
        
        // 5. This is the single, large, heap allocation for the lifetime of the context.
        nodePool.resize(max_nodes);
        std::cout << "--------------------------------------" << std::endl;
    }
    uint32_t add_default_node(){
        // This function adds a default node with cost 0 and decision 'L' (Low Energy).
        if (next_available_idx >= nodePool.size()) {
            throw std::runtime_error("Node pool exhausted, increase the initial size.");
        }
        uint32_t idx = next_available_idx++;
        nodePool[idx].costResults.cost = 0.0;
        nodePool[idx].costResults.signalEnergy = 0.0;
        nodePool[idx].decision = ' '; // 'L' for Low Energy
        nodePool[idx].children_idx[0] = NULL_NODE;
        nodePool[idx].children_idx[1] = NULL_NODE;
        nodePool[idx].children_idx[2] = NULL_NODE;
        nodePool[idx].children_idx[3] = NULL_NODE;
        return idx;
    }
    

};






struct HexResult {
    double cost = 0.0;
    std::string codeStream; // Each node will return its own piece of the code stream
};
class Hierarchical4DEncoder {
    ProbabilityModelArena mModelArena; 
public:
    ProcessingContext mProcessingContext;
    double mLambda = 0;
    double mRate  = 0;
    double mDistortion = 0;
    std::array<uint64_t,4> size;
    Block4D_ mSubbandLF_;   
    at::Tensor ignored;
    double currCost;
    ABACoder mEntropyCoder;
    ProbabilityModel *mPmodel;
    ProbabilityModel *mOptimizationPmodel;
    int flagZero = 0;
    int flagOne = 0;
    int flagTwo = 0;
    int mIgnored = 0;
    double mIgnoreEfficiency;
    int mSuperiorBitPlane, mInferiorBitPlane;
    int mSegmentationFlagProbabilityModelIndex;
    int mSymbolProbabilityModelIndex;
    int mPreSegmentation;
    std::string mSegmentationTreeCodeBuffer;
    long int mSegmentationTreeCodeBufferSize;
    int OptimumBitplaneFaster_(double lambda);
    Hierarchical4DEncoder(void);
    Hierarchical4DEncoder(int height, int width);
    ~Hierarchical4DEncoder(void);

    void encodeSubblockFromPool(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane, double lambda);
    void iterateEncoding(uint32_t current_node_idx, std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane) ;

    CostResults splitInFour(uint32_t current_node_idx,std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane);
    CostResults calculateTotalEnergy(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane);
    double build_optimal_tree_from_pool(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane, double lambda);
    void build_from_node(uint32_t current_node_idx,std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane);
    CostResults calculateElementCost(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane);
    bool checkSignificance(std::array<int64_t,4> position, std::array<int64_t,4> length, int bitplane);
    void StartEncoder(FILE *outputFilePointer);
    void RestartProbabilisticModel(void);
    void EncodeBlock(int position_t, int position_s, int position_v, int position_u, int length_t, int length_s, int length_v, int length_u, int bitplane);
    void EncodeBlock_(std::array<int64_t,4> position,std::array<int64_t, 4> length, int bitplane);
    void EncodeCoefficient(int coefficient, int bitplane);
    void EncodeSegmentationFlag(int flag, int bitplane);
    void EncodePartitionFlag(int flag);
    void EncodeSSI_(SgtSideInfo ssi);
    void EncodeInteger(int integerValue, int precision);
    void EncodeAll(double lambda, int inferiorBitPlane);
    void EncodeSubblock_(double lambda);
    HexResult RdOptimizeHexadecaTree_(std::array<int64_t,4> position,std::array<int64_t, 4> length, double lambda, int bitplane, double &signalEnergy,double& rate, double& distortion);
    void RdEncodeHexadecatree_(std::array<int64_t,4> position,std::array<int64_t, 4> length, int bitplane, int &flagIndex);
    void DoneEncoding(void);
    void LoadOptimizerState(void);
    void GetOptimizerProbabilisticModelState(ProbabilityModel **state);
    void SetOptimizerProbabilisticModelState(ProbabilityModel *state);
    void DeleteProbabilisticModelState(ProbabilityModel *state);
};

static inline void copyNProbabilityModels(
    ProbabilityModel* destination, 
    const ProbabilityModel* source, 
    size_t count
) {
    memcpy(destination, source, count * sizeof(ProbabilityModel));
}


// THE CONVENIENT WRAPPER:
// A specific function for the most common use case.
static inline void copyOptimizationModels(
    ProbabilityModel* destination, 
    const ProbabilityModel* source
) {
    // Calls the general workhorse with the known constant.
    copyNProbabilityModels(destination, source, NUMBER_OF_MODELS);
}
#endif /* HIERARCHICAL4DENCODER_H */

