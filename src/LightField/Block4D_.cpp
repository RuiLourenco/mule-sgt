#include "LightField/Block4D_.h"
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>
#include <opencv2/opencv.hpp>
#include "osqp.h"


#include <boost/range/irange.hpp>
#include <boost/range/join.hpp>

#include <boost/phoenix.hpp>
#include <limits>




using namespace boost;
using namespace boost::adaptors;
using int_arr_t = std::vector<std::int64_t>;
using tensor_arr_t = std::vector<at::Tensor>;
template<class Container> class to_t {};
template<class Range, class C>
auto operator|(Range&& r, to_t<C>) { return copy_range<C>(std::forward<Range>(r)); };



std::pair<at::Tensor, at::Tensor> make_function_grid(at::IntArrayRef sizes, at::ScalarType dtype = c10::ScalarType::Double) {
    using namespace phoenix::placeholders;

    for (const auto& sz : sizes) {
        if (sz <= 0) throw std::runtime_error("Sizes must be positive integers");
    }

    auto opts = at::TensorOptions{}.dtype(dtype).device(at::kCPU);
    auto ranges = sizes
        | transformed([&](auto&& sz){return at::arange(1-sz, sz, opts);})| to_t<tensor_arr_t>{};




#define DEBUG 0
#define MATLAB_DEBUG 0

void saveVectorAsMatlabScript(std::vector<double> vector,std::string name){
#if MATLAB_DEBUG == 1
    std::ofstream stuff;
    std::cout<<"YOH"<<std::endl;
    stuff.open("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/zMatlabFiles/"+name+".m");  
    stuff<<name+"_cpp = [";
    for(int n = 0; n < vector.size(); n++){
        if(n != 0) stuff<<",";
        stuff<<vector[n];
    }
    stuff<<"];"<<std::endl;
#else

#endif
}
void saveTensorAsMatlabScript(at::Tensor tensor,std::string name){
#if MATLAB_DEBUG == 1

    std::ofstream stuff;
    stuff.open("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/zMatlabFiles/"+name+".m");
    stuff<<name+"_cpp = [";
    for(int n = 0; n < tensor.size(0); n++){
        if(n != 0) stuff<<";"<<std::endl;
        for(int m = 0; m < tensor.size(1); m++){
            if (m != 0) stuff<<",";
            stuff<<tensor[n][m].item();
        }
    }
    stuff<<"];"<<std::endl;
#else

#endif
}

cv::Mat torchToCv(const torch::Tensor& tensor) {
    // Get tensor shape
    auto sizes = tensor.sizes();
    int height = sizes[0];
    int width = sizes[1];

    // Check if tensor is of type float64
    //cout<<"Tensor Type is "<<tensor.dtype()<<endl;
    tensor.to(at::kDouble);
    //TORCH_CHECK(tensor.dtype() == torch::kDouble, "Tensor data type must be double.");
    cv::Mat image(height, width, CV_64FC1, cv::Scalar(0));
    for( int h = 0 ; h < height; h++){
      for( int w = 0; w< width;w++){
        image.at<double>(h,w) = tensor[h][w].item<double>();
      }
    }
    return image;
}

void write_image(const cv::Mat& image,std::string path,std::array<double,2> valueRange = {1,1}){ 
    cv::Mat display_image;
    double min_val; 
    double max_val; 
    cv::Point min_loc; 
    cv::Point max_loc;
    if(valueRange[0] == valueRange[1]){ 
        minMaxLoc( image, &min_val, &max_val, &min_loc, &max_loc );
    }
    else{
        min_val = valueRange[0];
        max_val = valueRange[1];
    }
    std::cout<<path<<" Min Val: "<<min_val<<" Max Val: "<<max_val<<std::endl;
    display_image = min(max(image,min_val), max_val);
    display_image -= min_val;
    display_image /= max_val - min_val;
    display_image *= 255;
    display_image.convertTo(display_image,CV_8U);
    cv::Mat img_color;
    // Apply the colormap:
    applyColorMap(display_image, img_color, cv::COLORMAP_JET);
    // Write the image
    cv::imwrite(path, img_color);
    //cv::imwrite(path, display_image);
}
void write_tensor(torch::Tensor tensor, std::string path, std::array<double,2> valueRange){
#if DEBUG == 1
  cv::Mat mat = torchToCv(tensor);
  write_image(mat,path,valueRange);
#endif
}


#include <torch/torch.h>
#include <osqp/osqp.h>

// Function to convert a dense matrix (torch tensor) to CSC format
void convertToSparseCSC(const torch::Tensor& denseTensor, std::vector<double> &values, std::vector<c_int> &indices_row, std::vector<c_int> &indptr) {
    // Ensure the input tensor is a 2D tensor
    if (denseTensor.dim() != 2) {
        throw std::invalid_argument("The input tensor must be a 2D tensor.");
    }

    int64_t rows = denseTensor.size(0);
    int64_t cols = denseTensor.size(1);

    // Convert the tensor to a dense format (if it's not already dense)
    auto dense = denseTensor.contiguous();

    // Step 1: Collect the non-zero elements (values) and their indices

    std::vector<c_int> indices_col;

    for (int64_t col = 0; col < cols; ++col) {
        for (int64_t row = 0; row < rows; ++row) {
            if (dense[row][col].item<float>() != 0.0f) {
                values.push_back(dense[row][col].item<float>());
                indices_row.push_back(row);
                indices_col.push_back(col);
            }
        }
    }

    // Step 2: Build the `indptr` (Cumulative counts of non-zero entries in each column)
    
    for (int64_t i = 0; i < indices_col.size(); ++i) {
        indptr[indices_col[i] + 1]++;
    }

    // Perform cumulative sum on `indptr`
    for (int64_t col = 1; col <= cols; ++col) {
        indptr[col] += indptr[col - 1];
    }

    

    // You can return these tensors as a tuple or just one of them
    // Here we will just return values, indices_row, and indptr


}

at::Tensor SgtSideInfo::QPOptimization(at::Tensor P, at::Tensor q){
    // Ensure P and q are contiguous
    P = P.contiguous().to(torch::kDouble);
    q = q.contiguous().to(torch::kDouble);
    at::Tensor A = torch::eye(2,at::kDouble); 
    std::vector<double> valuesA;
    std::vector<c_int> indices_rowA;
    std::vector<c_int> indptrA(A.size(1) + 1, 0);
    convertToSparseCSC(A,valuesA,indices_rowA,indptrA);
    csc* A_sparse = csc_matrix(A.size(0), A.size(1), valuesA.size(), valuesA.data(), indices_rowA.data(), indptrA.data());
    // Get raw pointers to P and q
    //double* P_data = P.data_ptr<double>();
    double* q_data = q.data_ptr<double>();

    // Inequality constraints: 0 <= x <= 0.99999
    //torch::Tensor A_constraint = torch::ones({n,n}, torch::kDouble); // Identity matrix for bounds
    // torch::Tensor l = torch::tensor({0.,0.,-OSQP_INFTY,-OSQP_INFTY}, torch::kDouble);       // Lower bounds (0)
    // torch::Tensor u = torch::tensor({OSQP_INFTY,OSQP_INFTY,0.99999f,0.99999f}, torch::kDouble); // Upper bounds (0.99999)

    // // Ensure A_constraint, l, and u are contiguous
    // A_constraint = A_constraint.contiguous();
    // l = l.contiguous();
    // u = u.contiguous();



    // Get raw pointers to A_constraint, l, and u
    //double* A_constraint_data = A_constraint.data_ptr<double>();
    double l_data[2] = {-0.8,-0.8};
    double u_data[2] = {-log(0.99999),-log(0.99999)};
    //std::cout<<l_data[0]<<" "<<u_data[0]<<std::endl;
    std::vector<double> values;
    std::vector<c_int> indices_row;
    std::vector<c_int> indptr(P.size(1) + 1, 0);
    convertToSparseCSC(P.triu(),values,indices_row,indptr);
    // std::cout<<"Values: ";
    // for(int i = 0;i<values.size();i++){
    //     std::cout<<values[i]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"indices_row: ";
    // for(int i = 0;i<indices_row.size();i++){
    //     std::cout<<indices_row[i]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"indptr: ";
    // for(int i = 0;i<indptr.size();i++){
    //     std::cout<<indptr[i]<<" ";
    // }
    // std::cout<<std::endl;

    csc* P_sparse = csc_matrix(P.size(0), P.size(1), values.size(), values.data(), indices_row.data(), indptr.data());
    //std::cout<<P_sparse->x[values.size()-1]<<" "<<P_sparse->i[values.size()-1]<<" "<<P_sparse->p[0]<<std::endl;
    //std::cout<<P[0][0]<<std::endl;
    // OSQP data
    OSQPData data;
    data.n = 2;
    data.m = 2; // Number of constraints
    data.P = P_sparse; // Dense matrix
    data.q = q_data;
    data.A = A_sparse;
    //data.A = convertToSparseCSC(A_constraint); // Dense matrix
    data.l = l_data;
    data.u = u_data;
    //std::cout<<"I have initialized optimization data!"<<std::endl;
    // OSQP settings
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.eps_abs = 1e-8;  // Absolute tolerance
    settings.eps_rel = 1e-8;  // Relative tolerance
    settings.eps_prim_inf = 1e-8; // Primal infeasibility tolerance
    settings.eps_dual_inf = 1e-8; // Dual infeasibility tolerance
    settings.polish = 1; // Enable solution polishing
    settings.warm_start = 0;
    settings.rho = 0.000001;

    settings.verbose = 0; // Disable verbose output

    // OSQP workspace
    OSQPWorkspace* work;
    //std::cout<<"TAG1"<<std::endl;

    osqp_setup(&work, &data, &settings);
    //std::cout<<"I have setup optimization data!"<<std::endl;


    // Solve the problem
    osqp_solve(work);

    // Extract the solution
    torch::Tensor x_torch = torch::from_blob(work->solution->x, {2}, torch::kDouble).clone();
    //std::cout << "Solver status: " << work->info->status << std::endl;
    //std::cout<<"x_torch: "<<x_torch.unsqueeze(0)<<std::endl;
    // Clean up
    osqp_cleanup(work);

    return x_torch;
}
// Function to solve constrained least squares using OSQP
at::Tensor SgtSideInfo::constrainedLeastSquares(at::Tensor A, at::Tensor b){
    
    // Ensure A and b are 2D and 1D tensors respectively
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(b.squeeze().dim() == 1, "b must be a 1D tensor");

    // Get dimensions
    int m = A.size(0); // Number of rows in A
    int n = A.size(1); // Number of columns in A
    //std::cout<<A.sizes()<<std::endl;

    // Ensure A and b are contiguous and of type float
    torch::Tensor A_contiguous = A.contiguous().to(torch::kDouble);
    torch::Tensor b_contiguous = b.contiguous().to(torch::kDouble);

    // Get raw pointers to the data
    double* A_data = A_contiguous.data_ptr<double>();
    double* b_data = b_contiguous.data_ptr<double>();

    // Construct the QP problem
    // Minimize 0.5 * x^T * P * x + q^T * x
    // Subject to l <= A * x <= u

    // P = A^T * A
    torch::Tensor P = torch::matmul(A.t(), A);

    // q = -A^T * b
    torch::Tensor q = -torch::matmul(A.t(), b);

    // // Ensure P and q are contiguous
    // P = P.contiguous().to(torch::kDouble);
    // q = q.contiguous().to(torch::kDouble);

    // // Get raw pointers to P and q
    // double* P_data = P.data_ptr<double>();
    // double* q_data = q.data_ptr<double>();

    // // Inequality constraints: 0 <= x <= 0.99999
    // torch::Tensor A_constraint = torch::ones({n,n}, torch::kDouble); // Identity matrix for bounds
    // // torch::Tensor l = torch::tensor({0.,0.,-OSQP_INFTY,-OSQP_INFTY}, torch::kDouble);       // Lower bounds (0)
    // // torch::Tensor u = torch::tensor({OSQP_INFTY,OSQP_INFTY,0.99999f,0.99999f}, torch::kDouble); // Upper bounds (0.99999)

    // // // Ensure A_constraint, l, and u are contiguous
    // // A_constraint = A_constraint.contiguous();
    // // l = l.contiguous();
    // // u = u.contiguous();



    // // Get raw pointers to A_constraint, l, and u
    // //double* A_constraint_data = A_constraint.data_ptr<double>();
    // double l_data[2] = {0.,0.};
    // double u_data[2] = {0.99999f,0.99999f};
    // std::cout<<l_data[0]<<" "<<u_data[0]<<std::endl;
    // std::vector<double> values;
    // std::vector<c_int> indices_row;
    // std::vector<c_int> indptr(P.size(1) + 1, 0);
    // convertToSparseCSC(P.triu(),values,indices_row,indptr);
    // std::cout<<P.triu().sizes()<<std::endl;
    // csc* P_sparse = csc_matrix(P.size(0), P.size(1), values.size(), values.data(), indices_row.data(), indptr.data());
    // std::cout<<P_sparse->x[values.size()-1]<<" "<<P_sparse->i[values.size()-1]<<" "<<P_sparse->p[0]<<std::endl;
    // std::cout<<P[0][0]<<std::endl;
    // // OSQP data
    // OSQPData data;
    // data.n = n;
    // data.m = 2; // Number of constraints
    // data.P = P_sparse; // Dense matrix
    // data.q = q_data;
    // //data.A = convertToSparseCSC(A_constraint); // Dense matrix
    // data.l = l_data;
    // data.u = u_data;
    // std::cout<<"I have initialized optimization data!"<<std::endl;
    // // OSQP settings
    // OSQPSettings settings;
    // osqp_set_default_settings(&settings);

    // settings.verbose = 1; // Disable verbose output

    // // OSQP workspace
    // OSQPWorkspace* work;
    // std::cout<<"TAG1"<<std::endl;

    // osqp_setup(&work, &data, &settings);
    // std::cout<<"I have setup optimization data!"<<std::endl;


    // // Solve the problem
    // osqp_solve(work);

    // // Extract the solution
    // torch::Tensor x_torch = torch::from_blob(work->solution->x, {n}, torch::kFloat32).clone();

    // // Clean up
    // osqp_cleanup(work);

    return QPOptimization(P,q);
}


void Block4D_::Extend_T(int length_t) {
    this->data.index({at::indexing::Slice(length_t,data.size(0)),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(length_t-1,length_t),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}).repeat({data.size(0)-length_t,1,1,1});
 
}
void Block4D_::Extend_S(int length_s) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s,data.size(1)),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s-1,length_s),at::indexing::Slice(),at::indexing::Slice()}).repeat({1,data.size(1)-length_s,1,1});
}
void Block4D_::Extend_V(int length_v) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v,data.size(2)),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v-1,length_v),at::indexing::Slice()}).repeat({1,1,data.size(2)-length_v,1});
}
void Block4D_::Extend_U(int length_u) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u,data.size(3))}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u-1,length_u)}).repeat({1,1,1,data.size(1)-length_u});
}
void Block4D_::clip(int min, int max) {
    this->data = this->data.clamp(min,max);
}

Block4D_::operator at::Tensor() const{
    return this->data;
}

// Block4D_::Block4D_(const at::Tensor& data){
//     this->data = data;
//     this->size = {data.size(0),data.size(1),data.size(2),data.size(3)};
// #if FLAT_TRANSFORM == 1
//     this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
// #else 
//     this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
// #endif
//     this->sgtDomain = false;
// }   
Block4D_::Block4D_(std::array<int64_t,4> size,std::array<int64_t,4>lightFieldPosition,LightField* lightField)
    : size(size), lightFieldPosition(lightFieldPosition), lightField(lightField){
    this->data = torch::zeros({size[0], size[1], size[2], size[3]}, torch::kInt);
    this->sgtDomain = false;
#if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
#else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
#endif
    
}
void Block4D_::emptyTransform(){
    this->data = torch::zeros({this->transformSize[0], this->transformSize[1], this->transformSize[2], this->transformSize[3]}, torch::kInt);
    this->sgtDomain = true;
}

/**
 * Constructs a Block4D_ object by concatenating the data from four equal sized Block4D_ objects in a specific pattern.
 *
 * @param B00 The first Block4D_ object.
 * @param B01 The second Block4D_ object.
 * @param B10 The third Block4D_ object.
 * @param B11 The fourth Block4D_ object.
 *
 * @return None
 *
 * @throws None
 */
Block4D_::Block4D_(const Block4D_& B00, const Block4D_& B01, const Block4D_& B10, const Block4D_& B11,bool views) {
    int x1,x2;
    if(views){
        x1 = 0; x2 = 1;
    }else{
        x1 = 2; x2 = 3;
    }
    //std::cout<<"B00: "<<B00.size[0]<<"x"<<B00.size[1]<<"x"<<B00.size[2]<<"x"<<B00.size[3]<<std::endl;
    //std::cout<<"B00 Transform Size: "<<B00.transformSize[0]<<"x"<<B00.transformSize[1]<<"x"<<B00.transformSize[2]<<"x"<<B00.transformSize[3]<<std::endl;
    assert(B00.data.size(x1)+B10.data.size(x1) == B01.data.size(x1)+B11.data.size(x1) && "heights don't match" );
    assert(B00.data.size(x2)+B01.data.size(x2) == B10.data.size(x2)+B11.data.size(x2) && "widths don't match");
    this->data = torch::cat({torch::cat({B00.data, B10.data}, x1),torch::cat({B01.data, B11.data}, x1)},x2).to(at::kInt);
    this->size = B00.size;
    this->size[x1] = B00.size[x1]+B10.size[x1];
    this->size[x2] = B00.size[x2]+B01.size[x2];
    //std::cout<<" Inside: "<<this->size[0]<<"x"<<this->size[1]<<"x"<<this->size[2]<<"x"<<this->size[3]<<std::endl;
    
#if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
#else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
#endif
//std::cout<<" Inside Size: "<<this->size[0]<<"x"<<this->size[1]<<"x"<<this->size[2]<<"x"<<this->size[3]<<std::endl;
//std::cout<<" Inside Transform Size: "<<this->transformSize[0]<<"x"<<this->transformSize[1]<<"x"<<this->transformSize[2]<<"x"<<this->transformSize[3]<<std::endl;

    this->sgtDomain = B00.sgtDomain;
    this->lightFieldPosition = B00.lightFieldPosition;
    this->lightField = B00.lightField;
}
std::ostream &operator<<(std::ostream &os, std::array<int64_t,4> vec) { 
    return os << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << "]";
}
std::array<int64_t,4> Block4D_::toSGTCoords(std::array<int64_t,4> coords) const{
#if FLAT_TRANSFORM == 1
    std::array<int64_t,4> sgtCoords = {coords[0]/size[0],coords[0]/size[0],size[0]*coords[2],size[1]*coords[3]}; //DOES NOT WORK FOR VIEW SPLITTING
#else
    std::array<int64_t,4> sgtCoords = coords;
#endif
    return sgtCoords;
}

Block4D_ Block4D_::copySubblock(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset){
    
    Block4D_ deepCopy = this->clone();
    //Block4D_ deepCopy2 = this->clone();

    for( int i = 0; i < 4; i++){
        deepCopy.lightFieldPosition[i] = this->lightFieldPosition[i] + sourceOffset[i];
    }
    std::array<int64_t,4> length = {std::min(subblockLength[0], this->size[0]-sourceOffset[0]),
                                    std::min(subblockLength[1], this->size[1]-sourceOffset[1]),
                                    std::min(subblockLength[2], this->size[2]-sourceOffset[2]),
                                    std::min(subblockLength[3], this->size[3]-sourceOffset[3])};
    
    deepCopy.size = length;
    
    deepCopy.transformSize = toSGTCoords(length);

    //std::cout<<"sourceOffsety B4: "<<sourceOffset[0]<<" "<<sourceOffset[1]<<" "<<sourceOffset[2]<<" "<<sourceOffset[3]<<std::endl;

    if(this->sgtDomain){
        length = deepCopy.transformSize;
        sourceOffset = toSGTCoords(sourceOffset);
    }
    //std::cout<<"lengthy: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"sourceOffsety After: "<<sourceOffset[0]<<" "<<sourceOffset[1]<<" "<<sourceOffset[2]<<" "<<sourceOffset[3]<<std::endl;

    deepCopy.data  = deepCopy.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });

    //std::cout<<"Actual Size After Copy: "<<deepCopy.data.sizes()<<std::endl;
    //std::cout<<"Size After Copy: "<<deepCopy.size<<std::endl;
    //std::cout<<"Transform Size After Copy: "<<deepCopy.transformSize<<std::endl;


       

    return deepCopy;


    
}
/**
 * Copies a subblock from another Block4D_ object to the current Block4D_ object.
 *
 * @param B The Block4D_ object from which the subblock is copied.
 * @param sourceOffset An array of 4 integers representing the starting index of the source subblock.
 * @param targetOffset An array of 4 integers representing the starting index of the target subblock.
 *
 * @throws None
 */
void Block4D_::CopySubblockFrom(const Block4D_& B, std::array<int64_t,4> sourceOffset, std::array<int64_t,4> targetOffset){
    
    Block4D_ deepCopy = B.clone(); 
    std::array<int64_t,4> length = {std::min(this->data.size(0) - targetOffset[0],B.data.size(0)),
                                    std::min(this->data.size(1) - targetOffset[1],B.data.size(1)),
                                    std::min(this->data.size(2) - targetOffset[2],B.data.size(2)),
                                    std::min(this->data.size(3) - targetOffset[3],B.data.size(3))};
    //std::cout<<"csf length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"original: "<<B.data.to(at::kDouble).sizes()<<std::endl;
       
    this->data.index({at::indexing::Slice(targetOffset[0],targetOffset[0] + length[0]),
                          at::indexing::Slice(targetOffset[1],targetOffset[1] + length[1] ),
                          at::indexing::Slice(targetOffset[2],targetOffset[2] + length[2]),
                          at::indexing::Slice(targetOffset[3],targetOffset[3] + length[3])}) =
                          deepCopy.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });
    //std::cout<<"indexedClone: "<<deepCopy.data.index({at::indexing::Slice(sourceOffset[0],this->data.size(0)),
                                        // at::indexing::Slice(sourceOffset[1],this->data.size(1)),
                                        // at::indexing::Slice(sourceOffset[2],this->data.size(2)),
                                        // at::indexing::Slice(sourceOffset[3],this->data.size(3))
                                        // }).to(at::kDouble).var()<<std::endl;                             
    //std::cout<<"copy: "<<this->data.to(at::kDouble).var()<<std::endl;

    
}
at::Tensor Block4D_::getSgtTransformMatrix(const at::Tensor& cov, bool isHorizontal, at::Tensor& eigVals) const{
    //double eps1 = 1e10; 
    //double eps = 1e-0; 
    //double k = 10;
    //at::Tensor cov1 = cov+at::eye(cov.size(-1),cov.options())*eps1;

    at::Tensor transform = klt(cov,eigVals);

    
    at::Tensor normalizedTransform = transform.clone();
    double eps = 1/sqrt(transform.size(0)) *1e-5;
    //std::cout<<transform.sizes()<<std::endl;
    // auto indexVec = transform[0] < - eps;
    // transform.index({at::indexing::Slice(),indexVec}) = -transform.index({at::indexing::Slice(),indexVec});
    for(int i = 0; i < transform.size(1); i++){
        int bias = 0;
        while(bias < transform.size(0)){

            double reference = transform[bias][i].item<double>();
            if(abs(reference) > eps){
        
                if(reference < 0){
                    transform.index({at::indexing::Slice(),i}) =  -1 * transform.index({at::indexing::Slice(),i});   
                }          
                break;
            }

            bias++;

        }
    }
    return transform;
}
  at::Tensor Block4D_::batchedCovMatrix(bool isHorizontal) const {

    using namespace phoenix::placeholders;

    std::array<int64_t,2> arrayDims, arraycDims;

    if(isHorizontal){
        arrayDims = {1,3};
        arraycDims = {0,2};
    }else{
        arrayDims = {0,2};
        arraycDims = {1,3};
        
    }
    at::IntArrayRef dims = arrayDims;
    at::IntArrayRef cDims = arraycDims;

    // // number of not batched dims
    // const std::int64_t sample_rank = block.sizes().size() - batch_rank;
    // //std::cout<<"sample_rank"<<sample_rank<<std::endl;
    // const auto batch_dims = irange(batch_rank) | to_t<int_arr_t>{}; // 0...batch_rank-1
    // //std::cout<<"batch_dims = "<<batch_dims<<std::endl;
    // // dims to be stacked
    // const auto t_dims = dims | transformed(_1 + batch_rank) | to_t<int_arr_t>{};
    // //std::cout<<"t_dims = "<<t_dims<<std::endl;

    // // find complimentary sample dims
    // auto is_reduced = [&](auto i){ return find(dims, i) == dims.end(); }; // i not in dims
    // const auto c_dims = irange(sample_rank) | filtered(is_reduced) | transformed(_1 + batch_rank) | to_t<int_arr_t>{};
    //std::cout<<"c_dims = "<<c_dims<<std::endl;
    // compute average along unstacked sample dims
    auto centered = this->data - this->data.mean(cDims, true, at::kDouble);
    //std::cout<<"First Mean = "<< this->data.mean(cDims, true, at::kDouble)[0][0][0][0].item<double>()<<std::endl;
    //std::cout<<"average = "<<std::endl<<block.mean(c_dims, true)<<std::endl;
    //std::cout<<"centered = "<<std::endl<<centered<<std::endl;

    // transpose block to ... x (c_dims) x (t_dims)
    int_arr_t transposed_dims;
    push_back(transposed_dims, cDims);
    push_back(transposed_dims, dims);
    //std::cout<<"transposed_dims = "<<std::endl<<transposed_dims<<std::endl;

    // flatten along averaged and stacked dims
    auto flattened = centered.permute(transposed_dims)
      .flatten(0, cDims.size() - 1) // flatten is [a...b], not [a...b)
      .flatten(-dims.size());

    //std::cout<<"flattened = "<<std::endl<<flattened<<std::endl;

    // number of averaged samples
    auto to_size = [&](auto i){ return this->data.size(i); };
    auto averaged_size = boost::accumulate(dims | transformed(to_size), 1, _1 * _2);

    auto cov = at::einsum("...nx,...ny->...xy", {flattened, flattened}) / averaged_size;
    //std::cout<<"average_size = "<<averaged_size<<std::endl;
    //std::cout<<"cov = "<<std::endl<<cov<<std::endl;

    return cov;
  }

 


at::Tensor Block4D_::fetchBlockGradient(int64_t dimension) const{
    std::array<int64_t,4> trueSize;
    for( int i = 0; i < 4; i++){
        // if(lightField->data.size(i) >= lightFieldPosition[i]){
        //     return at::zeros(size, torch::kDouble);
        // }
        trueSize[i] = std::min(lightField->data.size(i) - lightFieldPosition[i], size[i]) ;
    }
    at::Tensor blockGradient = lightField->gradients.index({at::indexing::Slice({lightFieldPosition[0],lightFieldPosition[0]+trueSize[0]}),
                                                            at::indexing::Slice({lightFieldPosition[1],lightFieldPosition[1]+trueSize[1]}),
                                                            at::indexing::Slice({lightFieldPosition[2],lightFieldPosition[2]+trueSize[2]}),
                                                            at::indexing::Slice({lightFieldPosition[3],lightFieldPosition[3]+trueSize[3]}),
                                                           dimension});
                                                           write_tensor(this->data.index({at::indexing::Slice(),1,at::indexing::Slice(),2}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/gradient_"+std::to_string(dimension)+".png");
    // std::cout<<"size: "<<size[0]<<" "<<size[1]<<" "<<size[2]<<" "<<size[3]<<std::endl;
    // std::cout<<"lightFieldPosition: "<<lightFieldPosition[0]<<" "<<lightFieldPosition[1]<<" "<<lightFieldPosition[2]<<" "<<lightFieldPosition[3]<<std::endl;
    // std::cout<<"lightField size: "<<lightField->data.sizes()<<std::endl;
    // std::cout<<"lightField gradients size: "<<lightField->gradients.sizes()<<std::endl;
    // std::cout<<"trueSize: "<<trueSize[0]<<" "<<trueSize[1]<<" "<<trueSize[2]<<" "<<trueSize[3]<<std::endl;
    for( int i = 0; i < 4; i++){
        if(lightField->data.size(i) <= lightFieldPosition[i]){
            //std::cout<<lightField->data.size(i)<<" >= "<<lightFieldPosition[i]<<" at dimension "<<i<<std::endl;
             blockGradient = at::zeros(size, torch::kDouble);
             //std::cout<<"Block gradient is empty because lightFieldPosition is out of bounds!"<<std::endl;
        }
    }
    // std::cout<<"block gradient size: "<<blockGradient.sizes()<<std::endl;
    return blockGradient;
}

double Block4D_::computeGradientSum(int64_t dimension1, int64_t dimension2) const{
    int spatialBorder = 0;
    int angularBorder = 2;
    if(lightFieldPosition[3] == 0 || lightFieldPosition[3] == lightField->data.size(3)-1 || lightFieldPosition[2] == 0 || lightFieldPosition[2] == lightField->data.size(2)-1){
        spatialBorder = 2;
    }
    at::Tensor blockGradient = fetchBlockGradient(dimension1).index({at::indexing::Slice(angularBorder,size[0]-angularBorder),at::indexing::Slice(angularBorder,size[1]-angularBorder),at::indexing::Slice(spatialBorder,size[2]-spatialBorder),at::indexing::Slice(spatialBorder,size[3]-spatialBorder)}) 
                             * fetchBlockGradient(dimension2).index({at::indexing::Slice(angularBorder,size[0]-angularBorder),at::indexing::Slice(angularBorder,size[1]-angularBorder),at::indexing::Slice(spatialBorder,size[2]-spatialBorder),at::indexing::Slice(spatialBorder,size[3]-spatialBorder)});
    return blockGradient.sum().item<double>();
}
double Block4D_::epiStDisparityAvg() const{
    at::Tensor blockGradientS = fetchBlockGradient(1);
    at::Tensor blockGradientU = fetchBlockGradient(3);
    double accDisp = 0;
    for(int t = 0; t<size[0];++t){
        for(int v = 0; v < size[2];++v){
            at::Tensor Js = blockGradientS.index({t,at::indexing::Slice(2,size[1]-2),v,at::indexing::Slice(0,size[3])});
            at::Tensor Ju = blockGradientU.index({t,at::indexing::Slice(2,size[1]-2),v,at::indexing::Slice(0,size[3])});
            // double Jss = Js[4][16].pow(2).item<double>();
            // double Juu = Ju[4][16].pow(2).item<double>();
            // double Jsu = (Js*Ju)[4][16].item<double>();
            double Jss = Js.pow(2).mean().item<double>();
           double Juu = Ju.pow(2).mean().item<double>();
           double Jsu = (Js*Ju).mean().item<double>();

            double disparity = epiStDisparity(Jss,Juu,Jsu);
            //std::cout<<t<<" "<<v<<" "<<disparity<<" "<<Jss<<" "<<Juu<<" "<<Jsu<<std::endl;
            accDisp += disparity;
        }
    }
    return accDisp/(size[0]*size[2]);
}
double Block4D_::epiStDisparity(double jAng, double jSpc, double jSpcAng) const{
    double tmp = (jAng - jSpc);
    at::Tensor st = torch::zeros({2,2},torch::kDouble);
    st[0][0] = jAng;
    st[1][1] = jSpc;
    st[0][1] = jSpcAng;
    st[1][0] = jSpcAng;
    auto [L, Q] = torch::linalg::eigh(st, "U");
    //std::cout<<Q<<std::endl;

    double dx = (tmp + sqrt(tmp*tmp + 4 * jSpcAng*jSpcAng));
    double dy = (2*jSpcAng);
    //std::cout<<jAng<<" "<<jSpc<<" "<<jSpcAng<<std::endl;
    //std::cout<<"dx: "<<dx<<" dy: "<<dy<<std::endl;
    double disparity = dx/dy;

    double r = (tmp*tmp + 4 * jSpcAng*jSpcAng)/((jSpc+jAng)*(jSpc+jAng));
    //std::cout<<"r: "<<r<<std::endl;
    return disparity;
}
std::array<double,2> Block4D_::stAngleSeperable() const{
    double Jss = computeGradientSum(1,1);
    double Jtt = computeGradientSum(0,0);
    double Jvv = computeGradientSum(2,2);
    double Juu = computeGradientSum(3,3);
    double Jsu = computeGradientSum(1,3);
    double Jtv = computeGradientSum(0,2);

    std::array<double,2> angles;
    angles[1] = epiStDisparity(Jtt,Jvv,Jtv);
    angles[0] = epiStDisparity(Jss,Juu,Jsu);
    return angles;
}
at::Tensor Block4D_::structureTensor() const{
    at::Tensor secondMomentum = torch::zeros({4,4},torch::kDouble);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            secondMomentum[i][j] = computeGradientSum(i,j);
        }
    }
    //std::cout<<"L = "<<L<<std::endl;
    //std::cout<<"Q = "<<Q<<std::endl;
    return secondMomentum;
}
std::array<double,2> Block4D_::computeAnglesFromStructureTensor(std::array<double,2> disparityRange) const{


    //std::cout<<"Light Field Position:"<<this->lightFieldPosition<<std::endl;
    //std::cout<<"Size:"<<this->size<<std::endl;

    at::Tensor structureTensor = this->structureTensor();
    auto [L, Q] = torch::linalg::eigh(structureTensor, "U");
    std::array<double,2> angles, reciprocalAngles;
    
    
    angles[1] = - 180/PI * (atan(Q[0][3].item<double>()/Q[2][3].item<double>()));
    angles[0] = - 180/PI * (atan(Q[1][3].item<double>()/Q[3][3].item<double>()));

    reciprocalAngles[1] = - 180/PI * (atan(Q[2][3].item<double>()/Q[0][3].item<double>()));
    reciprocalAngles[0] = - 180/PI * (atan(Q[3][3].item<double>()/Q[1][3].item<double>()));
    double costV  = logDetCost(angles[1], false, disparityRange);
    double costV2 = logDetCost(reciprocalAngles[1], false, disparityRange);


    double costU = logDetCost(angles[0], true, disparityRange);
    double costU2 = logDetCost(reciprocalAngles[0], true, disparityRange);
    
    // if(size[2] == 128){
    //     std::cout<<L<<std::endl;
    //     std::cout<<Q<<std::endl;
    //     std::cout<<angles[0]<< ": "<<costU<<std::endl;
    //     std::cout<<angles[1]<< ": "<<costV<<std::endl;
    //     std::cout<<reciprocalAngles[0]<< ": "<<costU2<<std::endl;
    //     std::cout<<reciprocalAngles[1]<< ": "<<costV2<<std::endl<<std::endl;
    // }
    if (costV2 < costV) {
        angles[1] = reciprocalAngles[1];
    }
    if (costU2 < costU) {
        angles[0] = reciprocalAngles[0];
    }
    // std::array<double,2> angles2 = stAngleSeperable();
    // std::cout<<"SepAngles: "<<angles2[0]<<" "<<angles2[1]<<std::endl;
    // std::cout<<"real disp avg: " <<- 180/PI *atan(epiStDisparityAvg())<<std::endl;
    return angles;
}

void Block4D_::saveBlockGradient(int64_t dimension) const{
    at::Tensor blockGradient = fetchBlockGradient(dimension);
    write_tensor(blockGradient[4][4],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/gradientExample.png");
    std::cout<<"Tensor Written!"<<std::endl;
}
// void Block4D_::computeStructureTensor(at::Tensor& secondMomentum){
//     double Dt = computeAverageMomentum(secondMomentum,0);
//     double Ds = computeAverageMomentum(secondMomentum,1);
//     double Du = computeAverageMomentum(secondMomentum,2);
//     double Dv = computeAverageMomentum(secondMomentum,3);

//     at::Tensor structureTensor = torch::tensor({{Dt*Dt,Dt*Ds,Dt*Du,Dt*Dv},{Ds*Dt,Ds*Ds,Ds*Du,Ds*Dv},{Du*Dt,Du*Ds,Du*Du,Du*Dv},{Dv*Dt,Dv*Ds,Dv*Du,Dv*Dv}},at::kDouble);
//     auto [L, Q] = torch::linalg::eigh(structureTensor, "U");
//     std::cout<<"L = "<<L<<std::endl;
//     std::cout<<"Q = "<<Q<<std::endl;
// }


void Block4D_::sgtTransform(double scale){
    //std::cout<<"SGT transform"<<std::endl;
    
    //ssi.print();


    // write_tensor(this->data.index({at::indexing::Slice(),4,at::indexing::Slice(),16}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/EPI-b4transform.png");
    at::Tensor modelCovMatH = this->calcModelCovMatrix(ssi,true);
    at::Tensor modelCovMatV = this->calcModelCovMatrix(ssi,false);

    
    at::Tensor eigValsH,eigValsV;
    at::Tensor flatBlock = scale * getFlatBlock();
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true,eigValsH);
    at::Tensor sgtMatrixV =   getSgtTransformMatrix(modelCovMatV,false,eigValsV);
    


    
    
    this->eigenValuesH = eigValsH;
    this->eigenValuesV = eigValsV;
    
    //saveBlockGradient(0);
    //double Dt = computeGradientSum(0,0);
    //double Ds = computeGradientSum(1,1);
    //double Dv = computeGradientSum(2,2);
    //double Du = computeGradientSum(3,3);
    //std::cout<<" Ds/Du: "<<Ds/Du<<std::endl;
    //std::cout<<" Dt/Dv: "<<Dt/Dv<<std::endl;
    //std::cout<<" Ds/Dt: "<<Ds/Dt<<std::endl;
    //std::cout<<" Du/Dv: "<<Du/Dv<<std::endl;
    //write_tensor(flatBlock,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/2D-b4transform.png");
    at::Tensor flatTransform = sgt(flatBlock,sgtMatrixH,sgtMatrixV,eigValsH,eigValsV);
    //write_tensor(log(1+(flatTransform * flatTransform)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/2DFull.png");

    //flatTransform.index({at::indexing::Slice({flatTransform.size(0)/8,flatTransform.size(0)}),at::indexing::Slice({flatTransform.size(1)/8,flatTransform.size(1)})}) = 0 ;

    
    
#if FLAT_TRANSFORM == 1
    at::Tensor transform = flatTransform.unsqueeze(0).unsqueeze(0);
#else
    at::Tensor transform = orderCoefficientsByFrequency(flatTransform,sgtMatrixH,sgtMatrixV);       
    int border_size = 4;
    at::Tensor vizTransform = at::ones({flatTransform.size(0)+border_size*8,flatTransform.size(1)+border_size*8},flatTransform.dtype());
    int begin_i = 0;
    for(int i = 0; i < 9; i++){
        int begin_j = 0;
        for(int j = 0; j < 9; j++){
            vizTransform.index({at::indexing::Slice({begin_i,begin_i+32}),at::indexing::Slice({begin_j,begin_j+32})}) = transform[i][j]; 
            begin_j += 32+border_size;
        }
        begin_i += 32+border_size;

    }
    //write_tensor(log2(1+(vizTransform*vizTransform)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/4DTransformViz.png",{0,log2(1+(vizTransform*vizTransform)).max().item<double>()});
#endif
    //at::Tensor transform = flatTransform.unsqueeze(0).unsqueeze(0);
    this->data = transform.round().to(at::kInt).contiguous();
    this->sgtDomain = true;

}

at::Tensor Block4D_::autoCorr(bool isHorizontal){
    at::Tensor flatBlock = getFlatBlock().to(at::kDouble);
    if(isHorizontal){
        flatBlock = flatBlock.t();
    }

    at::Tensor autoCorr = at::zeros({flatBlock.size(0),flatBlock.size(0)},at::kDouble);
    for(int i=0;i<flatBlock.size(0);i++){
        for(int j=0;j<flatBlock.size(0);j++){
            autoCorr[i][j] = (flatBlock[i]*flatBlock[j]).sum();
        }
    }
    
    return autoCorr;

}

void Block4D_::kltTransform(double scale){

    at::Tensor flatBlock = getFlatBlock().to(at::kDouble);
    //at::Tensor currCovH = flatBlock.t().cov();
    //at::Tensor currCovV = flatBlock.cov();
    // at::Tensor currCovH = flatBlock.t().corrcoef();
    // at::Tensor currCovV = flatBlock.corrcoef();
    at::Tensor currCovH = autoCorr(true);
    at::Tensor currCovV = autoCorr(false);
    
    flatBlock*=scale;
    at::Tensor eigValsH, eigValsV;
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(currCovH,true,eigValsH);
    at::Tensor sgtMatrixV =   getSgtTransformMatrix(currCovV,false,eigValsV);
    this->eigenValuesH = eigValsH;
    this->eigenValuesV = eigValsV;
    at::Tensor flatTransform = sgt(flatBlock,sgtMatrixH,sgtMatrixV,eigValsH,eigValsV);
    flatTransform = flatTransform.unsqueeze(0).unsqueeze(0);
    this->data = flatTransform.round().to(at::kInt).contiguous();
    this->sgtDomain = true;
    std::cout<<"KLT Compressed!"<<std::endl;
}



at::Tensor Block4D_::ikltTransformData(double scale, at::Tensor covH, at::Tensor covV) {

    at::Tensor flatBlock = this->data.squeeze().to(at::kDouble)/scale;  
    at::Tensor eigValsH,eigValsV; 
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(covH,true,eigValsH);
    at::Tensor sgtMatrixV = getSgtTransformMatrix(covV,false,eigValsV);   
      
    
    at::Tensor flatTransform = isgt(flatBlock,sgtMatrixH,sgtMatrixV);
    at::Tensor newData = flat24D(flatTransform);
    return flat24D(flatTransform);
}
at::Tensor Block4D_::isgtTransformData(double scale, SgtSideInfo ssi) {
    this->ssi = ssi;
    //SgtSideInfo sortSSI(this->ssi.getAngleV(),this->ssi.getAngleH(),this->ssi.disparityRange);
    //std::cout<<"Inverse Transforming block of size: "<<this->data.sizes()<<std::endl;
    at::Tensor modelCovMatH = this->calcModelCovMatrix(ssi,true);
    at::Tensor modelCovMatV = this->calcModelCovMatrix(ssi,false);  
    at::Tensor eigValsH,eigValsV; 
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true,eigValsH);
    at::Tensor sgtMatrixV = getSgtTransformMatrix(modelCovMatV,false,eigValsV);
    
#if FLAT_TRANSFORM == 1
    at::Tensor flatBlock = this->data.squeeze().to(at::kDouble)/scale;  
#else
    at::Tensor block = this->data.squeeze().to(at::kDouble)/scale;  
    std::cout<<"Transforming block of size: "<<block.sizes()<<std::endl;
    at::Tensor flatBlock = reverseOrderCoefficientsByFrequency(block,sgtMatrixH,sgtMatrixV).squeeze();
#endif     
    at::Tensor flatTransform = isgt(flatBlock,sgtMatrixH,sgtMatrixV);
    std::cout<<"Decoded block of size: "<<flatTransform.sizes()<<std::endl;
    return flat24D(flatTransform);
}

at::Tensor Block4D_::klt(at::Tensor covMat, at::Tensor& eigVals){
    auto [L, Q] = torch::linalg::eigh(covMat, "U");
    eigVals = L.flip({-1});
    return Q.flip({-1}); // flip such that coefficients are in DESCENDING order
}

at::Tensor Block4D_::getOrderH(){
    return orderH;
}
at::Tensor Block4D_::getOrderV(){
    return orderV;
}

at::Tensor Block4D_::sgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV,const at::Tensor& eigValsH,const at::Tensor& eigValsV) {
    at::Tensor block = flatBlock.to(at::kDouble);
    at::Tensor transform = at::mm(at::mm(sgtMatrixV.t(), block), sgtMatrixH);
    return transform;
}

at::Tensor Block4D_::isgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) {
    at::Tensor transform = at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
    return transform;
}




at::Tensor Block4D_::flat24D(const at::Tensor& flatBlock) const{
    at::Tensor block;
    if(includesInvalidCorners){
        block = flat24DValid(flatBlock);
    }else{
        block = flat24DAll(flatBlock);
    }
    return block;
}
at::Tensor Block4D_::flat24DAll(const at::Tensor& flatBlock) const{
    //std::cout<<flatBlock.sizes()<<std::endl;
    std::array<int64_t,4> sizeShifted = {size[1],size[3],size[0],size[2]};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = flatBlock.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});   
    //std::cout<<"size unflattened: "<< unflattened_block.size(0)<<" "<<unflattened_block.size(1)<<" "<<unflattened_block.size(2)<<" "<<unflattened_block.size(3)<<std::endl;

    return  unflattened_block;
}
at::Tensor Block4D_::flat24DValid(const at::Tensor& flatBlock) const {
    at::Tensor padded_block = torch::zeros({data.size(1)*data.size(3),data.size(0)*data.size(2)}).to(at::kDouble);
    auto a = at::meshgrid({validPositions.valid_positions_v, validPositions.valid_positions_h},"ij");
    padded_block = padded_block.index_put({a[0],a[1]},flatBlock);
    std::array<int64_t,4> sizeShifted = {data.size(0),data.size(2),data.size(1),data.size(3)};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = padded_block.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});
    return  unflattened_block;

}
at::Tensor Block4D_::getFlatBlock() {
    at::Tensor flatBlock;
    if(includesInvalidCorners){
        flatBlock = getFlatBlockValid();
    }else{
        //std::cout<<"All!"<<std::endl;
        flatBlock = getFlatBlockAll();
    }
    //std::cout<<"flatBlock size: "<<flatBlock.sizes()<<std::endl;
    return flatBlock;
}


at::Tensor Block4D_::getFlatBlockAll(){
    if(sgtDomain){
        std::cerr<<"getFlatBlockAll: sgtDomain is not supported!"<<std::endl;
        abort();
    }
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    dims = {1,3}; 
    cDims = {0,2};
    //std::cout<<"FlatBlock: "<<dims[0]<<" "<<dims[1]<<" | "<<cDims[0]<<" "<<cDims[1]<<std::endl;
    const std::int64_t sample_rank = 4;
    const auto t_dims = dims;
    std::array<int64_t,4> transposed_dims = {cDims[0],cDims[1],dims[0],dims[1]};
    //std::cout<<"transposed_dims: "<<transposed_dims[0]<<" "<<transposed_dims[1]<<" | "<<transposed_dims[2]<<" "<<transposed_dims[3]<<std::endl;
    auto flatBlock = this->data.permute(transposed_dims).flatten(0,1).flatten(-dims.size());
    return flatBlock;

}

at::Tensor Block4D_::getFlatBlockValid(){
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    at::Tensor flatBlock = getFlatBlockAll();
    flatBlock = flatBlock.index({at::indexing::Slice(), validPositions.valid_positions_h});
    flatBlock = flatBlock.index({validPositions.valid_positions_v,at::indexing::Slice()});
    return flatBlock;
}

void Block4D_::sgtTransform(double scale,std::array<double,2> dispRange){

    this->ssi = SgtSideInfo(*this,dispRange);
    //ssi.print();
    sgtTransform(scale);
}

void Block4D_::ikltTransform(double scale,at::Tensor covH, at::Tensor covV){
    at::Tensor recoveredBlock = ikltTransformData(scale,covH,covV);
    this->data = recoveredBlock;
    //std::cout<<"Final Data = "<<this->data.sizes()<<std::endl;
    this->sgtDomain = false;    
}
void Block4D_::isgtTransform(double scale,SgtSideInfo ssi){
    at::Tensor recoveredBlock = isgtTransformData(scale,ssi);
    this->data = recoveredBlock;
    //std::cout<<"Final Data = "<<this->data.sizes()<<std::endl;
    this->sgtDomain = false;
    
}

double Block4D_::varianceFromCov(const at::Tensor& cov) {
    int64_t k_center = cov.size(0)/2;
    int64_t m_center = cov.size(1)/2; 
    double var = cov[k_center][m_center].item<double>(); 
    return var;
}
at::Tensor Block4D_::corrFun(bool isHorizontal) const{
    using namespace phoenix::placeholders;
    using namespace torch::fft;
    using namespace at::indexing;
    

    std::array<int64_t,2> arrayDims, arraycDims;
    //std::cout<<"block: "<<std::endl<<this->data[0][1]<<std::endl;
    //std::cout << "block type = "<<data.dtype()<<std::endl;
    //std::cout<< "average = " <<data.mean(at::kDouble)<<std::endl;
    //std::cout<< "average Different= " <<data.to(at::kDouble).mean()<<std::endl;
    
    if(isHorizontal){
        arrayDims = {1,3};
        arraycDims = {0,2};
    }else{
        arrayDims = {0,2};
        arraycDims = {1,3};
    }
    at::IntArrayRef dims = arrayDims;
    at::IntArrayRef cDims = arraycDims;

    //std::cout<<"tDims = "<<dims<<std::endl;
    //std::cout<<"cDims = "<<cDims<<std::endl;

    // number of not batched dims
    const std::int64_t sample_rank = this->data.sizes().size();
    // compute average along unstacked sample dims
    //std::cout<<"mean: "<<mean.squeeze()<<std::endl;
    //std::cout<<"var mean: "<<mean.var()<<std::endl;


    auto dft_shape = dims
      | transformed([&](auto i){ return 2*this->data.size(i) - 1;})
      | to_t<int_arr_t>{};


    // dft of each block
    auto block_dft = fftn(this->data, dft_shape,dims);

    // average power spectral density (biased)
    auto block_spd = block_dft.abs().pow(2).mean(cDims, true,at::kDouble); // keepdims here

    // invert cov and unshift along transformed axes
    auto cov = fftshift(at::real(ifftn(block_spd, {}, dims)), dims);
    // squeeze along reduced axes
    for (auto i : cDims | reversed)
      cov.squeeze_(i);

    //std::cout<<"cov raw: "<<cov<<std::endl;
    // compute ramp to compensate correlation average
    auto triang = [&](auto i){
      return at::concat({at::arange(1, this->data.size(i)), at::arange(this->data.size(i), 0, -1)});
    };
    auto ramp = boost::accumulate(dims | transformed(triang), at::ones({1}), bind(at::unsqueeze, _1, -1) * _2);

    return cov / ramp.squeeze(0);
}
at::Tensor Block4D_::covFun(bool isHorizontal) const{
    using namespace phoenix::placeholders;
    using namespace torch::fft;
    using namespace at::indexing;
    

    std::array<int64_t,2> arrayDims, arraycDims;
    //std::cout<<"block: "<<std::endl<<this->data[0][1]<<std::endl;
    //std::cout << "block type = "<<data.dtype()<<std::endl;
    //std::cout<< "average = " <<data.mean(at::kDouble)<<std::endl;
    //std::cout<< "average Different= " <<data.to(at::kDouble).mean()<<std::endl;
    
    if(isHorizontal){
        arrayDims = {1,3};
        arraycDims = {0,2};
    }else{
        arrayDims = {0,2};
        arraycDims = {1,3};
    }
    at::IntArrayRef dims = arrayDims;
    at::IntArrayRef cDims = arraycDims;

    //std::cout<<"tDims = "<<dims<<std::endl;
    //std::cout<<"cDims = "<<cDims<<std::endl;

    // number of not batched dims
    const std::int64_t sample_rank = this->data.sizes().size();
    // compute average along unstacked sample dims
    auto mean = this->data.mean(cDims, /*keepdims=*/true,at::kDouble);
    //std::cout<<"mean: "<<mean.squeeze()<<std::endl;
    //std::cout<<"var mean: "<<mean.var()<<std::endl;


    auto dft_shape = dims
      | transformed([&](auto i){ return 2*this->data.size(i) - 1;})
      | to_t<int_arr_t>{};


    // dft of each block
    auto block_dft = fftn(this->data - mean, dft_shape, dims);

    // average power spectral density (biased)
    auto block_spd = block_dft.abs().pow(2).mean(cDims, true,at::kDouble); // keepdims here

    // invert cov and unshift along transformed axes
    auto cov = fftshift(at::real(ifftn(block_spd, {}, dims)), dims);
    // squeeze along reduced axes
    for (auto i : cDims | reversed)
      cov.squeeze_(i);

    //std::cout<<"cov raw: "<<cov<<std::endl;
    // compute ramp to compensate correlation average
    auto triang = [&](auto i){
      return at::concat({at::arange(1, this->data.size(i)), at::arange(this->data.size(i), 0, -1)});
    };
    auto ramp = boost::accumulate(dims | transformed(triang), at::ones({1}), bind(at::unsqueeze, _1, -1) * _2);

    return cov / ramp.squeeze(0);
}

SgtSideInfo::SgtSideInfo(int RhoSInt,int RhoTInt,int RhoUInt,int RhoVInt,int angleVInt, int angleHInt, std::array<double,2> dispRange){
    this->rhoSInt = RhoSInt;
    this->rhoTInt = RhoTInt;
    this->rhoUInt = RhoUInt;
    this->rhoVInt = RhoVInt;
    this->angleVInt = angleVInt;
    this->angleHInt = angleHInt;
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
}

void SgtSideInfo::print(){
    std::cout<<"Rho S = "<<getRhoS()<<" Rho T = "<<getRhoT()<<" Rho U = "<<getRhoU()<<" Rho V = "<<getRhoV()<<" Angle V = "<<getAngleV()<< " Angle H = "<<getAngleH()<<std::endl;
    std::cout<<"Rho S Code = "<<getRhoSCode()<<" Rho T Code = "<<getRhoTCode()<<" Rho U Code = "<<getRhoUCode()<<" Rho V Code = "<<getRhoVCode()<<" Angle Code H: "<<angleHInt<<" Angle Code V: "<<angleVInt<<std::endl;
}
SgtSideInfo::SgtSideInfo(std::array<double,2> dispRange){
    setSpatialRhos();
    setAngularRhos();
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
    setAngleH(0);
    setAngleV(0);
}


SgtSideInfo::SgtSideInfo(const Block4D_& block,std::array<double,2> dispRange){
    //std::cout<<"We're getting our Side Info Ready!"<<std::endl;
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
    //std::cout<<"Angle Range:" <<angleRange[0]<<" "<<angleRange[1]<<std::endl;

    //this->angleRange = {180/acos(-1) * atan(dispRange[0]),180/acos(-1) * atan(dispRange[1])};
    estimateDisparity(block);
    
    //std::cout<<"I have estimated the angles to be "<< this->getAngleH()<<" and "<<this->getAngleV()<<std::endl;
    
    //estimateAngleFromMonotony(block);

    //std::cout<<"Angle Range:" <<angleRange[0]<<" "<<angleRange[1]<<std::endl;
    //print();
    //this->setAngleH(-47);
    //this->setAngleV(-47);
    //std::cout<<"Disparity = "<<this->getDisparity()<<std::endl;
    //print();

    estimateRhos(block,VARIANCE_THRESHOLD);
    //std::cout<<"Previous:"<<std::endl;
    //print();
    //estimateRhosFromMonotony(block);
    //std::cout<<"New:"<<std::endl;
    //print();

    //estimateRhoAngle(block);
}


int SgtSideInfo::codeRho(double rho) const{
    // std::cout<<"Code: 0.99  "<<0.99 * getRhoCodeScale() + getRhoCodeBias()<<std::endl;
    // std::cout<<"Code: 0.999  "<<0.999 * getRhoCodeScale() + getRhoCodeBias()<<std::endl;
    // std::cout<<"Code: 0.9999  "<<0.9999 * getRhoCodeScale() + getRhoCodeBias()<<std::endl;
    // std::cout<<"Code: 0.99999  "<<0.99999 * getRhoCodeScale() + getRhoCodeBias()<<std::endl;
    int code = int(rho * getRhoCodeScale() + getRhoCodeBias());
    //std::cout<<"MIN"<<MIN_RHO * getRhoCodeScale() + getRhoCodeBias()<<" "<<DecodeRho(MIN_RHO * getRhoCodeScale() + getRhoCodeBias())<<std::endl;
    //std::cout<<"MAX"<<MAX_RHO * getRhoCodeScale() + getRhoCodeBias()<<" "<<DecodeRho(MAX_RHO * getRhoCodeScale() + getRhoCodeBias())<<std::endl;
    //std::cout<<"codeRho() Curr Code: "<<rho<<"  "<<code<<std::endl;
    //std::cout<<"Code = "<<code<<" TEST: "<<rho<<"  =  "<<DecodeRho(code)<<std::endl;
    return code;
}
int SgtSideInfo::codeAngle(double theta) const{

    //std::cout<<"Angle Range codeAngle: "<<this-> angleRange[0]<<" "<<this->angleRange[1]<<std::endl;
    //std::cout<<"theta: "<<theta<<" CodeScale: "<<getAngleCodeScale()<< " CodeBias: "<<getAngleCodeBias()<<" Double Result: "<<theta * getAngleCodeScale() + getAngleCodeBias()<<std::endl;
    if(theta > angleRange[1] ) theta = angleRange[1];
    if(theta < angleRange[0] ) theta = angleRange[0];
    return round(theta * getAngleCodeScale() + getAngleCodeBias());
}

double SgtSideInfo::DecodeRho(int rhoCode) const{
    //std::cout<<rhoCode<<":"<<(rhoCode - getRhoCodeBias()) / getRhoCodeScale()<<std::endl;
    return (rhoCode - getRhoCodeBias()) / getRhoCodeScale();
}
double SgtSideInfo::DecodeAngle(int dCode) const{
    return (dCode - getAngleCodeBias()) / getAngleCodeScale();
}

void SgtSideInfo::setRhoSCode(int code){
    this->rhoSInt = code;
}
void SgtSideInfo::setRhoTCode(int code){
    this->rhoTInt = code;
}
void SgtSideInfo::setRhoUCode(int code){
    this->rhoUInt = code;
}
void SgtSideInfo::setRhoVCode(int code){
    this->rhoVInt = code;
}
void SgtSideInfo::setAngleVCode(int code){
    this->angleVInt = code;
}
void SgtSideInfo::setAngleHCode(int code){
    this->angleHInt = code;
}
int SgtSideInfo::getRhoSCode() const{
    return this->rhoSInt;
}
int SgtSideInfo::getRhoTCode() const{
    return this->rhoTInt;
}
int SgtSideInfo::getRhoUCode() const{
    return this->rhoUInt; 
}
int SgtSideInfo::getRhoVCode() const{
    return this->rhoVInt;
}
int SgtSideInfo::getAngleVCode() const{
    return this->angleVInt;
}
int SgtSideInfo::getAngleHCode() const{
    return this->angleHInt;
}
double SgtSideInfo::getRhoCodeBias() const{
    
    return -MIN_RHO/PRECISION_RHO;
    
}
double SgtSideInfo::getRhoCodeScale() const{
    return 1/PRECISION_RHO;
}



nlohmann::json SgtSideInfo::toJson() const {
    nlohmann::json j;
    j["angleH"] = getAngleH();
    j["angleV"] = getAngleV();
    j["rhoS"] = getRhoS();
    j["rhoU"] = getRhoU();
    j["rhoT"] = getRhoT();
    j["rhoV"] = getRhoV();
    return j;
}

SgtSideInfo SgtSideInfo::fromJson(const nlohmann::json& j) {
    SgtSideInfo sgtSideInfo({-100,100});
    sgtSideInfo.setAngleH(j["angleH"].get<double>());
    sgtSideInfo.setAngleV(j["angleV"].get<double>());
    sgtSideInfo.setRhoS(j["rhoS"].get<double>());
    sgtSideInfo.setRhoU(j["rhoU"].get<double>());
    sgtSideInfo.setRhoT(j["rhoT"].get<double>());
    sgtSideInfo.setRhoV(j["rhoV"].get<double>());
    return sgtSideInfo;
}

double SgtSideInfo::getAngleCodeBias() const{   
    //std::cout<<"Angle Range Bias: "<<this-> angleRange[0]<<" "<<this->angleRange[1]<<std::endl;
    //std::cout<<-angleRange[0]<<" "<<PRECISION_ANGLE<<std::endl;
    return -angleRange[0]/PRECISION_ANGLE;
}
double SgtSideInfo::getAngleCodeScale() const{
    return 1/PRECISION_ANGLE;
}
double SgtSideInfo::getRhoS() const{
    double rhoS = DecodeRho(rhoSInt);
    return rhoS;
}
double SgtSideInfo::getRhoU() const{
    double rhoU = DecodeRho(rhoUInt);
    return rhoU;
}
double SgtSideInfo::getRhoT() const{
    double rhoT = DecodeRho(rhoTInt);
    return rhoT;
}
double SgtSideInfo::getRhoV() const{
    double rhoV = DecodeRho(rhoVInt);
    return rhoV;
}
double SgtSideInfo::getDisparityV() const{
     double disparity = tan(acos(-1)/180 * DecodeAngle(angleVInt));
    return disparity;
}
double SgtSideInfo::getDisparityH() const{
     double disparity = tan(acos(-1)/180 * DecodeAngle(angleHInt));
    return disparity;
}
double SgtSideInfo::getAngleV() const {
    double angle = DecodeAngle(angleVInt);
    return angle;
}
double SgtSideInfo::getAngleH() const{
    double angle = DecodeAngle(angleHInt);
    return angle;
}
int SgtSideInfo::getRhoPrecision() const{
    int rhoPrecision = std::ceil(log2(codeRho(MAX_RHO)+1));
    return rhoPrecision;
}
int SgtSideInfo::getAnglePrecision() const{
    return std::ceil(log2(codeAngle(this->angleRange[1])+1));
}
void SgtSideInfo::setRhoS(double rhoS){
    //std::cout<<"MAX_RHO = "<<MAX_RHO<<" rhoS = "<<rhoS<<std::endl;
    if(!isfinite(rhoS)) rhoS = FIXED_ANGULAR_RHO;

    rhoS = std::clamp(rhoS,MIN_RHO,MAX_RHO);
    this->rhoSInt = codeRho(rhoS);
}
void SgtSideInfo::setRhoU(double rhoU){
    if(!isfinite(rhoU)) rhoU = FIXED_SPATIAL_RHO;

    rhoU = std::clamp(rhoU,MIN_RHO,MAX_RHO);
    this->rhoUInt = codeRho(rhoU);

}
void SgtSideInfo::setRhoT(double rhoT){
    if(!isfinite(rhoT)) rhoT = FIXED_ANGULAR_RHO;

    rhoT = std::clamp(rhoT,MIN_RHO,MAX_RHO);
    this->rhoTInt = codeRho(rhoT);

}
void SgtSideInfo::setRhoV(double rhoV){
    if(!isfinite(rhoV)) rhoV = FIXED_SPATIAL_RHO;
    rhoV = std::clamp(rhoV,MIN_RHO,MAX_RHO);
    this->rhoVInt = codeRho(rhoV);
}
void SgtSideInfo::setAngleV(double theta){
    if(!isfinite(theta)) theta = 0;
    this->angleVInt =codeAngle(theta);
}
void SgtSideInfo::setAngleH(double theta){
    if(!isfinite(theta)) theta = 0;
    this->angleHInt =codeAngle(theta);
    if(this->angleHInt < 0){

        std::cout<<"ERROR: Negative Angle H Code: "<<this->angleHInt<<std::endl;
        std::cout<<"Angle H: "<<theta<<std::endl;
        std::cout<<"Angle Range: "<<this->angleRange[0]<<" "<<this->angleRange[1]<<std::endl;
        abort();
    }
    
}
void SgtSideInfo::setAngleVFromDisparity(double d){
    this->angleVInt =codeAngle(atan(d) * 180 / acos(-1));
}
void SgtSideInfo::setAngleHFromDisparity(double d){
    this->angleHInt =codeAngle(atan(d) * 180 / acos(-1));
}


void SgtSideInfo::setSpatialRhos(const at::Tensor& covFunH, const at::Tensor& covFunV){
    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2; 
    double varH = covFunH[k_center][m_center].item<double>(); 
    setRhoU((covFunH[k_center][m_center+1]/varH).item<double>());
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2; 
    double varV= covFunV[l_center][n_center].item<double>(); 
    setRhoV( (covFunV[l_center][n_center+1]/varV).item<double>());
}

void SgtSideInfo::setSpatialRhos(const double rhoU, const double rhoV){
    setRhoU(rhoU);
    setRhoV(rhoV);
}
void SgtSideInfo::setAngularRhos(const double rhoS, const double rhoT){
    setRhoS(rhoS);
    setRhoT(rhoT);
}
at::Tensor Block4D_::calcModelCovFun(SgtSideInfo ssi,bool isHorizontal) const{
    double rhoSpt, rhoAng,disparity;
    int64_t sizeSpt,sizeAng;
    //ssi.print();
    if(isHorizontal){
        rhoSpt = ssi.getRhoU();
        sizeSpt = this->size[3];
        rhoAng = ssi.getRhoS();
        sizeAng = this->size[1];
        disparity = ssi.getDisparityH();
    }else{
        rhoSpt = ssi.getRhoV();
        sizeSpt = this->size[2];
        rhoAng = ssi.getRhoT(); 
        sizeAng = this->size[0];
        disparity = ssi.getDisparityV();
    }

    //std::cout<<"cov size: "<<sizeAng<<" "<<sizeSpt<<" "<<rhoSpt<<" "<<rhoAng<<std::endl;
    auto [s,u] = make_function_grid({sizeAng, sizeSpt});
        //std::cout<<"func gri dome: "<<sizeAng<<" "<<sizeSpt<<std::endl;

    
    //std::cout<<s<<std::endl;
    //std::cout<<u<<std::endl;
    at::Tensor modelCovFun = torch::pow(rhoSpt,(at::abs(u - (disparity * s)))) * torch::pow(rhoAng,s.abs());
    //std::cout<<"Hello People!"<<std::endl;
    return modelCovFun;
}
at::Tensor Block4D_::calcModelCovMatrix(SgtSideInfo ssi,bool isHorizontal) const{
    at::Tensor modelCovFun = calcModelCovFun(ssi, isHorizontal);
    at::Tensor modelCovMat = covFun2Mat(modelCovFun,isHorizontal);
    return modelCovMat;
}

at::Tensor Block4D_::normalizeCov(at::Tensor cov){
    //Get the index of the central value (variance)
    int64_t s_center = cov.sizes()[0]/2;
     int64_t u_center = cov.sizes()[1]/2;
     //normalize so variance = 1
     cov = cov / cov[s_center][u_center];
    return cov;
}

double Block4D_::logDetCost(double angle, bool isHorizontal, std::array<double,2> disparityRange) const{
    SgtSideInfo temp(disparityRange);
    at::Tensor covFunH = Block4D_::normalizeCov(this->covFun(true));
    at::Tensor covFunV = Block4D_::normalizeCov(this->covFun(false));

    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2;     
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2;     
    temp.setRhoU(covFunH[k_center][m_center+1].item<double>());
    temp.setRhoV(covFunV[l_center][n_center+1].item<double>());
    temp.setRhoS(SgtSideInfo::MAX_RHO);
    temp.setRhoT(SgtSideInfo::MAX_RHO);
    if(isHorizontal){
        temp.setAngleH(angle);
    }else{
        temp.setAngleV(angle);
    }
    at::Tensor modelCovMat = this->calcModelCovMatrix(temp,isHorizontal);
    at::Tensor iSqrtCovMat = this->iSqrtCovMat(isHorizontal);
    double genDiv = temp.genDivergence(modelCovMat,iSqrtCovMat);
    return genDiv;
}
std::array<double,2> Block4D_::logDetAngleEstimation(double precision, std::array<double,2> dispRange) const{
    at::Tensor covFunH = Block4D_::normalizeCov(this->covFun(true));
    at::Tensor covFunV = Block4D_::normalizeCov(this->covFun(false));

    SgtSideInfo temp(dispRange);
    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2;     
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2;     
    temp.setRhoU(covFunH[k_center][m_center+1].item<double>());
    temp.setRhoV(covFunV[l_center][n_center+1].item<double>());
    temp.setRhoS(SgtSideInfo::MAX_RHO);
    temp.setRhoT(SgtSideInfo::MAX_RHO);
    at::Tensor iSqrtCovMatH = this->iSqrtCovMat(true);
    //std::cout<<"iSqrt: "<<std::endl<<iSqrtCovMatH[0]<<std::endl<<std::endl<<std::endl;
    at::Tensor iSqrtCovMatV = this->iSqrtCovMat(false);
    double minV = std::numeric_limits<double>::max();
    double minH = std::numeric_limits<double>::max();
    double angleV = 0;
    double angleH = 0;
    for(double theta = temp.angleRange[0];theta<=temp.angleRange[1];theta+=precision){
        temp.setAngleH(theta);
        temp.setAngleV(theta);
        at::Tensor modelCovMatH = this->calcModelCovMatrix(temp,true);
        at::Tensor modelCovMatV = this->calcModelCovMatrix(temp,false);
        double genDivH = temp.genDivergence(modelCovMatH,iSqrtCovMatH);
        double genDivV = temp.genDivergence(modelCovMatV,iSqrtCovMatV);
        if(genDivH < minH){
            minH = genDivH;
            angleH = theta;
        }
        if(genDivV < minV){
            minV = genDivV;
            angleV = theta;
        }             
    }
    return {angleH,angleV};


}
void SgtSideInfo::estimateDisparity(const Block4D_& block){
    //std::cout<<"We're estimating disparity... I think I ruined something!"<<std::endl;
    at::Tensor covFunH = Block4D_::normalizeCov(block.covFun(true));
    at::Tensor covFunV = Block4D_::normalizeCov(block.covFun(false));
    //std::cout<<"covFunH:"<<covFunH.sizes()<<std::endl;
    //std::cout<<"covFunV:"<<covFunV.sizes()<<std::endl;
    // at::Tensor covFunH = Block4D_::normalizeCov(block.cov(true));
    // at::Tensor covFunV = Block4D_::normalizeCov(block.covFun(false));

    //saveTensorAsMatlabScript(covFunH,"covFunH");
    //saveTensorAsMatlabScript(covFunV,"covFunV");
    std::vector<double> genDivHVec;
    std::vector<double> genDivVVec;
    std::vector<double> thetaVec;

    SgtSideInfo temp(this->disparityRange);
    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2;     
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2;     
    temp.setRhoU(covFunH[k_center][m_center+1].item<double>());
    temp.setRhoV(covFunV[l_center][n_center+1].item<double>());
    temp.setAngularRhos();
    

    at::Tensor iSqrtCovMatH = block.iSqrtCovMat(true);
    at::Tensor iSqrtCovMatV = block.iSqrtCovMat(false);
    double min = std::numeric_limits<double>::max();
    double minV = std::numeric_limits<double>::max();
    double minH = std::numeric_limits<double>::max();
    double angleV = 0;
    double angleH = 0;
    
    for(double theta = this->angleRange[0];theta<=this->angleRange[1];theta+=PRECISION_ANGLE){
        temp.setAngleH(theta);
        temp.setAngleV(theta);
        at::Tensor modelCovMatH = block.calcModelCovMatrix(temp,true);
        at::Tensor modelCovMatV = block.calcModelCovMatrix(temp,false);
        double genDivH = genDivergence(modelCovMatH,iSqrtCovMatH);
        double genDivV = genDivergence(modelCovMatV,iSqrtCovMatV);

        double genDiv = genDivH + genDivV;
        genDivHVec.push_back(genDivH);
        genDivVVec.push_back(genDivV);
        thetaVec.push_back(theta);

        if (genDivH < minH){
            minH = genDivH;
            angleH = theta;
            setAngleH(theta);
        }
        if (genDivV< minV){
            minV = genDivV;
            angleV = theta;
            setAngleV(theta);
        }
    }
}


double SgtSideInfo::genDivergence(const at::Tensor& p, const at::Tensor& qRsqrt){
    double eps = 1e-2; 
    at::Tensor p_regularized = p+at::eye(p.size(-1),p.options())*eps;


    auto prod = at::einsum("...xm,...xy,...yn->...mn", {qRsqrt, p_regularized, qRsqrt});
    auto eigvals = torch::linalg::eigvalsh(prod, "U");

   return (eigvals.mean(-1).log() - eigvals.log().mean(-1)).item<double>();
}


at::Tensor Block4D_::iSqrtCovMat(at::Tensor covMat, bool isHorizontal ) const{
    auto [L, Q] = torch::linalg::eigh(covMat, "U");
    
    auto&& [Lmax, Lloc] = L.max(-1, /*keepdims*/true);
    auto Lrel = L / Lmax; // 'relative' eigenvalues
    auto&& [Lmin, min_where]  = L.view({-1, L.size(-1)}).min(0);
    auto Lkeep = Lmin > 1e-4;
    auto Lselect = L.index({"...", Lkeep});
    auto Qselect = Q.index({"...", Lkeep});
    auto Lrsqrt = Lselect.rsqrt_();

    auto result = at::einsum("...x,...x->...x", {Qselect, Lrsqrt});
    return result;
}
at::Tensor Block4D_::iSqrtCovMat(bool isHorizontal ) const{
    at::Tensor covFun = normalizeCov(this->covFun(isHorizontal));
    at::Tensor covMat = covFun2Mat(covFun,isHorizontal);
    return iSqrtCovMat(covMat,isHorizontal );
}

at::Tensor Block4D_::covFun2Mat(const at::Tensor& covFun,bool isHorizontal) const{
    at::Tensor covMat;
    if(includesInvalidCorners){
        std::cerr<<"Invalid Corners?????????"<<std::endl;
        covMat = covFun2MatValid(covFun,isHorizontal);
    }else{
        covMat = covFun2MatAll(covFun);
    }
    return covMat;
}
at::Tensor Block4D_::covFun2MatAll(const at::Tensor& covFun) const{

    std::int64_t sample_rank = covFun.sizes().size();

    // map from 0...size[i]
    tensor_arr_t aranges;
    for (auto i : irange(sample_rank))
      aranges.push_back(at::arange((covFun.size(i) + 1)/2));

    // stacked indices of proper shape
    auto cart = at::cartesian_prod(aranges).t();

    // initialize indexing vector with "..."
    std::vector<at::indexing::TensorIndex> indexing {"..."};
    for(auto i : irange(sample_rank)) {
      auto offset = (covFun.size(i) + 1) / 2 - 1;
      auto dim_index = cart[i].unsqueeze(-1) - cart[i] + offset;
      indexing.push_back(dim_index);
    }
    at::Tensor cov_mat = covFun.index(indexing);
    return cov_mat;
}
at::Tensor Block4D_::covFun2MatValid(const at::Tensor& covFun,bool isHorizontal) const {
    at::Tensor validIndexes;
    if(isHorizontal){
        validIndexes = this->validPositions.valid_positions_h;
    }else{
        validIndexes = this->validPositions.valid_positions_v;
    }
    at::Tensor covMat = covFun2MatAll(covFun);
    covMat = covMat.index({at::indexing::Slice(), validIndexes});
    covMat = covMat.index({validIndexes,at::indexing::Slice()});
    return covMat;
  }
  at::Tensor batched_cov_fun_to_mat(const at::Tensor& cov_fun, std::int64_t batch_rank) {

    std::int64_t sample_rank = cov_fun.sizes().size() - batch_rank;

    // map from 0...size[i]
    tensor_arr_t aranges;
    for (auto i : irange(sample_rank))
      aranges.push_back(at::arange((cov_fun.size(batch_rank + i) + 1)/2));

    // stacked indices of proper shape
    auto cart = at::cartesian_prod(aranges).t();

    // initialize indexing vector with "..."
    std::vector<at::indexing::TensorIndex> indexing {"..."};
    for(auto i : irange(sample_rank)) {
      auto offset = (cov_fun.size(batch_rank + i) + 1) / 2 - 1;
      auto dim_index = cart[i].unsqueeze(-1) - cart[i] + offset;
      indexing.push_back(dim_index);
    }

    return cov_fun.index(indexing);
  } 

void SgtSideInfo::estimateRhoAngle(const Block4D_& block){
    at::Tensor iSqrtCovMatH = block.iSqrtCovMat(true);
    at::Tensor iSqrtCovMatV = block.iSqrtCovMat(false);
    at::Tensor covFunH = block.covFun(true);
    at::Tensor covFunV = block.covFun(false);
    double varH = Block4D_::varianceFromCov(covFunH);
    double varV = Block4D_::varianceFromCov(covFunV);
    double min = std::numeric_limits<double>::max();
    std::vector<double> spatialRhos = {0.8,0.9,0.99};
    std::vector<double> angularRhos = {0.8,0.9,0.99,0.999,0.9999};
    SgtSideInfo temp = *this;


    if (varH < 0){
        this->setRhoS(FIXED_ANGULAR_RHO);
        this->setRhoU(FIXED_SPATIAL_RHO);
    }else{

        for(double rho_u_order = 0;rho_u_order < spatialRhos.size();rho_u_order++){
            for(double rho_s_order = 0;rho_s_order < angularRhos.size();rho_s_order++){
                double rhoS = angularRhos[rho_s_order];
                double rhoSpace = spatialRhos[rho_u_order];
                
                temp.setRhoU(rhoSpace);
                temp.setRhoS(rhoS);
                at::Tensor modelCovMatH = block.calcModelCovMatrix(temp,true);

                double genDiv = genDivergence(modelCovMatH,iSqrtCovMatH);
                //std::cout<<"("<<rhoS<<", "<<rhoSpace<<"): "<<genDiv<<std::endl;

                if (genDiv < min){
                    min = genDiv;
                    
                    this->setRhoS(rhoS);
                    this->setRhoU(rhoSpace);
                }
            }
        }
    }
     if (varV < 0){
        this->setRhoT(FIXED_ANGULAR_RHO);
        this->setRhoV(FIXED_SPATIAL_RHO);
    }else{
    
        min = std::numeric_limits<double>::max();

        for(double rho_v_order = 0;rho_v_order < spatialRhos.size();rho_v_order++){
            for(double rho_t_order = 0;rho_t_order < angularRhos.size();rho_t_order++){
                double rhoT = angularRhos[rho_t_order];
                double rhoSpace = spatialRhos[rho_v_order];
                temp.setRhoT(rhoT);
                temp.setRhoV(rhoSpace);
                at::Tensor modelCovMatV = block.calcModelCovMatrix(temp,false);
                double genDiv = genDivergence(modelCovMatV,iSqrtCovMatV);
                //std::cout<<"("<<rhoT<<", "<<rhoSpace<<"): "<<genDiv<<std::endl;

                if (genDiv < min){
                    min = genDiv;
                    this->setRhoT(rhoT);
                    this->setRhoV(rhoSpace);
                }
            }
        }  
    }  
} 

std::array<double,2> SgtSideInfo::angleRangeFromDispRange(std::array<double,2> dispRange){
    double minAngle = floor(180/acos(-1) * atan(dispRange[0]));
    double maxAngle = 180/acos(-1) * atan(dispRange[1]);
    double N = ceil((maxAngle - minAngle)/PRECISION_ANGLE);
    maxAngle = minAngle + N * PRECISION_ANGLE;
    return {minAngle,maxAngle};
}
SgtSideInfo::SgtSideInfo(double angleV, double angleH, std::array<double,2> dispRange){
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
    this->setAngleH(angleH);
    this->setAngleV(angleV);

    setSpatialRhos();
    setAngularRhos();
}

void SgtSideInfo::estimateRhos(const Block4D_& block, double varianceThreshold){
    at::Tensor covFunH = block.covFun(true);
    at::Tensor covFunV = block.covFun(false);
    at::Tensor corrFunH = block.corrFun(true);
    at::Tensor corrFunV = block.corrFun(false);
    bool calculateRhos = varianceThreshold < 0;
    //std::cout<<covFunH.sizes()<<" "<<corrFunH.sizes()<<std::endl;
    double varH = Block4D_::varianceFromCov(covFunH);
    double varV = Block4D_::varianceFromCov(covFunV);

    //std::cout<<"VarH = "<<varH<<" VarV = "<<varV<<std::endl;
    //if(varH < varianceThreshold){
    
    //std::cout<<"FIXED_ANGULAR_RHO : "<<FIXED_ANGULAR_RHO<<" FIXED_SPATIAL_RHO : "<<FIXED_SPATIAL_RHO<<std::endl;
    //std::cout<<"vars: "<<varH<<" > "<<varianceThreshold<<std::endl;
#if ADAPTIVE_RHO_CALC == 1
    if(varH < varianceThreshold && !calculateRhos){
#else
    if(true){
#endif
        setRhoS(FIXED_ANGULAR_RHO);
        setRhoU(FIXED_SPATIAL_RHO);
    }else{
        estimateRhosLS(covFunH,true);

    }
#if ADAPTIVE_RHO_CALC == 1
    if(varV < varianceThreshold && !calculateRhos){
#else
    if(true){
#endif
        setRhoT(FIXED_ANGULAR_RHO);
        setRhoV(FIXED_SPATIAL_RHO);
    }else{
        estimateRhosLS(covFunV,false);
    }
    //std::cout<<this->getRhoS()<<" "<<this->getRhoU()<<std::endl;
    //std::cout<<"Finished Estimating Rhos"<<std::endl;
}

void SgtSideInfo::estimateRhosLS(const at::Tensor& cov, bool isHorizontal){
    int64_t k_center = cov.size(0)/2;
    int64_t m_center = cov.size(1)/2; 
    int64_t m_samples = std::min(4,(int)m_center);
    int64_t k_samples = std::min(4,(int)m_center);
    double disparity;
    if (isHorizontal){
        disparity = this->getDisparityH();
    }else{
        disparity = this->getDisparityV();
    }
    at::Tensor normalized_cov = cov.clone();
    normalized_cov /= cov[k_center][m_center]; 
    //std::cout<<"cov size: "<<normalized_cov.sizes()<<" m_center = "<<m_center<<std::endl;
    at::Tensor working_cov = at::clamp(normalized_cov.index({at::indexing::Slice(k_center - k_samples,k_center + k_samples+1),
                                                            at::indexing::Slice(m_center-m_samples,m_center+m_samples+1)}),0.01,2);
    //std::cout<<working_cov.sizes()<<std::endl;
    at::Tensor m = torch::arange(-m_samples,m_samples +1);
    at::Tensor k = torch::arange(- k_samples,k_samples+1);
    auto meshes = at::meshgrid({k.to(at::kDouble),m.to(at::kDouble)}, "ij");

    double best_alpha;
    auto b = log(working_cov).flatten().unsqueeze(1);
    at::Tensor temp = abs(meshes[1] - meshes[0]*disparity );
    auto A = torch::cat({abs(meshes[0].flatten().unsqueeze(1)),temp.flatten().unsqueeze(1)},1); 


    //auto A_t = A.t();
    // //std::cout<<A.sizes()<<" "<<b.sizes()<<std::endl;
    //auto beta = matmul(matmul(inverse(matmul(A_t,A)),A_t),b);
    //std::cout<<"New Stuff Starts Here"<<std::endl;
    auto beta = constrainedLeastSquares(A,b);
    //std::cout<<beta.sizes()<<std::endl;

    //auto r = ((matmul(A,beta) - b)*(matmul(A,beta) - b));
    //double current_cost = (r.sum()/(r.size(0)-1)).item<double>();
    at::Tensor best_beta = beta.exp();
    //std::cout<<"RhoS = "<<best_beta<<std::endl;
    if(isHorizontal){
        setRhoS(best_beta[0].item<double>());
        setRhoU(best_beta[1].item<double>());
    }else{
        setRhoT(best_beta[0].item<double>());
        setRhoV(best_beta[1].item<double>());
    }  
}




void Block4D_::Ones(){
    this->data = torch::ones(this->data.sizes(),this->data.dtype());
}
void Block4D_::Zeros(){
    this->data = torch::zeros(this->data.sizes(),this->data.dtype());
}
void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s){
    if(shift > 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_left_shift(shift);
    }
    if(shift < 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_right_shift(-shift);
    }
}

Block4D_ Block4D_::operator + (const Block4D_ &B) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data + B.data;
    return newBlock;
}
Block4D_ Block4D_::operator * (const Block4D_ &B) const{
    Block4D_ newBlock = this ->clone();
    newBlock.data = this->data * B.data;
    return newBlock;
}
Block4D_ Block4D_::operator - (const Block4D_ &B) const {
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data - B.data;
    return newBlock;
}
Block4D_ Block4D_::operator + (const at::Tensor &B) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data + B;
    return newBlock;
}
Block4D_ Block4D_::operator * (const at::Tensor &B) const{
    Block4D_ newBlock = this ->clone();
    newBlock.data = this->data * B;
    return newBlock;
}
Block4D_ Block4D_::operator - (const at::Tensor &B) const {
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data - B;
    return newBlock;
}

Block4D_ operator * (const int a,const Block4D_ &B ){
    return B * a;
}

Block4D_ operator + (const int a,const Block4D_ &B ){
    return B + a;
}

Block4D_ operator - (const int a,const Block4D_ &B ){
    return -1 * (B - a);
}

Block4D_ Block4D_::operator / (const int a) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data / a;
    return newBlock;
}
Block4D_ Block4D_::operator / (const double a) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data / a;
    return newBlock;
}
Block4D_ Block4D_::operator + (const int a) const {
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data + a;
    return newBlock;
}
Block4D_ Block4D_::operator - (const int a) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data - a;
    return newBlock;

}
Block4D_ Block4D_::operator * (const int a) const{
    Block4D_ newBlock = this->clone();
    newBlock.data = this->data *a;
    return newBlock;
}
void Block4D_::operator += (const Block4D_ &B){
    this->data = this->data + B.data;
}
void Block4D_::operator -= (const Block4D_ &B){
    this->data = this->data - B.data;
}
void Block4D_::operator *= (const Block4D_ &B){
    this->data = this->data * B.data;
}
void Block4D_::operator = (const Block4D_ &B){
    
    this->data = B.data;
    this->size = B.size;
    this->transformSize = B.transformSize;
    this->sgtDomain = B.sgtDomain;
    this->ssi = B.ssi;
    this->includesInvalidCorners = B.includesInvalidCorners;
    this->orderH = B.orderH;
    this->orderV = B.orderV;
    this->lightFieldPosition = B.lightFieldPosition;
    this->lightField = B.lightField;
}

Block4D_ Block4D_::clone() const{

    Block4D_ newBlock(this->size,this->lightFieldPosition,this->lightField);
    newBlock.data = this->data.clone();    
    newBlock.size = this->size;
    newBlock.transformSize = this->transformSize;
    newBlock.sgtDomain = this->sgtDomain;
    newBlock.ssi = this->ssi;
    newBlock.includesInvalidCorners = this->includesInvalidCorners;
    newBlock.orderH = this->orderH;
    newBlock.orderV = this->orderV;
    newBlock.lightFieldPosition = this->lightFieldPosition;
    newBlock.lightField = this->lightField;

    return newBlock;
}

/********Static Functions******/

at::Tensor Block4D_::get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal){
  
  
  int64_t view_coordinate = 0;
  int64_t spatial_coordinate = 2;
  if(is_horizontal){
    view_coordinate = 1;
    spatial_coordinate = 3;
  }
  int lf_extra_size = (int)(abs(round(adjustment_d*(lf_shape[view_coordinate]-1))));
  double true_alpha;
  if(adjustment_d == 0){
    true_alpha = 0;
  }else{
    true_alpha = adjustment_d/abs(adjustment_d) * (double)lf_extra_size/((double)lf_shape[view_coordinate]-1);
  }
  //cout<<"      True alpha = "<<true_alpha<<endl;
  std::vector<at::Tensor> padding_coordinates;
  //cout<<"      Block Start: "<<block_start[spatial_coordinate]<<endl;
  for(int l_ = 0; l_<lf_shape[view_coordinate]; l_++){
    if(block_start[view_coordinate]>l_) {
      //cout<<"Skipping! "<<block_start[view_coordinate]<<" "<<l_<<endl;
      continue;
    }
    if(block_start[view_coordinate]+block_shape[view_coordinate] - 1 < l_){
      //cout<<"Skipping! "<<block_start[view_coordinate]+block_shape[view_coordinate] - 1 <<" "<<l_<<endl;
      continue;
    } 
    int n_start = round(l_*true_alpha);
    int n_end = (lf_shape[spatial_coordinate]) + round(l_*true_alpha);

    if(true_alpha < 0){
      n_start -= (lf_shape[view_coordinate]-1)*true_alpha;
      n_end -= (lf_shape[view_coordinate]-1)*true_alpha;
    }

    torch::TensorOptions options = torch::TensorOptions();
    at::Tensor indexes =  at::empty({0},options.dtype(at::kLong));
    if(n_start-block_start[spatial_coordinate] < block_shape[spatial_coordinate]){
      int64_t blk_n_start = std::max(n_start-block_start[spatial_coordinate],(int64_t)0);
      int64_t blk_n_end = std::min(n_end-block_start[spatial_coordinate],block_shape[spatial_coordinate]);
      //cout<<"      "<<l_<<": "<<blk_n_start<<" "<<blk_n_end<<endl;
      indexes = (at::range(blk_n_start,blk_n_end-1,1)+block_shape[spatial_coordinate]*l_).to(at::kLong);
    }
    padding_coordinates.push_back(indexes);
  }
  at::Tensor vectorized_padding = torch::cat(padding_coordinates);
  return vectorized_padding;
}

at::Tensor Block4D_::filter2D(const at::Tensor& input, const at::Tensor& kernel) {
    // Ensure the input and kernel are 2D tensors
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(kernel.dim() == 2, "Kernel tensor must be 2D");

    // Get the kernel size
    int64_t kernel_h = kernel.size(0);
    int64_t kernel_w = kernel.size(1);

    // Pad the input tensor
    int64_t pad_h_top = std::ceil(kernel_h / 2.0);
    int64_t pad_h_bottom = std::ceil(kernel_h / 2.0);
    int64_t pad_w_left = std::ceil(kernel_w / 2.0);
    int64_t pad_w_right = std::ceil(kernel_w / 2.0);
    at::Tensor padded_input = at::constant_pad_nd(input, {pad_w_left, pad_w_right, pad_h_top, pad_h_bottom}, 0);
    //std::cout<<"Input Size: "<<input.sizes()<<std::endl;
    //std::cout<<"Kernel Size: "<<kernel.sizes()<<std::endl;
    //std::cout<<"Padded Input Size: "<<padded_input.sizes()<<std::endl;

    // Perform 2D convolution
    at::Tensor output = torch::nn::functional::conv2d(
        padded_input.unsqueeze(0).unsqueeze(0),  // Add batch and channel dimensions
        kernel.unsqueeze(0).unsqueeze(0),       // Add batch and channel dimensions
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding({kernel.size(0) / 2, kernel.size(1) / 2}) // Set stride and padding
    ).squeeze();                                // Remove batch and channel dimensions
   // std::cout<<"Output Size: "<<output.sizes()<<std::endl;

    // Crop the output to match OpenCV's behavior
    int64_t crop_h = input.size(0) + kernel_h;
    int64_t crop_w = input.size(1) + kernel_w;
    //std::cout<<"Crop H: "<<crop_h<<" Crop W: "<<crop_w<<std::endl;
    output = output.index({at::indexing::Slice(1, crop_h), at::indexing::Slice(1, crop_w)});
    

    return output;
}
double Block4D_::getOrientationFromCovariance(double precision, std::array<double,2> dispRange, bool isHorizontal) const{
    double chosenAngle = 0;
    double max = 0;

    at::Tensor cov = this->covFun(isHorizontal);
    int64_t size3,size4;
    if(isHorizontal){
        size3 = this->size[1];
        size4 = this->size[3];
    }else{
        size3 = this->size[0];
        size4 = this->size[2];
    }

    //calc weights
    at::Tensor w1 = torch::ones({size3,1},at::kDouble);
    at::Tensor w2 = torch::ones({1,2*size4-1},at::kDouble);

    at::Tensor weights = filter2D(w1,w1);
    //std::cout<<"Weight Size after filtering!"<<weights.sizes()<<" "<<w2.sizes()<<std::endl;
    weights = weights.matmul(w2);
    //std::cout<<"Weight Size after matmul!"<<weights.sizes()<<std::endl;

    //calc auto-cov
    at::Tensor autocov = cov.index({size3-1,at::indexing::Slice()}).squeeze();
    int p_min = dispRange[0]*(double)(size3 - 1) - 1;
	int p_max = dispRange[1]*(double)(size3 - 1) + 1;

    double* autocov_ptr = autocov.data_ptr<double>();
	std::vector<double> alphas;
	for (int p = p_min; p < p_max; p++)
		alphas.push_back(atan(p / (double)(size3 - 1)));
    
    for (int alpha_idx = 0; alpha_idx < alphas.size(); alpha_idx++){
        at::Tensor model = torch::zeros({2*size3-1,2*size4-1},torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU));
        int64_t row_stride = model.stride(0);
        int64_t col_stride = model.stride(1);
        double* model_ptr = model.data_ptr<double>();
        double alpha = alphas[alpha_idx];
        for (int h = 0;h<2*size3-1;h++){
            //calc shift
            int shift = round((size3 - (h + 1)) * tan(alpha));
            for( int w = 0; w < 2*size4 - 1; w++){
                int w_new = w+shift;
            
                //if in boundaries
                if(w_new >= 0 && w_new < (2*size4-1)){                   

                    int64_t index = h * row_stride + w_new*col_stride;
                    //model[h][w_new] = autocov[w];
                    model_ptr[index] = autocov_ptr[w];
                }
            }
        }
        //std::cout<<"Model Size: "<<model.sizes()<<std::endl;
        model = model * weights;
        //if(alpha == 1) write_tensor(model,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/model.png");
        double result = model.mul(cov).sum().item<double>();
        //std::cout<<model.sizes()<<" "<<cov.sizes()<<std::endl;
        //std::cout<<theta<<": "<<result<<std::endl;
        if (result > max){
            max = result;
            chosenAngle = atan(alpha) * 180.0 / M_PI;
        }
    }

    return -chosenAngle;
}




