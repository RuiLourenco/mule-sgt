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

    // tensor_arr_t ranges;
    // for (auto sz : sizes) {
    //     std::cout<<"sz = "<<sz<<std::endl;
    //     auto range = at::arange(1-sz, sz, opts);
    //     std::cout << "Range for sz=" << sz << ": " << range << std::endl;
    //     ranges.push_back(range);
    // }
        
    auto meshes = at::meshgrid(ranges, "ij");
    return {meshes[0], meshes[1]};
}

#define ADAPTIVE_RHO_CALC 1
#define FLAT_TRANSFORM 1

#define DEBUG 1
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
// at::Tensor SgtSideInfo::constrainedLeastSquares(at::Tensor A, at::Tensor b){
//     // Compute Quadratic and Linear term for QP: 
//     // P = A^T A (Hessian matrix)
//     // q = -A^T b (Linear term)
//     std::cout<<"0"<<std::endl;
//     torch::Tensor P = A.t().mm(A);
//     torch::Tensor q = -A.t().mm(b);
//     std::cout<<"1"<<std::endl;

//     // Convert tensors to raw C arrays (OSQP requires raw pointers)
//     auto P_data = P.contiguous().data_ptr<double>();
//     auto q_data = q.contiguous().data_ptr<double>();

//     std::cout<<"2"<<std::endl;

//     // Setup OSQP problem
//     OSQPSettings settings;
//     osqp_set_default_settings(&settings);
//     std::cout<<"3"<<std::endl;

//     settings.alpha = 1.0;  // Step size parameter
//     settings.verbose = true;

//     OSQPWorkspace *work;
//     OSQPData data;
//     c_int n = 2;  // Number of variables
//     std::cout<<"4"<<std::endl;
//     std::cout<<A.sizes()<<std::endl;

//     // Convert P (Hessian) into sparse format (only diagonal stored)
//     csc *P_sparse = csc_matrix(P.size(0), P.size(1), P.numel(), P_data, nullptr, nullptr);
//     //csc *A_sparse = csc_matrix(A_constrains.size(0), A_constrains.size(1), A_constrains.numel(), A_constrains_data.contiguous().data_ptr<double>(), nullptr, nullptr);
//     std::cout<<"5"<<std::endl;
//     std::vector<c_float> l(n, 0.0), u(n, 0.99999);
    
//     std::cout<<"6"<<std::endl;

//     // Initialize OSQP data
//     data.n = n;
//     data.m = 0;
//     data.P = P_sparse;
//     data.q = q_data;
//     data.l = l.data();
//     data.u = u.data();

//     std::cout<<"I have initialized optimization data!"<<std::endl;
//     // Setup solver
//     osqp_setup(&work, &data, &settings);
//     std::cout<<"setup is complete!"<<std::endl;
//     // Solve the problem
//     osqp_solve(work);
//     std::cout<<"I have found the solution! It should be "<<work->solution->x[0]<<" and "<<work->solution->x[1]<<std::endl;

//     // Extract solution into a Torch tensor (from raw pointer)
//     torch::Tensor x = torch::from_blob(work->solution->x, {n}, torch::kDouble);
//     return x;

// }

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

Block4D_::Block4D_(const at::Tensor& data){
    this->data = data;
    this->size = {data.size(0),data.size(1),data.size(2),data.size(3)};
#if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
#else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
#endif
    this->sgtDomain = false;
}   
Block4D_::Block4D_(std::array<int64_t,4> size){
    this->data = torch::zeros({size[0], size[1], size[2], size[3]}, torch::kInt);
    this->sgtDomain = false;
    this->size = size;
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
    assert(B00.data.size(x1)+B10.data.size(x1) == B01.data.size(x1)+B11.data.size(x1) && "heights don't match" );
    assert(B00.data.size(x2)+B01.data.size(x2) == B10.data.size(x2)+B11.data.size(x2) && "widths don't match");
    this->data = torch::cat({torch::cat({B00.data, B10.data}, x1),torch::cat({B01.data, B11.data}, x1)},x2).to(at::kInt);
    this->size = B00.size;
    this->size[x1] = B00.data.size(x1)+B10.data.size(x1);
    this->size[x2] = B00.data.size(x1)+B01.data.size(x1);
    //std::cout<<" Inside: "<<this->size[0]<<"x"<<this->size[1]<<"x"<<this->size[2]<<"x"<<this->size[3]<<std::endl;
#if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
#else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
#endif
    //std::cout<<" Inside Transform: "<<this->transformSize[0]<<"x"<<this->transformSize[1]<<"x"<<this->transformSize[2]<<"x"<<this->transformSize[3]<<std::endl;

    this->sgtDomain = B00.sgtDomain;
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

    for(int i = 0; i < transform.size(1); i++){
        int bias = 0;
        while(true){
            double reference = transform[bias][i].item<double>();
            if(abs(reference) > eps){
                if(reference < 0){
                    transform.index({at::indexing::Slice(),i}) =  -1 * transform.index({at::indexing::Slice(),i});                }
                break;
            }

            bias++;

        }
    }
    return transform;
}
torch::Tensor Block4D_::computeLaplacian(SgtSideInfo ssi, bool isHorizontal) {

    double theta;
    int64_t height,width;
    double verticalWeight = 1;
    if(isHorizontal){
        theta = ssi.getAngleH();
        height = this->size[1];
        width = this->size[3];

    }else{
        theta = ssi.getAngleV();
        height = this->size[0];
        width = this->size[2];
    }
    theta = 0;

    int64_t num_nodes = width * height;
    torch::Tensor laplacian = torch::zeros({num_nodes, num_nodes},at::kDouble);
    
    double radians = theta * M_PI / 180.0;
    //int64_t mid_y = height / 2;
    
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            //if(y==x) continue;
            int64_t node = y * width + x;
            int64_t degree = laplacian.index({node, node}).item<int64_t>();
            
            // Connect horizontally
            if (x > 0) {
                int64_t left = node - 1;
                laplacian.index_put_({node, left}, -1);
                laplacian.index_put_({left, node}, -1);
                //laplacian.index_put_({node, node}, laplacian.index({node, node}) + 1);
                //laplacian.index_put_({left, left}, laplacian.index({left, left}) + 1);
            }
            if (x < width - 1) {
                int64_t right = node + 1;
                laplacian.index_put_({node, right}, -1);
                laplacian.index_put_({right, node}, -1);
                //laplacian.index_put_({node, node}, laplacian.index({node, node}) + 1);
                //laplacian.index_put_({right, right}, laplacian.index({right, right}) + 1);
            }
            
            // Compute vertical connection
            double x_intersect = x + tan(radians);
            int64_t closest_x = std::round(x_intersect);
            closest_x = std::clamp(closest_x, int64_t(0), width - 1);
            if (y < height - 1) {
                int64_t target = (y + 1) * width + closest_x;
                laplacian.index_put_({node, target}, -1*verticalWeight);
                laplacian.index_put_({target, node}, -1*verticalWeight);
                //laplacian.index_put_({node, node}, laplacian.index({node, node}) + 1);
                //laplacian.index_put_({target, target}, laplacian.index({target, target}) + 1);
            }
        }
    }
    std::cout<<"stuff"<<std::endl;
    for(int i = 0; i < laplacian.size(0);++i){
        laplacian[i][i] = -laplacian[i].sum();
    }
    std::cout<<"and things"<<std::endl;
    return laplacian;
}  

void Block4D_::sgtTransform(double scale){
    //std::cout<<"SGT transform"<<std::endl;
    
    //ssi.print();
    at::Tensor modelCovMatH = this->computeLaplacian(ssi,true);
    std::cout<<"Horizontal Over"<<std::endl;
    at::Tensor modelCovMatV = this->computeLaplacian(ssi,false);
       std::cout<<"Vertical Over"<<std::endl;

    at::Tensor eigValsH,eigValsV;
    
    
    at::Tensor flatBlock = scale * getFlatBlock();
    
    



    
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true,eigValsH);
    at::Tensor sgtMatrixV =   getSgtTransformMatrix(modelCovMatV,false,eigValsV);
    std::cout<<"Got the Matrices!"<<std::endl;
    at::Tensor basis = get2DBasis(sgtMatrixH, 0);
    std::cout<<"WritingBasisImage: "<<basis.max().item()<<" "<<basis.min().item()<<" "<<(basis.max()-basis.min()).item()<<std::endl;
    write_tensor(basis, "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/basisViz.png",{0.0589256,0.0589256+1.33851e-14});

    write_tensor(flatBlock,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/2D-b4transform.png");
    at::Tensor flatTransform = sgt(flatBlock,sgtMatrixH,sgtMatrixV,eigValsH,eigValsV);
    write_tensor(log(1+(flatTransform * flatTransform)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/2DFull.png");
    
#if FLAT_TRANSFORM == 1
    at::Tensor transform = flatTransform.unsqueeze(0).unsqueeze(0);
#else
    at::Tensor transform = orderCoefficientsByFrequency(flatTransform,sgtMatrixH,sgtMatrixV);       
    // int border_size = 4;
    // at::Tensor vizTransform = at::ones({flatTransform.size(0)+border_size*8,flatTransform.size(1)+border_size*8},flatTransform.dtype());
    // int begin_i = 0;
    // for(int i = 0; i < 9; i++){
    //     int begin_j = 0;
    //     for(int j = 0; j < 9; j++){
    //         //std::cout<<begin_i<<" "<<begin_i+32<<"| "<<begin_j<<" "<<begin_j+32<<std::endl;
    //         vizTransform.index({at::indexing::Slice({begin_i,begin_i+32}),at::indexing::Slice({begin_j,begin_j+32})}) = transform[i][j]; 
    //         begin_j += 32+border_size;

    //         std::string filename = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/4D"+std::to_string(i)+std::to_string(j)+".png";
            
    //         write_tensor(log2(1+(transform[i][j]*transform[i][j])),filename,{0,log2(1+(transform[0][0]*transform[0][0])).max().item<double>()});
        
    //     }
    //     begin_i += 32+border_size;

    // }
    //write_tensor(log2(1+(vizTransform*vizTransform)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/4DTransformViz.png",{0,log2(1+(vizTransform*vizTransform)).max().item<double>()});
#endif
    this->data = transform.round().to(at::kInt).contiguous();
    this->sgtDomain = true;
}





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> findLocalMaxima(const torch::Tensor& tensor, float relative_threshold = 0.0) {
    // Ensure the input is a 2D tensor
    TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2D");

    // Find the global maximum value in the tensor
    auto global_max = tensor.max().item<float>();

    // Calculate the absolute threshold based on the relative threshold
    float absolute_threshold = global_max * relative_threshold;

    // Pad the tensor to handle border elements
    auto padded_tensor = torch::constant_pad_nd(tensor, {1, 1, 1, 1}, -std::numeric_limits<float>::infinity());

    // Get the size of the original tensor
    auto rows = tensor.size(0);
    auto cols = tensor.size(1);

    // Create shifted versions of the tensor to compare with neighbors
    std::vector<torch::Tensor> neighbors;
    std::vector<std::pair<int64_t, int64_t>> shifts = {
        {0, 0},   // center
        {0, 1},   // right
        {0, -1},  // left
        {1, 0},   // down
        {-1, 0},  // up
        {1, 1},   // down-right
        {1, -1},  // down-left
        {-1, 1},  // up-right
        {-1, -1}  // up-left
    };

    for (const auto& shift : shifts) {
        neighbors.push_back(padded_tensor.slice(0, 1 + shift.first, 1 + shift.first + rows)
                                         .slice(1, 1 + shift.second, 1 + shift.second + cols));
    }

    // Stack all shifted tensors along a new dimension
    auto neighbor_stack = torch::stack(neighbors);

    // Compare the center with all its neighbors
    auto local_maxima = (tensor > neighbor_stack.slice(0, 1, neighbor_stack.size(0))).all(0);

    // Apply the threshold to filter out insignificant maxima
    local_maxima = local_maxima & (tensor >= absolute_threshold);

    // Get the indices of local maxima
    auto maxima_indices = local_maxima.nonzero();

    // Extract vertical (n) and horizontal (m) coordinates
    auto n = maxima_indices.select(1, 0);
    auto m = maxima_indices.select(1, 1);

    // Extract the values of the peaks
    auto peaks = tensor.index({n, m});

    return std::make_tuple(peaks, n, m);
}


at::Tensor Block4D_::get2DBasis(at::Tensor sgtMatrix, int n) const{
    return sgtMatrix.index({at::indexing::Slice(),n}).reshape({this->size[0],this->size[2]});
}

double deg2rad(double angle){
    return angle * M_PI / 180.0;
}
std::array<double,2> Block4D_::getMainFrequency(at::Tensor basisFunction,double angle, int index) const{
    using namespace torch::fft;
    int scale = 4;
    double slope = -(angle);
    double orthAngle = slope+90;
    std::array<int64_t,2> fftSize = {4*basisFunction.size(0),4*basisFunction.size(1)};

    int bias_h = round(fftSize[1]/2);
    int bias_v = round(fftSize[0]/2);

    //std::cout<<bias_h<<" "<<bias_v<<std::endl;
    double weight_h = 2.0/fftSize[1];
    double weight_v = 2.0/fftSize[0];

    at::Tensor v_th = torch::tensor({sin(deg2rad(orthAngle)),cos(deg2rad(orthAngle))});
    

    auto block_dft = fftshift(fftn(basisFunction, fftSize).abs());

    auto [peaks,w_k,w_m] = findLocalMaxima(block_dft,0.2);
    // std::cout<<peaks<<std::endl;
    // std::cout<<w_k<<std::endl;
    // std::cout<<w_m<<std::endl;
    #if DEBUG == 1
    if(index == 96){
        write_tensor(block_dft,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/frequency.png");
    }
    #endif
    auto [peaks_unq, unq] = at::_unique(peaks.round(5),true,true);
    auto w_k_unq = w_k.index({unq});
    auto w_m_unq = w_m.index({unq});

    w_k_unq = (w_k_unq - bias_v)*weight_v;
    w_m_unq = (w_m_unq - bias_h)*weight_h;
    //   std::cout<<peaks_unq<<" "<<unq<<std::endl;
    // std::cout<<w_k_unq<<std::endl;
    // std::cout<<w_m_unq<<std::endl;
    auto w = at::stack({w_k_unq,w_m_unq},0);
    at::Tensor w_th = at::mm(v_th.unsqueeze(0),w).squeeze();
    auto w_th_min = w_th.abs().min();
    auto min_index = w_th.argmin();
    auto w_m_min = w_m_unq.index({min_index}).abs();
    return {w_th_min.item<double>(),w_m_min.item<double>()};

}
at::Tensor Block4D_::getFrequencyOrdering(at::Tensor basisFrequenciesH, at::Tensor basisFrequenciesV){

    double angleWeight = 4;
    double spatialWeight = 1;
    at::Tensor ordering = at::zeros({basisFrequenciesH.size(0),basisFrequenciesV.size(0)},at::kDouble);
    int64_t orderingIndex = 0;
    for(int i = 0; i < basisFrequenciesH.size(0);i++){
        for(int j = 0; j < basisFrequenciesV.size(0);j++){
            double distance = sqrt((angleWeight * (basisFrequenciesH[i][0]*basisFrequenciesH[i][0])
                                   + angleWeight * (basisFrequenciesV[j][0]*basisFrequenciesV[j][0])
                                   + spatialWeight * (basisFrequenciesH[i][1] *basisFrequenciesH[i][1])
                                   + spatialWeight * (basisFrequenciesV[j][1] * basisFrequenciesV[j][1])
                                   ).item<double>());
            //if(i == 0) std::cout<<"("<<basisFrequenciesH[i][0].item<double>()<<" "<<basisFrequenciesH[i][1].item<double>()<<" "<<basisFrequenciesV[j][0].item<double>()<<" "<<basisFrequenciesV[j][1].item<double>()<<") = " <<distance<<std::endl;
            ordering[j][i] = distance;
        }
    }
    write_tensor(ordering,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/anOrdering4.png");
    auto [orderingSorted,orderingIndices] = ordering.flatten().sort({},false);
    std::cout<<orderingSorted.index({at::indexing::Slice(0,10)}).unsqueeze(0)<<std::endl;
    return orderingIndices;
}

at::Tensor Block4D_::getOrdinalFrequencies(const at::Tensor& basisFrequencies) const {
    std::cout<<"Something is gonna get fucked up"<<std::endl;
    auto [sortedTh,indexTh] = basisFrequencies.index({at::indexing::Slice(),0}).sort(true,0,false);
    torch::Tensor range = torch::arange(indexTh.size(0), indexTh.options());
    torch::Tensor orderTh = torch::empty_like(indexTh);
    orderTh.scatter_(0, indexTh, range);
    at::Tensor ordinalTh = (orderTh/transformSize[3]).floor().to(at::kLong); //TO DO: This does not work for rectangular blocks.

    
    at::Tensor basisFrequenciesM = basisFrequencies.index({at::indexing::Slice(),1});
    at::Tensor basisFrequenciesTh = basisFrequencies.index({at::indexing::Slice(),0});
    at::Tensor frequencyWeight = basisFrequenciesM*basisFrequenciesM; //+ 200*(basisFrequenciesTh*basisFrequenciesTh);
    at::Tensor basisFrequenciesMReordered = torch::empty_like(basisFrequenciesM);
    basisFrequenciesMReordered.scatter_(0,orderTh,frequencyWeight);
    //std::cout<<"basisFrequenciesM "<<std::endl<<basisFrequenciesM.index({at::indexing::Slice(32,64)})<<std::endl;
    //auto [sortedM,ordinalM] = basisFrequencies.index({at::indexing::Slice(),1}).sort(true,0,false);
    at::Tensor ordinalM = at::zeros(ordinalTh.sizes(),at::kLong);
    //std::cout<<ordinalTh.unsqueeze(0)<<std::endl;
    at::Tensor orderedOrdinalTh = ordinalTh.index({indexTh});

    //std::cout<<"We should be good still"<<std::endl;
    
    for(int i = 0; i <transformSize[1];i++){
        int originalIndexBias = i*transformSize[3]; 
        at::Tensor dbgOrdinalSegment = orderedOrdinalTh.index({at::indexing::Slice(i*transformSize[3],(i+1)*transformSize[3])});
        at::Tensor freqSegment = basisFrequenciesMReordered.index({at::indexing::Slice(i*transformSize[3],(i+1)*transformSize[3])});
        auto [sortedMSegment,indexM] = freqSegment.sort(true,0,false);
        torch::Tensor range2 = torch::arange(indexM.size(0), indexM.options());
        torch::Tensor segmentOrdinalM = torch::empty_like(indexM);
        segmentOrdinalM.scatter_(0, indexM, range2);
        ordinalM.index({at::indexing::Slice(i*transformSize[3],(i+1)*transformSize[3])}) = segmentOrdinalM;
        //std::cout<<at::stack({range2+originalIndexBias,freqSegment,segmentOrdinalM},0)<<std::endl;
    }
    ordinalM = ordinalM.index({{orderTh}});
    //std::cout<<"ordinalM: "<<ordinalM.index({at::indexing::Slice(32,39)}).unsqueeze(0) <<std::endl;
    //std::cout<<"cardinalM: "<<basisFrequenciesM.index({at::indexing::Slice(32,39)}).unsqueeze(0) <<std::endl;
    //std::cout<<"cardinalMReordered: "<<basisFrequenciesMReordered.index({at::indexing::Slice(32,39)}).unsqueeze(0) <<std::endl;

    //std::cout<<"ordinalTH: "<<ordinalTh.sizes()<<std::endl<< ordinalTh.index({at::indexing::Slice()}).unsqueeze(0) <<std::endl;
    //std::cout<<"ordinalM: "<<ordinalM.sizes()<<std::endl<< ordinalM.unsqueeze(0) <<std::endl;
    
    
    //std::cout<<"basisFrequenciesTH:"<<std::endl<<basisFrequencies.index({at::indexing::Slice(0,10),0}).unsqueeze(0)<<std::endl;

    return at::stack({ordinalTh,ordinalM},0);
}
at::Tensor getFullOrdinalFrequency(const at::Tensor& basisFrequenciesH, const at::Tensor& basisFrequenciesV,const at::Tensor& ordinalFrequenciesH, const at::Tensor& ordinalFrequenciesV, at::Tensor& frequencies){
    //at::Tensor fullOrdinalFrequencies = at::zeros({ordinalFrequenciesV.size(1),ordinalFrequenciesH.size(1),4},at::kDouble);
    //frequencies = at::zeros(fullOrdinalFrequencies.sizes(),at::kDouble);
    at::Tensor ordinalH = ordinalFrequenciesH.permute({1, 0}).unsqueeze(0);
    at::Tensor ordinalV = ordinalFrequenciesV.permute({1, 0}).unsqueeze(1);
    ordinalH = ordinalH.expand({ordinalFrequenciesV.size(1), ordinalFrequenciesH.size(1), 2});
    ordinalV = ordinalV.expand({ordinalFrequenciesV.size(1), ordinalFrequenciesH.size(1), 2});
    at::Tensor fullOrdinalFrequencies = torch::cat({ordinalV, ordinalH}, /*dim=*/2);
    at::Tensor indexes = torch::tensor({0,2,1,3},torch::kLong);
    fullOrdinalFrequencies = fullOrdinalFrequencies.index({at::indexing::Slice(),at::indexing::Slice(),indexes});
    fullOrdinalFrequencies = fullOrdinalFrequencies.flatten(0,1).t();
    return fullOrdinalFrequencies;
}
at::Tensor Block4D_::to4DTransform(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const {
    at::Tensor output = at::zeros(transformSize,coefficients.dtype());
     auto flattened_indices = fullOrdinalFrequencies[0] * output.size(1) * output.size(2) * output.size(3)
                           + fullOrdinalFrequencies[1] * output.size(2) * output.size(3)
                           + fullOrdinalFrequencies[2] * output.size(3)
                           + fullOrdinalFrequencies[3];
     output.view(-1).index_add_(0, flattened_indices, coefficients);                      

    return output;
}

at::Tensor Block4D_::from4DTransform(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients4D) const {
    at::Tensor coefficients = at::zeros({coefficients4D.numel()},coefficients4D.dtype());
    //std::cout<<coefficients.sizes()<<std::endl;
    //std::cout<<coefficients4D.sizes()<<std::endl;
    for(int i = 0; i < fullOrdinalFrequencies.size(1); i++){
        //output[fullOrdinalFrequencies[0][i].item<int64_t>()][fullOrdinalFrequencies[1][i].item<int64_t>()][fullOrdinalFrequencies[2][i].item<int64_t>()][fullOrdinalFrequencies[3][i].item<int64_t>()] = 1000000;
        coefficients[i] = coefficients4D[fullOrdinalFrequencies[0][i].item<int64_t>()][fullOrdinalFrequencies[1][i].item<int64_t>()][fullOrdinalFrequencies[2][i].item<int64_t>()][fullOrdinalFrequencies[3][i].item<int64_t>()] ;
    }
    coefficients = coefficients.reshape({1,1,size[0]*size[2],size[1]*size[3]});
    return coefficients;
}
// at::Tensor Block4D_::getFullOrder(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const {
    
//     at::Tensor output = at::zeros(coefficients.sizes(),at::kDouble);
//     std::cout<<"Empty Tensor Created"<<std::endl;
//     std::cout<<"fullOrdinalFrequencies: "<<fullOrdinalFrequencies.sizes()<<std::endl;
//     std::cout<<"fullOrdinalFrequencies: "<<std::endl<<fullOrdinalFrequencies.index({at::indexing::Slice(),at::indexing::Slice({0,10})})<<std::endl;
//     int i = 0;
//     int64_t maxSum = (this->size[0]-1)+(this->size[1]-1);
//     int64_t maxInnerSum = (this->size[2]-1)+(this->size[3]-1);
//         std::cout<<"Max Sums Computed. Outer Max: "<<maxSum<<" Inner Max: "<<maxInnerSum<<std::endl;


//     for (int summ = 0; summ < maxSum; summ++){
//         std::vector<int64_t> indices;
//         for (int64_t w_l = 0; w_l < transformSize[0]; w_l++){
//             for (int64_t w_k = 0; w_k < transformSize[1]; w_k++){

//             at::Tensor currentW = fullOrdinalFrequencies.index({at::indexing::Slice(),j});
            
//             //std::cout<<j<<":"<<currentW<<std::endl;
//             int64_t sumAngle = (currentW[0] + currentW[1]).item<int64_t>();
//             //if(summ == 0) std::cout<<sumAngle<<" ("<<currentW[0].item()<<" "<<currentW[1].item()<<std::endl;

//             // if(currentW[0].item<int64_t>() == 0 && currentW[1].item<int64_t>() == 0 && summ == 0){
//             //     std::cout<<"apt apt apt: "<<sumAngle<<" ("<<currentW[0].item()<<" "<<currentW[1].item()<<" )"<<coefficients[indices[j]].item()<<std::endl;
//             // }
//             if (sumAngle == summ){
//                 indices.push_back(j);
//             }
//         }
//         std::cout<<"Finished Outer Loop "<<indices.size()<<std::endl;


//         for (int innerSum = 0; innerSum < maxInnerSum; innerSum++){
//                 for(int64_t j = 0; j < indices.size(); j++){
//                     at::Tensor currentW = fullOrdinalFrequencies.index({at::indexing::Slice(),indices[j]});
//                     int64_t sumSpace = (currentW[2] + currentW[3]).item<int64_t>();
//                     if(currentW[0].item<int64_t>() == 0 && currentW[1].item<int64_t>() == 0 && innerSum == 0){
//                         //std::cout<<"apt apt: "<<sumSpace<<" "<<currentW[2]<<" "<<currentW[3]<<" "<<coefficients[indices[j]]<<std::endl;
//                     }
//                     if (sumSpace == innerSum){
//                         output[i] = coefficients[indices[j]];
//                         //std::cout<<"hello "<<sumSpace<<" "<<innerSum<<" "<<indices[j]<<" "<<coefficients[indices[j]]<<std::endl;
//                         i++;
//                     }
//                 }
//            //std::cout<<"Finished Inner Loop "<<innerSum<<std::endl;    
//         }
//     }
//     std::cout<<"Output: "<<std::endl;
//     std::cout<<output.index({at::indexing::Slice(0,10)})<<std::endl;
//     return output;
// }

at::Tensor Block4D_::recoverOrder(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const {

    at::Tensor order = at::zeros(coefficients.sizes(),at::kDouble);
    int i = 0;
    int64_t maxSum = (this->size[0]-1)+(this->size[1]-1);
    int64_t maxInnerSum = (this->size[2]-1)+(this->size[3]-1);

    for (int summ = 0; summ < maxSum; summ++){
        std::vector<int64_t> indices;
        for (int64_t j = 0; j < fullOrdinalFrequencies.size(2); j++){
            at::Tensor currentW = fullOrdinalFrequencies.index({at::indexing::Slice(),j});
            int64_t sumAngle = (currentW[0] + currentW[1]).item<int64_t>();
            if (sumAngle == summ){
                indices.push_back(j);
            }
        }

        for (int innerSum = 0; innerSum < maxInnerSum; innerSum++){
            for(int64_t j = 0; j < indices.size(); j++){
                at::Tensor currentW = fullOrdinalFrequencies.index({at::indexing::Slice(),indices[j]});
                int64_t sumSpace = (currentW[2] + currentW[3]).item<int64_t>();
                if (sumSpace == innerSum){
                    order[i] = indices[j];
                    i++;
                }
            }
            
        }
    }
    at::Tensor output = coefficients.index({order});
    return output;
}
                
void Block4D_::view4DFrequencies(at::Tensor fullOrdinalFrequencies, at::Tensor frequencies) const{
    at::Tensor output = at::zeros(this->transformSize,frequencies.dtype());
    for( int j = 0; j < frequencies.size(0);j++){
        for(int i = 0; i < fullOrdinalFrequencies.size(1); i++){
            //output[fullOrdinalFrequencies[0][i].item<int64_t>()][fullOrdinalFrequencies[1][i].item<int64_t>()][fullOrdinalFrequencies[2][i].item<int64_t>()][fullOrdinalFrequencies[3][i].item<int64_t>()] = 1000000;
            output[fullOrdinalFrequencies[0][i].item<int64_t>()][fullOrdinalFrequencies[1][i].item<int64_t>()][fullOrdinalFrequencies[2][i].item<int64_t>()][fullOrdinalFrequencies[3][i].item<int64_t>()] = frequencies[j][i];
        }
        //std::cout<<output.min().item()<<" "<<output.max().item()<<std::endl;
        int border_size = 4;
        at::Tensor vizTransform = at::zeros({output.size(0)*output.size(2)+border_size*8,output.size(1)*output.size(3)+border_size*8},at::kDouble);
        int begin_i = 0;
        for(int k = 0; k < this->transformSize[1]; k++){
            int begin_j = 0;
            for(int l = 0; l < this->transformSize[0]; l++){
               //std::cout<<j<<","<<k<<","<<l<<std::endl;
                //std::cout<<"hello!"<<std::endl;
                //std::cout<<begin_i<<" "<<begin_i+32<<"| "<<begin_j<<" "<<begin_j+32;
                vizTransform.index({at::indexing::Slice({begin_i,begin_i+this->transformSize[3]}),at::indexing::Slice({begin_j,begin_j+this->transformSize[3]})}) = output[k][l]; 
                begin_j += this->transformSize[2]+border_size;
                //std::cout<<" done"<<std::endl;
            }
            begin_i += this->transformSize[3]+border_size;
        }
        //std::cout<<"vizTransform: "<<vizTransform.min().item()<<" "<<vizTransform.max().item()<<std::endl;

        write_tensor(vizTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/Frequency4D_"+std::to_string(j)+".png");
        // write_tensor(vizTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/Frequency4D_"+std::to_string(j)+".png",{0,vizTransform.max().item<double>()});
    }
}

at::Tensor Block4D_::orderCoefficientsByFrequency(const at::Tensor& coefficientBlock,const at::Tensor& sgtMatrixH,const at::Tensor& sgtMatrixV) const{
    //std::cout<<"Ordering Coefficients"<<std::endl;
    at::Tensor flatBlockDouble = coefficientBlock.to(at::kDouble);
    //std::cout<<log2(1+(flatBlockDouble[37][0].abs().item<double>()))<<" "<<log2(1+(flatBlockDouble[37][0].abs().item<double>()))<<std::endl;
    double angleH = this->ssi.getAngleH();
    double angleV = this->ssi.getAngleV();
    at::Tensor basisFrequenciesH = getBasisFrequencies(sgtMatrixH,angleH);
    at::Tensor basisFrequenciesV = getBasisFrequencies(sgtMatrixV,angleV);
    //std::cout<<"Index of Max Frequency: "<<basisFrequenciesV.index({at::indexing::Slice(),0}).argmax().item<int64_t>()<<std::endl;
    at::Tensor basisFrequenciesViewTH = basisFrequenciesH.index({at::indexing::Slice(),0}).squeeze().repeat({basisFrequenciesV.size(0),1});
    at::Tensor basisFrequenciesViewTV = basisFrequenciesV.index({at::indexing::Slice(),0}).t().repeat({basisFrequenciesH.size(0),1}).t();
    write_tensor(basisFrequenciesViewTH,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/BasisFrequenciesH.png");
    write_tensor(basisFrequenciesViewTV,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/BasisFrequenciesV.png");

    at::Tensor ordinalFrequenciesH = getOrdinalFrequencies(basisFrequenciesH);
    at::Tensor ordinalFrequenciesV = getOrdinalFrequencies(basisFrequenciesV);
    //std::cout<<"RelevantH: "<<ordinalFrequenciesH.t()[0]<<std::endl;
#if DEBUG == 1
    std::cout<<"Acquired Basis Frequencies"<<std::endl;
    //std::cout<<"RelevantH: "<<basisFrequenciesH[0]<<std::endl;
    std::cout<<"RelevantV: "<<basisFrequenciesV[96]<<std::endl;
    std::cout<<"RelevantV: "<<ordinalFrequenciesV.t()[96]<<std::endl;
#endif    
    
        std::cout<<"Acquired Frquency Order of Basis"<<std::endl;
    at::Tensor frequencies;
    
    at::Tensor ordinalFrequenciesFull = getFullOrdinalFrequency(basisFrequenciesH, basisFrequenciesV,ordinalFrequenciesH,ordinalFrequenciesV, frequencies);
        std::cout<<"Acquired Full Coefficient Order"<<std::endl;
    //std::cout<<ordinalFrequenciesH.index({at::indexing::Slice(),at::indexing::Slice()})<<std::endl;
    //std::cout<<ordinalFrequenciesFull.index({at::indexing::Slice(),at::indexing::Slice({32,64})})<<std::endl;
    //std::cout<<"frequencies: "<<std::endl<<frequencies.index({at::indexing::Slice(),at::indexing::Slice(32,64)})<<std::endl;
    //view4DFrequencies(ordinalFrequenciesFull,frequencies);
    //at::Tensor reOrdered = getFullOrder(ordinalFrequenciesFull,flatBlockDouble.flatten());
    at::Tensor reOrdered = to4DTransform(ordinalFrequenciesFull,flatBlockDouble.flatten());
        std::cout<<"Reordered Coefficients"<<std::endl;
    //view4DFrequencies(ordinalFrequenciesFull,frequencies);
    //reOrdered = reOrdered.reshape({this->transformSize[0],this->transformSize[1],this->transformSize[2],this->transformSize[3]});
    std::cout<<"Reshaped to 4D"<<std::endl;

    return reOrdered;    
}



at::Tensor Block4D_::reverseOrderCoefficientsByFrequency(const at::Tensor& coefficientBlock,const at::Tensor& sgtMatrixH,const at::Tensor& sgtMatrixV) const{
    std::cout<<"Ordering Coefficients"<<std::endl;
    //at::Tensor flatBlockDouble = coefficientBlock.to(at::kDouble);
    double angleH = this->ssi.getAngleH();
    double angleV = this->ssi.getAngleV();
    at::Tensor basisFrequenciesH = getBasisFrequencies(sgtMatrixH,angleH);
    at::Tensor basisFrequenciesV = getBasisFrequencies(sgtMatrixV,angleV);
        std::cout<<"Acquired Basis Frequencies"<<std::endl;

    at::Tensor ordinalFrequenciesH = getOrdinalFrequencies(basisFrequenciesH);
    at::Tensor ordinalFrequenciesV = getOrdinalFrequencies(basisFrequenciesV);
        std::cout<<"Acquired Frquency Order of Basis"<<std::endl;
    at::Tensor frequencies;
    
    at::Tensor ordinalFrequenciesFull = getFullOrdinalFrequency(basisFrequenciesH, basisFrequenciesV,ordinalFrequenciesH,ordinalFrequenciesV, frequencies);
        std::cout<<"Acquired Full Coefficient Order"<<std::endl;
    //std::cout<<ordinalFrequenciesH.index({at::indexing::Slice(),at::indexing::Slice()})<<std::endl;
    //std::cout<<ordinalFrequenciesFull.index({at::indexing::Slice(),at::indexing::Slice({32,64})})<<std::endl;
    //std::cout<<"frequencies: "<<std::endl<<frequencies.index({at::indexing::Slice(),at::indexing::Slice(32,64)})<<std::endl;

    //at::Tensor reOrdered = getFullOrder(ordinalFrequenciesFull,flatBlockDouble.flatten());
    at::Tensor reOrdered = from4DTransform(ordinalFrequenciesFull,coefficientBlock);
        std::cout<<"Reordered Coefficients"<<std::endl;
    //view4DFrequencies(ordinalFrequenciesFull,frequencies);
    //reOrdered = reOrdered.reshape({this->transformSize[0],this->transformSize[1],this->transformSize[2],this->transformSize[3]});
    std::cout<<"Reshaped to 2D:  "<<reOrdered.sizes()<<std::endl;

    return reOrdered;    
}
// at::Tensor Block4D_::recoverCoefficientOrder(const at::Tensor& reOrderedCoefficientBlock, const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) const{
//         double angleH = this->ssi.getAngleH();
//     double angleV = this->ssi.getAngleV();
//     at::Tensor basisFrequenciesH = getBasisFrequencies(sgtMatrixH,angleH);
//     at::Tensor basisFrequenciesV = getBasisFrequencies(sgtMatrixV,angleV);
    
//     at::Tensor ordinalFrequenciesH = getOrdinalFrequencies(basisFrequenciesH);
//     at::Tensor ordinalFrequenciesV = getOrdinalFrequencies(basisFrequenciesV);
//     at::Tensor ordinalFrequenciesFull = getFullOrdinalFrequency(ordinalFrequenciesH,ordinalFrequenciesV);
//     at::Tensor recovered = recoverOrder(ordinalFrequenciesFull,reOrderedCoefficientBlock.flatten());
//     recovered = recovered.reshape({1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]});
//     return recovered;
// }
at::Tensor Block4D_::frequencyOrderedSgt(at::Tensor flatBlock, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV){
    flatBlock = flatBlock.to(at::kDouble);
    double angleH = this->ssi.getAngleH();
    double angleV = this->ssi.getAngleV();
    at::Tensor basisFrequenciesH = getBasisFrequencies(sgtMatrixH,angleH);
    at::Tensor basisFrequenciesV = getBasisFrequencies(sgtMatrixV, angleV);
    std::cout<<"Got Frequencies!"<<std::endl;
    at::Tensor ordering = getFrequencyOrdering(basisFrequenciesH,basisFrequenciesV);
    std::cout<<"Got the Ordering!"<<std::endl;
    std::cout<<ordering.index({at::indexing::Slice(0,10)}).unsqueeze(0)<<std::endl;
    std::cout<<flatBlock.dtype()<<std::endl;
    at::Tensor sgtCoefficients = at::mm(at::mm(sgtMatrixV.t(), flatBlock), sgtMatrixH);

    std::cout<<"Got the Coefficients!"<<std::endl;
    std::cout<<"THIS > "<<basisFrequenciesH[2][0]<<std::endl;
    at::Tensor orderedCoefficients = reOrderCoefficients(sgtCoefficients,ordering);
    std::cout<<"Got the Ordered Coefficients!"<<std::endl;
    std::cout<<orderedCoefficients.index({at::indexing::Slice(0,10)}).unsqueeze(0)<<std::endl;
    //at::Tensor sortedSgt = triangleSorting(transformSize, orderedCoefficients);
    //at::Tensor sortedSgt = squareSorting(transformSize, orderedCoefficients);
    at::Tensor sortedSgt = quadTreeSorting(transformSize, orderedCoefficients);
    std::cout<<"Got the Sorted SGT!"<<std::endl;
    std::cout<<sortedSgt.sizes()<<std::endl;
    write_tensor(log(1+(sortedSgt * sortedSgt)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/aFrequencyQuadTreeSgT.png");
    write_tensor(log(1+(sgtCoefficients * sgtCoefficients)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/anUnsortedSgT.png");

    return sortedSgt;
}
at::Tensor Block4D_::reOrderCoefficients(const at::Tensor& coefficients, at::Tensor ordering){
    at::Tensor flatCoefficients = coefficients.flatten();
    at::Tensor orderedCoefficients = coefficients.flatten().index({ordering});
    return orderedCoefficients;
}
at::Tensor Block4D_::getBasisFrequencies(const at:: Tensor sgtMatrix, double angle) const{

    at::Tensor basisFrequencies = at::zeros({sgtMatrix.size(1),2},at::kDouble);
    for(int i=0;i<sgtMatrix.size(1);i++){
        
        at::Tensor basis = get2DBasis(sgtMatrix,i);
        
        std::array<double,2> freq = getMainFrequency(basis,angle,i);
        basisFrequencies[i][0] = freq[0];
        basisFrequencies[i][1] = freq[1];
    }
    return basisFrequencies;
}

at::Tensor Block4D_::isgtTransformData(double scale, SgtSideInfo ssi) {
    this->ssi = ssi;
    //SgtSideInfo sortSSI(this->ssi.getAngleV(),this->ssi.getAngleH(),this->ssi.disparityRange);
    //std::cout<<"Inverse Transforming block of size: "<<this->data.sizes()<<std::endl;
    at::Tensor modelCovMatH = this->computeLaplacian(ssi,true);
    at::Tensor modelCovMatV = this->computeLaplacian(ssi,false);  
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
    eigVals = L;
    std::cout<<"This should be Zero: "<<eigVals[0]<<std::endl;
    std::cout<<"Eigen: "<< eigVals.index({at::indexing::Slice(0,10)}).unsqueeze(0)<<std::endl;
    return Q; 
}



at::Tensor Block4D_::sgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV,const at::Tensor& eigValsH,const at::Tensor& eigValsV) {
    at::Tensor block = flatBlock.to(at::kDouble);
    for(int i = 0; i < eigValsH.size(0); i++){
        if (eigValsH[i].item<double>() < 0) {
            //std::cerr<<"eigValsH < 0 Should happen Once!"<<std::endl;
           // exit(-2);
        }
        if (eigValsV[i].item<double>() < 0) {
            //std::cerr<<"eigValsV Should happen Once!< 0"<<std::endl;
            //exit(-2);
        }
    }
    at::Tensor transform = at::mm(at::mm(sgtMatrixV.t(), block), sgtMatrixH);
    
   write_tensor(log(1+(transform * transform)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/originalFinal4D.png");

    return transform   ;
}

at::Tensor Block4D_::isgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) {
    //std::cout<<"sgtMatrixV: "<<sgtMatrixV.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;
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



void Block4D_::isgtTransform(double scale,SgtSideInfo ssi){
    at::Tensor recoveredBlock = isgtTransformData(scale,ssi);
    this->data = recoveredBlock;
    this->sgtDomain = false;
}





SgtSideInfo::SgtSideInfo(int angleVInt, int angleHInt, std::array<double,2> dispRange){

    this->angleVInt = angleVInt;
    this->angleHInt = angleHInt;
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
}









void SgtSideInfo::print(){
    //std::cout<<"Angle V = "<<getAngleV()<< " Angle H = "<<getAngleH()<<std::endl;
    //std::cout<<"Angle Code H: "<<angleHInt<<" Angle Code V: "<<angleVInt<<std::endl;
}
SgtSideInfo::SgtSideInfo(std::array<double,2> dispRange){
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
}




int SgtSideInfo::codeAngle(double theta) const{
    return theta * getAngleCodeScale() + getAngleCodeBias();
}

double SgtSideInfo::DecodeAngle(int dCode) const{
    return (dCode - getAngleCodeBias()) / getAngleCodeScale();
}


void SgtSideInfo::setAngleVCode(int code){
    this->angleVInt = code;
}
void SgtSideInfo::setAngleHCode(int code){
    this->angleHInt = code;
}
int SgtSideInfo::getAngleVCode() const{
    return this->angleVInt;
}
int SgtSideInfo::getAngleHCode() const{
    return this->angleHInt;
}
double SgtSideInfo::getAngleCodeBias() const{   
    //std::cout<<"Angle Range Bias: "<<this-> angleRange[0]<<" "<<this->angleRange[1]<<std::endl;
    //std::cout<<-angleRange[0]<<" "<<PRECISION_ANGLE<<std::endl;
    return -angleRange[0]/PRECISION_ANGLE;
}
double SgtSideInfo::getAngleCodeScale() const{
    return 1/PRECISION_ANGLE;
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

int SgtSideInfo::getAnglePrecision() const{
    return std::ceil(log2(codeAngle(this->angleRange[1])));
}

void SgtSideInfo::setAngleV(double theta){
    this->angleVInt =codeAngle(theta);
}
void SgtSideInfo::setAngleH(double theta){
    this->angleHInt =codeAngle(theta);
}
void SgtSideInfo::setAngleVFromDisparity(double d){
    this->angleVInt =codeAngle(atan(d) * 180 / acos(-1));
}
void SgtSideInfo::setAngleHFromDisparity(double d){
    this->angleHInt =codeAngle(atan(d) * 180 / acos(-1));
}






double SgtSideInfo::genDivergence(const at::Tensor& p, const at::Tensor& qRsqrt){
    double eps = 1e-2; 
    //double eps = 1e-0; 
    //double k = 10;
    at::Tensor p_regularized = p+at::eye(p.size(-1),p.options())*eps;
    //saveTensorAsMatlabScript(p,"p");
    //saveTensorAsMatlabScript(p_regularized,"pReg");

    auto prod = at::einsum("...xm,...xy,...yn->...mn", {qRsqrt, p_regularized, qRsqrt});
    auto eigvals = torch::linalg::eigvalsh(prod, "U");
    // double genDivExp =  ((eigvals.mean(-1).log())/ exp(eigvals.log().mean(-1))).item<double>();
    // double regQuotient = 1 + k * genDivExp;
    // double genDiv = genDivExp / regQuotient;
    //std::cout<<eigvals<<std::endl;

    //std::cout<<"Divergence: "<<eigvals.mean(-1).log().item<double>()<< " "<<eigvals.log().mean(-1).item<double>()<<" "<<(eigvals.mean(-1).log() - eigvals.log().mean(-1)).item<double>() <<std::endl;
   return (eigvals.mean(-1).log() - eigvals.log().mean(-1)).item<double>();
   //return genDiv;
}


at::Tensor Block4D_::iSqrtCovMat(at::Tensor covMat, bool isHorizontal ) const{
    auto [L, Q] = torch::linalg::eigh(covMat, "U");
    //std::cout<<"Cov Mat Size:"<<covMat.sizes()<<std::endl;
    //std::cout<<"L Size:"<<L.sizes()<<std::endl;
    //std::cout<<"Q Size:"<<Q.sizes()<<std::endl;
    // get largest eigenvalue of each matrix
    auto&& [Lmax, Lloc] = L.max(-1, /*keepdims*/true);
    auto Lrel = L / Lmax; // 'relative' eigenvalues
    // take worst case across batches for condition number
    auto&& [Lmin, min_where]  = L.view({-1, L.size(-1)}).min(0);
    //std::cout<<Lmin<<std::endl;
    // indices of eigenpairs to keep
    auto Lkeep = Lmin > 1e-4;
    // take
    auto Lselect = L.index({"...", Lkeep});
    auto Qselect = Q.index({"...", Lkeep});
    //std::cout<<"LSelect = "<<Lselect.sizes()<<std::endl;
    // safely take the inverse sqrt
    auto Lrsqrt = Lselect.rsqrt_();

    auto result = at::einsum("...x,...x->...x", {Qselect, Lrsqrt});
    return result;
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


std::array<double,2> SgtSideInfo::angleRangeFromDispRange(std::array<double,2> dispRange){
    double minAngle = floor(180/acos(-1) * atan(dispRange[0]));
    double maxAngle = ceil(180/acos(-1) * atan(dispRange[1]));
    return {minAngle,maxAngle};
}
SgtSideInfo::SgtSideInfo(double angleV, double angleH, std::array<double,2> dispRange){
    this->disparityRange = dispRange;
    this->angleRange = angleRangeFromDispRange(dispRange);
    this->setAngleH(angleH);
    this->setAngleV(angleV);
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
void Block4D_::YCbCr2RGB_BT601(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Cb, Block4D_ const &Cr, int Scale) {
    // std::cout<<"Data type = "<<Y.data.dtype()<<std::endl;
    // std::cout<<"Decoding Scale = "<<Scale<<std::endl;
    // std::cout<<"Chrominance Bias Correction: "<<((Scale+1)/2)<<std::endl;
    at::Tensor Ytemp = Y.data.to(at::kDouble);
    auto CbTemp = (Cb.data - ((Scale+1)/2)).to(at::kDouble);
    auto CrTemp = (Cr.data - ((Scale+1)/2)).to(at::kDouble);

    //std::cout<<Ytemp[0][0][0][0].item()<<" "<<CbTemp[0][0][0][0].item()<<" "<<CrTemp[0][0][0][0].item()<<std::endl;
    R = (Ytemp - 0.0000071525 * CbTemp + 1.4020 * CrTemp).round().to(at::kInt);
    G = (Ytemp- 0.34413 * CbTemp - 0.71414 * CrTemp).round().to(at::kInt);
    B = (Ytemp + 1.7720 * CbTemp - 0.000040249 * CrTemp).round().to(at::kInt);
   //std::cout<<Ytemp[0][0][0][0].item()<<" "<<-0.34413 * CbTemp[0][0][0][0].item<double>()<<" "<<- 0.71414 *(CrTemp[0][0][0][0].item<double>())<<" "<<G.data[0][0][0][0]<<std::endl;

}

void Block4D_::YCoCg2RGB(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale) {
    auto CoTemp = Co.data - (Scale+1)/2;
    auto CgTemp = Cg.data - (Scale+1)/2;
    auto t = Y - (CgTemp.bitwise_right_shift(1));
    G = CgTemp + t;
    B.data = t - (CoTemp.bitwise_right_shift(1));
    R.data = B.data + CoTemp;          
}
void Block4D_::RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    static const auto Y_weights = at::tensor({0.299, 0.587, 0.114}, at::kDouble).reshape({3, 1});  
    static const auto Cb_weights = at::tensor({-0.168736, -0.331264, 0.5}, at::kDouble).reshape({3, 1}); 
    static const auto Cr_weights = at::tensor({0.5, -0.418688, -0.081312}, at::kDouble).reshape({3, 1}); 
    static const int D = 1<<((int)log2(Scale+1)-8);
    static const int Y8bitBias = 0;
    static const int CbCr8bitBias = (1<<9);
    //std::cout<<"D = "<<D<<" CbCr8bitBias = "<<CbCr8bitBias<<" Y8bitBias = "<<Y8bitBias<<std::endl;

    auto Ey = R.data.to(at::kDouble)/Scale * Y_weights[0] + G.data.to(at::kDouble)/Scale * Y_weights[1] + B.data.to(at::kDouble)/Scale * Y_weights[2];
    //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Y_weights[0].item()<<" Ey min = "<<Ey.min().item()<<" max = "<<Ey.max().item()<<std::endl;

    Y.data = ((1023 * Ey + Y8bitBias)).round().to(at::kInt);

    auto Ecb = R.data.to(at::kDouble)/Scale * Cb_weights[0] + G.data.to(at::kDouble)/Scale * Cb_weights[1] + B.data.to(at::kDouble)/Scale * Cb_weights[2];
    //std::cout<<"ECb min = "<<Ecb.min().item()<<" max = "<<Ecb.max().item()<<std::endl;
    Cb.data = ((1023 * Ecb + CbCr8bitBias)).round().to(at::kInt);
    auto Ecr = R.data.to(at::kDouble)/Scale * Cr_weights[0] + G.data.to(at::kDouble)/Scale * Cr_weights[1] + B.data.to(at::kDouble)/Scale * Cr_weights[2];
    //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Cr_weights[0]<<" Ecr min = "<<Ecr.min().item()<<" max = "<<Ecr.max().item()<<std::endl;
    
    Cr.data = ((1023 * Ecr + CbCr8bitBias)).round().to(at::kInt); 
    Y.validPositions = R.validPositions;
    Cb.validPositions = R.validPositions;
    Cr.validPositions = R.validPositions;
}

void Block4D_::RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    Co = R.data - B.data;
    auto temp = B.data + Co.data.bitwise_right_shift(1);
    Cg = G.data - temp;
    Y = temp + Cg.data.bitwise_right_shift(1);
    Co.data+= (Scale + 1)/2;
    Cg.data+= (Scale + 1)/2;
    Y.validPositions = R.validPositions;
    Co.validPositions = R.validPositions;
    Cg.validPositions = R.validPositions;        
}

Block4D_ Block4D_::operator + (const Block4D_ &B){
    return Block4D_(this->data + B.data);
}
Block4D_ Block4D_::operator * (const Block4D_ &B){
    return Block4D_(this->data * B.data);
}
Block4D_ Block4D_::operator - (const Block4D_ &B){
    return Block4D_(this->data - B.data);
}
Block4D_ Block4D_::operator / (const int &a){
    return Block4D_(this->data / a);
}
Block4D_ Block4D_::operator + (const int &a){
    return Block4D_(this->data + a);
}
Block4D_ Block4D_::operator - (const int &a){
    return Block4D_(this->data - a);
}
Block4D_ Block4D_::operator * (const int &a){
    return Block4D_(this->data * a);
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
}

Block4D_ Block4D_::clone() const{
    Block4D_ newBlock = Block4D_(this->data.clone());
    newBlock.size = this->size;
    newBlock.transformSize = this->transformSize;
    newBlock.sgtDomain = this->sgtDomain;
    newBlock.ssi = this->ssi;
    newBlock.includesInvalidCorners = this->includesInvalidCorners;

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

// void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s) {

//     this->data[position_t][position_s] = this->data[position_t][position_s]<<shift;

// }



