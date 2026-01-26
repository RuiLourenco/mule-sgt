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
    int64_t fillHLow = std::abs(lightField->preSlantTan) * lightField->data.size(1);

    int64_t fillHHigh = lightField->data.size(3) - fillHLow - size[3];
    int64_t fillVLow = std::abs(lightField->preSlantTan) * lightField->data.size(0);
    int64_t fillVHigh = lightField->data.size(2) - fillVLow - size[2];
    //std::cout<<fillHLow<<" "<<fillHHigh<<" "<<fillVLow<<" "<<fillVHigh<<std::endl;
    //std::cout<<"LightField Position: "<<lightFieldPosition[0]<<" "<<lightFieldPosition[1]<<" "<<lightFieldPosition[2]<<" "<<lightFieldPosition[3]<<std::endl;
    //std::cout<<lightField->preSlantTan<<std::endl;
    
    #if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
    #else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
    #endif
    
    if( (lightFieldPosition[3] > fillHLow && lightFieldPosition[3] < fillHHigh &&
        lightFieldPosition[2] > fillVLow && lightFieldPosition[2] < fillVHigh) || lightField->preSlantTan == 0){
            //std::cout<<"No Invalid Corners. "<<std::endl;
            this->includesInvalidCorners = false;
    }
    else{
        std::array<int64_t,4> lfSize;
        for (int i = 0; i < 4; i++){
            lfSize[i] = lightField->data.size(i);
        }
        at::Tensor validPositionH = get_valid_position(lightField->preSlantTan,lfSize,size,lightFieldPosition,true);
        at::Tensor validPositionV = get_valid_position(lightField->preSlantTan,lfSize,size,lightFieldPosition,false);
        this->validPositions = ValidPositions{validPositionH,validPositionV};
        if(validPositionH.size(0) == size[1] * size[3] && validPositionV.size(0) == size[0] * size[2]){
                //std::cout<<"All positions are valid for subblock copy. "<<std::endl;
                this->includesInvalidCorners = false;
        }
        else{

            // std::cout<<"Some positions are invalid for subblock copy. "<<std::endl;
            // std::cout<<"Valid Positions Horizontal: "<<validPositionH.sizes()<<"/"<<size[1] * size[3]<<std::endl;
            // std::cout<<"Valid Positions Vertical: "<<validPositionV.sizes()<<"/"<<size[0] * size[2]<<std::endl;

            this->includesInvalidCorners = true;
            #if FLAT_TRANSFORM == 1
            this->transformSize = {1,1,validPositionV.size(0),validPositionH.size(0)};
            #else 
            std::cerr<<"Pre-Slant Not implemented for non-flat transform!"<<std::endl;
            #endif
        }
    }


    
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
    if( (!B00.includesInvalidCorners && !B01.includesInvalidCorners && !B10.includesInvalidCorners && !B11.includesInvalidCorners)){
            //std::cout<<"No Invalid Corners. "<<std::endl;
            this->includesInvalidCorners = false;
    }
    else{
        std::array<int64_t,4> lfSize;
        for (int i = 0; i < 4; i++){
            lfSize[i] = this->lightField->data.size(i);
        }
        at::Tensor validPositionH = get_valid_position(lightField->preSlantTan,lfSize,this->size,this->lightFieldPosition,true);
        at::Tensor validPositionV = get_valid_position(lightField->preSlantTan,lfSize,this->size,this->lightFieldPosition,false);
        this->validPositions = ValidPositions{validPositionH,validPositionV};
        if(validPositionH.size(0) == size[1] * size[3] && validPositionV.size(0) == size[0] * size[2]){
                //std::cout<<"All positions are valid for subblock copy. "<<std::endl;
                this->includesInvalidCorners = false;
        }
        else{
            //std::cout<<"Some positions are invalid for subblock copy. "<<std::endl;
            //std::cout<<"Valid Positions Horizontal: "<<validPositionH.sizes()<<"/"<<size[1] * size[3]<<std::endl;
            //std::cout<<"Valid Positions Vertical: "<<validPositionV.sizes()<<"/"<<size[0] * size[2]<<std::endl;

            this->includesInvalidCorners = true;
            #if FLAT_TRANSFORM == 1
            this->transformSize = {1,1,validPositionV.size(0),validPositionH.size(0)};
            #else 
            std::cerr<<"Pre-Slant Not implemented for non-flat transform!"<<std::endl;
            #endif
        }
    }
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

void Block4D_::copySubblockData(Block4D_& destination, std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset) const{

    for( int i = 0; i < 4; i++){
        destination.lightFieldPosition[i] = this->lightFieldPosition[i] + sourceOffset[i];
    }
    //std::cout<<"first few elements destination early: "<<destination.data.index({0,0,0,at::indexing::Slice(0,3)})<<std::endl;


    std::array<int64_t,4> length = {std::min(subblockLength[0], this->size[0]-sourceOffset[0]),
                                    std::min(subblockLength[1], this->size[1]-sourceOffset[1]),
                                    std::min(subblockLength[2], this->size[2]-sourceOffset[2]),
                                    std::min(subblockLength[3], this->size[3]-sourceOffset[3])};
    
    destination.size = length;
    if(sgtDomain)std::cout<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    if(!destination.includesInvalidCorners){
        destination.transformSize = toSGTCoords(length);
        //std::cout<<"no invalids length: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"no invalids destination.transformSize: "<<destination.transformSize[0]<<"x"<<destination.transformSize[1]<<"x"<<destination.transformSize[2]<<"x"<<destination.transformSize[3]<<std::endl;
    }else{
        destination.transformSize = {1,1,destination.validPositions.valid_positions_v.size(0),destination.validPositions.valid_positions_h.size(0)};
        
    }

    if(this->sgtDomain){
        length = destination.transformSize;
        //std::cout<< destination.validPositions.valid_positions_v.size(0)<<"x"<<destination.validPositions.valid_positions_h.size(0)<<std::endl;
        sourceOffset = toSGTCoords(sourceOffset);
        if(this->includesInvalidCorners){
           
            int offsetReductionV = computePreviousInvalidNumber(this->lightField->preSlantTan,this->lightFieldPosition[2],destination.lightFieldPosition[2],false);
            int offsetReductionH= computePreviousInvalidNumber(this->lightField->preSlantTan,this->lightFieldPosition[3],destination.lightFieldPosition[3],true);
            sourceOffset = {0,0,sourceOffset[2] - offsetReductionV,sourceOffset[3] -offsetReductionH };
            //length = {length[0],length[1],destination.validPositions.valid_positions_v.size(0),destination.validPositions.valid_positions_h.size(0)};
            // if(destination.lightFieldPosition[3] == 192 && destination.lightFieldPosition[2] == 64 && subblockLength[3] == 64){
            //     std::cout<<"originalOffset: "<<sourceOffset[2]+offsetReductionV<<"x"<<sourceOffset[3]+offsetReductionH<<std::endl;
            //     std::cout<<"parent LightField Position: "<<this->lightFieldPosition[2]<<"x"<<this->lightFieldPosition[3]<<std::endl;
            //     std::cout<<"destination lightField Position: "<<destination.lightFieldPosition[2]<<"x"<<destination.lightFieldPosition[3]<<std::endl;
            //     std::cout<<"offsetReduction: "<<offsetReductionV<<"x"<<offsetReductionH<<std::endl;
            //     std::cout<<"sourceOffset: "<<sourceOffset[2]<<"x"<<sourceOffset[3]<<std::endl;
            //     std::cout<<"length: "<<length[2]<<"x"<<length[3]<<std::endl; 
            //     std::cout<<"destination.has invalid corners: "<<destination.includesInvalidCorners<<std::endl;
            // }
            //std::cout<<"destination.data.size(): "<< destination.data.sizes()<<std::endl;
            //std::cout<<"first pixel: "<<destination.data[0][0][][]
        }
    }

    destination.data  = destination.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });
    //std::cout<<"first few elements destination final: "<<destination.data.index({0,0,0,at::indexing::Slice(0,3)})<<std::endl;

       

}
int Block4D_::computePreviousInvalidNumber(double preSlantTan,int parentBlockN, int subblockN,bool isHorizontal) const{
    int angleVariable = isHorizontal ? 1 : 0; // 1 for horizontal, 0 for vertical
    int spaceVariable = isHorizontal ? 3 : 2; // 3 for horizontal, 2 for vertical
    int size_increase = (int)(abs(round(preSlantTan*(this->lightField->data.size(angleVariable)-1))));

    double d = preSlantTan/abs(preSlantTan) * (double)size_increase/((double)this->lightField->data.size(angleVariable)-1);
    double spaceSize = (double)this->lightField->data.size(spaceVariable) - size_increase;
    double a = 1/d;
    int maxAngle = this->lightField->data.size(angleVariable);
    int count_upper = 0;
    int count_lower = 0;
    int count = 0;
    for(int n = parentBlockN; n < subblockN; n++){
        int upper_l_boundary;
        int lower_l_boundary;
        if(d < 0){
            int offset = size_increase;
            //int l_min = static_cast<int>(floor((n + 1.0 - offset) *a)) + 1;
            lower_l_boundary = static_cast<int>(floor((n - offset + 1.0) * a));
            upper_l_boundary = static_cast<int>(ceil((n - offset - spaceSize) * a));
            
        }else{
            //int first_invalid_l = static_cast<int>(ceil((n + 1.0) * a));
            upper_l_boundary = static_cast<int>(ceil((n + 1.0) * a));
            lower_l_boundary = static_cast<int>(ceil((n - spaceSize) * a));
            //int last_invalid_l = static_cast<int>(ceil((n +  * a));
            // if(parentBlockN == 0 && subblockN == 32){
            //     std::cout<<n<<":"<<" "<<first_invalid_l<<" "<<maxAngle - first_invalid_l<<" "<< count<<std::endl;
            // }
        }
        count_upper += std::max(0, maxAngle - upper_l_boundary);
        count_lower += std::max(lower_l_boundary + 1, 0);
        count += std::max(0, maxAngle - upper_l_boundary) + std::max(lower_l_boundary + 1, 0);
        if(parentBlockN >=512&& subblockN < 512 + 17){
            std::cout<<n<<":"<<" upper_l_boundary: "<<upper_l_boundary<<" lower_l_boundary: "<<lower_l_boundary<<" count_upper: "<<std::max(0, maxAngle - upper_l_boundary)<<" count_lower: "<<std::max(lower_l_boundary + 1, 0)<<" total count: "<<std::max(0, maxAngle - upper_l_boundary) + std::max(lower_l_boundary + 1, 0)<<std::endl;
        }

    }
    if(parentBlockN >=512&& subblockN < 512 + 17){
        std::cout<<"preSlantTan: "<<preSlantTan<<std::endl;
        std::cout<<"parentBlockN: "<<parentBlockN<<" subblockN: "<<subblockN<<std::endl;
        std::cout<<"size_increase: "<<size_increase<<std::endl;
        std::cout<<"d: "<<d<<std::endl;
        std::cout<<"a: "<<a<<std::endl;
        std::cout<<"spaceSize: "<<spaceSize<<std::endl;
        std::cout<<"count_upper: "<<count_upper<<" count_lower: "<<count_lower<<" total count: "<<count<<std::endl;
    }

    return count;

}

std::vector<int64_t> Block4D_::copyValidSubblockPositions(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset, bool isHorizontal){
    
    
    
    std::vector<int64_t> validPositions;
    int angleCoord = isHorizontal ? 1 : 0; // 1 for horizontal, 0 for vertical
    int spatialCoord = isHorizontal ? 3 : 2; // 3 for horizontal, 2 for vertical
    auto* data = isHorizontal ? this->validPositions.valid_positions_h.data_ptr<int64_t>() : this->validPositions.valid_positions_v.data_ptr<int64_t>();
    auto* end = isHorizontal ? data + this->validPositions.valid_positions_h.size(0) : data + this->validPositions.valid_positions_v.size(0);
    for(int l = sourceOffset[angleCoord]; l < sourceOffset[angleCoord] + subblockLength[angleCoord]; l++){
        if(l< 0 || l >= this->size[angleCoord]) throw std::runtime_error("Invalid sourceOffset for Block4D_ copy");
        for(int n = sourceOffset[spatialCoord]; n < sourceOffset[spatialCoord] + subblockLength[spatialCoord]; n++){
            if(n< 0 || n >= this->size[spatialCoord]) throw std::runtime_error("Invalid sourceOffset for Block4D_ copy");
            int64_t oldPosition = l * this->size[spatialCoord] + n;
            int64_t position = l * subblockLength[spatialCoord] + n - sourceOffset[spatialCoord];
            
            
            auto it = std::find(data, end, oldPosition);
            if(it != end){
                validPositions.push_back(position);
            }
        }
    }
    //std::cout<<sourceOffset[spatialCoord]<<" "<<sourceOffset[spatialCoord] + subblockLength[spatialCoord]<<std::endl;
    //std::cout<<"Valid Subblock Positions: "<<validPositions.size()<< " max subblock positions:"<< subblockLength[spatialCoord] * subblockLength[angleCoord]<< std::endl;
    //std::cout<<std::endl;
    return validPositions;
}

Block4D_ Block4D_::copySubblock(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset){
    //std::cout<<"Copying subblock of size: "<<subblockLength[0]<<"x"<<subblockLength[1]<<"x"<<subblockLength[2]<<"x"<<subblockLength[3]<<std::endl;
    //std::cout<<"deepCopy.data.device(): "<<this->data.device()<<std::endl;
    if(this->data.size(0) == 1 && this->data.size(1) == 1){
       // std::cout<<" isSGT:" << this->sgtDomain << std::endl;
        //std::cout<<"valid position size:"<<this->validPositions.valid_positions_v.size(0) << std::endl;
    }

    Block4D_ deepCopy = this->clone();
        //std::cout<<"deepCopy.data.device(): "<<deepCopy.data.device()<<std::endl;
    
    if(includesInvalidCorners){
        std::vector<int64_t> validPositionsH = copyValidSubblockPositions(subblockLength, sourceOffset, true);
        std::vector<int64_t> validPositionsV = copyValidSubblockPositions(subblockLength, sourceOffset, false);
        if(deepCopy.sgtDomain){
            std::cout<<"sourceOffset: "<<sourceOffset[0]<<"x"<<sourceOffset[1]<<"x"<<sourceOffset[2]<<"x"<<sourceOffset[3]<<std::endl;
            std::cout<<"subblockLength: "<<subblockLength[0]<<"x"<<subblockLength[1]<<"x"<<subblockLength[2]<<"x"<<subblockLength[3]<<std::endl;
            std::cout<<"validPositionsH.size():"<<validPositionsH.size()<<" validPositionsV.size():"<<validPositionsV.size()<<std::endl;
            std::cout<<"this->validPositions.valid_positions_h.size(0):"<<this->validPositions.valid_positions_h.size(0)<<" this->validPositions.valid_positions_v.size(0):"<<this->validPositions.valid_positions_v.size(0)<<std::endl;
        }
        if(validPositionsH.size() == 0 && validPositionsV.size() == 0){
            //std::cout<<"No valid positions found for subblock copy. Returning empty Block4D_."<<std::endl;
        }
        if(validPositionsH.size() == subblockLength[1] * subblockLength[3] && validPositionsV.size() == subblockLength[0] * subblockLength[2]){
            //std::cout<<"All positions are valid for subblock copy. "<<std::endl;
            deepCopy.includesInvalidCorners = false;
        }else{
            //std::cout<<"Not all positions are valid for subblock copy. "<<std::endl;
            deepCopy.includesInvalidCorners = true;
            at::Tensor validPositionsTensorH = torch::from_blob(validPositionsH.data(), {static_cast<int64_t>(validPositionsH.size())}, torch::kLong).clone();
            at::Tensor validPositionsTensorV = torch::from_blob(validPositionsV.data(), {static_cast<int64_t>(validPositionsV.size())}, torch::kLong).clone();
            deepCopy.validPositions.valid_positions_h = validPositionsTensorH;
            deepCopy.validPositions.valid_positions_v = validPositionsTensorV;
        }


    }
            

    //std::cout<<"Hello!"<<std::endl;
    if(this->data.size(0) == 1 && this->data.size(1) == 1){
        std::cout<<" hasInvalid:" << this->includesInvalidCorners << std::endl;
        std::cout<<"copy has invalids:"<<deepCopy.includesInvalidCorners<< std::endl;
        std::cout<<"first few elements deepCopy: "<<deepCopy.data.index({0,0,0,at::indexing::Slice(0,3)})<<std::endl;

    }

    copySubblockData(deepCopy,subblockLength, sourceOffset);

    //std::cout<<"Goodbye!"<<std::endl;

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

  std::vector<std::array<int64_t,4>> Block4D_::treeOrderedCoefficientPositions(std::array<int64_t,4> size){
    std::vector<std::array<int64_t,4>> positions;
    splitHexaDecaTree(size,{0,0,0,0},positions);
    return positions;
}


at::Tensor Block4D_::fetchBlockGradient(int64_t dimension) const{
    std::array<int64_t,4> trueSize;
    for( int i = 0; i < 4; i++){
        // if(lightField->data.size(i) >= lightFieldPosition[i]){
        //     return at::zeros(size, torch::kDouble);
        // }
        trueSize[i] = std::min(lightField->data.size(i) - lightFieldPosition[i], size[i]) ;
    }
    std::array<int64_t,4> gradientPosition = lightFieldPosition;
    if (lightField->secondHalfGradientsComputed) {
        for (int i = 2; i < 3; i++) {
            gradientPosition[i] -= lightField->secondHalfBias;
        }
    }
    at::Tensor blockGradient = lightField->gradients.index({at::indexing::Slice({gradientPosition[0],gradientPosition[0]+trueSize[0]}),
                                                            at::indexing::Slice({gradientPosition[1],gradientPosition[1]+trueSize[1]}),
                                                            at::indexing::Slice({gradientPosition[2],gradientPosition[2]+trueSize[2]}),
                                                            at::indexing::Slice({gradientPosition[3],gradientPosition[3]+trueSize[3]}),
                                                           dimension});

                                                           write_tensor(this->data.index({at::indexing::Slice(),1,at::indexing::Slice(),2}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/gradient_"+std::to_string(dimension)+".png");
    // std::cout<<"size: "<<size[0]<<" "<<size[1]<<" "<<size[2]<<" "<<size[3]<<std::endl;
    //std::cout<<"lightFieldPosition: "<<lightFieldPosition[0]<<" "<<lightFieldPosition[1]<<" "<<lightFieldPosition[2]<<" "<<lightFieldPosition[3]<<std::endl;
    //std::cout<<"gradientPosition: "<<gradientPosition[0]<<" "<<gradientPosition[1]<<" "<<gradientPosition[2]<<" "<<gradientPosition[3]<<std::endl;
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
    //std::cout<<structureTensor<<std::endl;
    if(structureTensor.sum().item<double>() == 0){
        return {0.0,0.0};
    }
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
void Block4D_::splitHexaDecaTree(std::array<int64_t,4> length,std::array<int64_t,4> position,std::vector<std::array<int64_t,4>> &positions){
    int64_t numElems = length[0]*length[1]*length[2]*length[3];
    //std::cout<<"Position: "<<position[0]<<","<<position[1]<<","<<position[2]<<","<<position[3]<<" Length: "<<length[0]<<","<<length[1]<<","<<length[2]<<","<<length[3]<<std::endl;

    if(numElems == 1){
        positions.push_back(position);
        return;
    }

    std::array<int64_t,4> half_length;
    std::array<int64_t,4> number_of_subdivisions;
    for(int i = 0; i < 4; i++){
        half_length[i] = (length[i] > 1) ? length[i]/2 : 1;
        number_of_subdivisions[i] = (length[i] > 1) ? 2 : 1;
    }

    for(int index_t = 0; index_t < number_of_subdivisions[0]; index_t++) {     
        for(int index_s = 0; index_s < number_of_subdivisions[1]; index_s++) {    
            for(int index_v = 0; index_v < number_of_subdivisions[2]; index_v++) {    
                for(int index_u = 0; index_u < number_of_subdivisions[3]; index_u++) {
                    std::array<int64_t,4> new_position;
                    std::array<int64_t,4> new_length;
                    std::array<int64_t,4> index = {index_t,index_s,index_v,index_u};
                    for(int i = 0; i < 4; i++){
                        new_position[i] = position[i]+index[i]*half_length[i];
                        new_length[i] = (index[i] == 0) ? half_length[i] : (length[i] - half_length[i]);
                    }  
                        //std::cout<<"New Position: "<<new_position[0]<<","<<new_position[1]<<","<<new_position[2]<<","<<new_position[3]<<"New Length: "<<new_length[0]<<","<<new_length[1]<<","<<new_length[2]<<","<<new_length[3]<<std::endl;

                    splitHexaDecaTree(new_length,new_position,positions);
                }
                    
            }
            
        }   
    }         
} 

void Block4D_::quadTreeUnsorting(at::Tensor treeSortedTransform, at::Tensor& sortedTransformCoefficients){
    //std::cout<<treeSortedTransform.sizes()<<std::endl;
    std::vector<std::array<int64_t,4>> treeOrderedPositions = Block4D_::treeOrderedCoefficientPositions({1,1,treeSortedTransform.size(0),treeSortedTransform.size(1)});
        //std::cout<<"2 "<<treeSortedTransform.size(0)<<" "<<treeSortedTransform.size(1)<<std::endl;

    int numElems = treeSortedTransform.size(0)*treeSortedTransform.size(1);
        //std::cout<<"2 numElems = "<<numElems<<std::endl;

    sortedTransformCoefficients = at::zeros({numElems});
        //std::cout<<"2"<<std::endl;

    for(int i = 0; i < treeOrderedPositions.size(); ++i){
        sortedTransformCoefficients[i] = treeSortedTransform.index({treeOrderedPositions[i][2],treeOrderedPositions[i][3]});
    }
    //std::cout<<"sortedVectorWritten"<<std::endl;
}
at::Tensor Block4D_::quadTreeSorting(std::array<int64_t,4> size, at::Tensor sortedTransformCoefficients){
    std::vector<std::array<int64_t,4>> treeOrderedPositions = Block4D_::treeOrderedCoefficientPositions(size);
    at::Tensor orderedFlatTransform = at::zeros(size);
    for(int i = 0; i < treeOrderedPositions.size(); ++i){
        orderedFlatTransform.index({treeOrderedPositions[i][0],treeOrderedPositions[i][1],treeOrderedPositions[i][2],treeOrderedPositions[i][3]}) =sortedTransformCoefficients[i];
    }
    return orderedFlatTransform.squeeze();
}

at::Tensor Block4D_::blockAsSecondModelOrderedVector(at::Tensor flatBlock, SgtSideInfo secondModel, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV){
    at::Tensor sortModelCovMatH = this->calcModelCovMatrix(secondModel,true);
    at::Tensor sortModelCovMatV = this->calcModelCovMatrix(secondModel,false);
    at::Tensor sortCoeffsH = at::diag(at::mm(at::mm(sgtMatrixH.t(),sortModelCovMatH),  sgtMatrixH)).unsqueeze(0);
    at::Tensor sortCoeffsV = at::diag(at::mm(at::mm(sgtMatrixV.t(),sortModelCovMatV),  sgtMatrixV)).unsqueeze(0);
    sortCoeffsH = sortCoeffsH/sortCoeffsH.abs().max();
    sortCoeffsV = sortCoeffsV/sortCoeffsV.abs().max();
    //saveTensorAsMatlabScript(at::mm(sortCoeffsV.t(),sortCoeffsH),"sortCoeffProportion");
    //saveTensorAsMatlabScript(flatBlock,"zzBlock");


    at::Tensor coefficientProportion = at::mm(sortCoeffsV.t(),sortCoeffsH).flatten();
    auto [a,indEigOrder] = at::sort(coefficientProportion,0,true);
    at::Tensor sortedFlatTransform = flatBlock.flatten().index({indEigOrder});
    //this->ssi.print();
    //secondModel.print();
    //std::cout<<sortCoeffsH.sizes()<<std::endl;
    // std::cout<<sortCoeffsH.index({0,at::indexing::Slice(0,6)}).unsqueeze(0)<<std::endl;
    // std::cout<<sortCoeffsV.index({0,at::indexing::Slice(0,6)}).unsqueeze(0)<<std::endl;
    // std::cout<<flatBlock.flatten().index({at::indexing::Slice(0,6)}).unsqueeze(0)<<std::endl;
    return sortedFlatTransform;
}
void Block4D_::reOrderSGTMatrices(at::Tensor& sgtMatrixH, at::Tensor& sgtMatrixV, SgtSideInfo secondModel){
    at::Tensor sortModelCovMatH = this->calcModelCovMatrix(secondModel,true);
    at::Tensor sortModelCovMatV = this->calcModelCovMatrix(secondModel,false);
    at::Tensor sortCoeffsH = at::diag(at::mm(at::mm(sgtMatrixH.t(),sortModelCovMatH),  sgtMatrixH));
    at::Tensor sortCoeffsV = at::diag(at::mm(at::mm(sgtMatrixV.t(),sortModelCovMatV),  sgtMatrixV));

    auto [coeffsH,indH] = at::sort(sortCoeffsH,0,true);
    auto [coeffsV,indV] = at::sort(sortCoeffsV,0,true);
    
    sgtMatrixH = sgtMatrixH.index({at::indexing::Slice(),indH});
    sgtMatrixV = sgtMatrixV.index({at::indexing::Slice(),indV});


}
// void Block4D_::reReOrderSGTMatrices(at::Tensor& sgtMatrixH, at::Tensor& sgtMatrixV, SgtSideInfo secondModel){
//     at::Tensor sortModelCovMatH = this->calcModelCovMatrix(secondModel,true);
//     at::Tensor sortModelCovMatV = this->calcModelCovMatrix(secondModel,false);
//     at::Tensor sortCoeffsH = at::diag(at::mm(at::mm(sgtMatrixH.t(),sortModelCovMatH),  sgtMatrixH));
//     at::Tensor sortCoeffsV = at::diag(at::mm(at::mm(sgtMatrixV.t(),sortModelCovMatV),  sgtMatrixV));

//     auto [coeffsH,indH] = at::sort(sortCoeffsH,0,true);
//     auto [coeffsV,indV] = at::sort(sortCoeffsV,0,true);
    
//     sgtMatrixH.index({at::indexing::Slice(),indH}) = sgtMatrixH;
//     sgtMatrixV.index({at::indexing::Slice(),indV}) = sgtMatrixV;
// }
at::Tensor Block4D_::secondModelOrderedBlock2SGT(at::Tensor flatTransform, SgtSideInfo secondModel, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV){
    at::Tensor sortedTransformCoefficients;
    quadTreeUnsorting(flatTransform,sortedTransformCoefficients);
    
    at::Tensor sortModelCovMatH = this->calcModelCovMatrix(secondModel,true);
    at::Tensor sortModelCovMatV = this->calcModelCovMatrix(secondModel,false);
    at::Tensor sortCoeffsH = at::diag(at::mm(at::mm(sgtMatrixH.t(),sortModelCovMatH),  sgtMatrixH)).unsqueeze(0);
    at::Tensor sortCoeffsV = at::diag(at::mm(at::mm(sgtMatrixV.t(),sortModelCovMatV),  sgtMatrixV)).unsqueeze(0);
   // saveTensorAsMatlabScript(at::mm(sortCoeffsV.t(),sortCoeffsH),"sortCoeffProportion");


    at::Tensor coefficientProportion = at::mm(sortCoeffsV.t(),sortCoeffsH).flatten();
    auto [a,indEigOrder] = at::sort(coefficientProportion,0,true);

    at::Tensor indices = indEigOrder;

    at::Tensor sgtCoefficients = at::zeros(sortedTransformCoefficients.sizes());
    sgtCoefficients = sgtCoefficients.index_put({indices},sortedTransformCoefficients);
    flatTransform = sgtCoefficients.reshape(flatTransform.sizes()).to(at::kDouble);
    
    
    return flatTransform;
}


at::Tensor Block4D_::blockAsEigenOrderedVector(at::Tensor flatBlock, at::Tensor eigenValuesH, at::Tensor eigenValuesV){
    eigenValuesH = eigenValuesH.unsqueeze(0);
    eigenValuesV = eigenValuesV.unsqueeze(0);
    at::Tensor coefficientProportion = at::mm(eigenValuesV.t(),eigenValuesH).flatten();
    auto [a,indEigOrder] = at::sort(coefficientProportion,0,true);
    //saveTensorAsMatlabScript(at::mm(eigenValuesV.t(),eigenValuesH),"eigenCoeffProportion");
    at::Tensor sortedFlatTransform = flatBlock.flatten().index({indEigOrder});
    return sortedFlatTransform;
}
void Block4D_::sgtTransform(double scale){
    //std::cout<<"SGT transform"<<std::endl;
    
    //ssi.print();


    // write_tensor(this->data.index({at::indexing::Slice(),4,at::indexing::Slice(),16}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/EPI-b4transform.png");
    //std::cout<<"In Block4D_::sgtTransform"<<std::endl;
    //std::cout<<"Valid Positions H 42: "<<this->validPositions.valid_positions_h[42].item()<<std::endl;
    //if(this->includesInvalidCorners){
        // if (ssi.getAngleH() == 0 && ssi.getAngleV() == 0){
        //    ssi.setAngleH(2*SgtSideInfo::PRECISION_ANGLE);
        //    ssi.setAngleV(2*SgtSideInfo::PRECISION_ANGLE);
        // }
    //}
    at::Tensor modelCovMatH = this->calcModelCovMatrix(ssi,true);
    //std::cout<<"Valid Positions H 42 AFTER: "<<this->validPositions.valid_positions_h[42].item()<<std::endl;

    at::Tensor modelCovMatV = this->calcModelCovMatrix(ssi,false);

    //std::cout<<"Includes Invalid Corners: "<<this->includesInvalidCorners<<std::endl;

    //std::cout<<"Model Covariance Matrices: "<<modelCovMatH.sizes()<<" "<<modelCovMatV.sizes()<<std::endl;

    
    at::Tensor eigValsH,eigValsV;
    at::Tensor flatBlock = scale * getFlatBlock();
    //std::cout<<"Flat Block Size: "<<flatBlock.sizes()<<std::endl;
    //ssi.print();
    //std::cout<<this->includesInvalidCorners<<std::endl;
    //std::cout<<modelCovMatH.sizes()<<" "<<modelCovMatV.sizes()<<std::endl;
    // if(modelCovMatH.size(0) == 0 || modelCovMatH.size(0) == 0){
    //     std::cout<<"Warning: Model Cov Mat Matrix is empty!"<<std::endl;
    //     return;
    // }
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true,eigValsH);
    at::Tensor sgtMatrixV =   getSgtTransformMatrix(modelCovMatV,false,eigValsV);

    //saveTensorAsMatlabScript(sgtMatrixH,"sgtMatrixH_n47");
    
    
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

at::Tensor Block4D_::getZigZagIndexes(std::array<int64_t,2> size){
    int64_t size2 = size[0]*size[1];

    at::Tensor indMatrix = at::range(0,size[0]*size[1]-1,1).reshape({size[0],size[1]});

    torch::TensorOptions options = torch::TensorOptions();
    at::Tensor zigZag = at::empty({0},options.dtype(at::kLong));

    for(int i = -(size[0] - 1); i < size[1] ; ++i){

        at::Tensor diag = at::diag(indMatrix.fliplr(),i).to(at::kLong);
        if(i%2 == 0){
            diag = diag.flip(0);
        }
        zigZag =at::cat({zigZag,diag},0);
    }
    zigZag = zigZag.flip(0);
    //std::cout<<zigZag.index({at::indexing::Slice(0,10)})<<std::endl;
    return zigZag;
}

at::Tensor Block4D_::getOrdered2DFromZigZagCoeffs(at::Tensor coeffs){
    std::array<int64_t,2> size = {data.size(0)*data.size(2),data.size(1)*data.size(3)};
    at::Tensor flatBlock = coeffs.reshape(size);
    //std::cout<<"data type = "<<flatBlock.dtype()<<std::endl;
    return flatBlock;
}
at::Tensor Block4D_::flatBlockFrom4DTensor(at::Tensor block){
    at::Tensor coeffs = diagonalOrder4DSampling(block);
    //std::cout<<"coeffs: "<<coeffs.index({at::indexing::Slice(0,10)})<<std::endl;
    coeffs = coeffs.index({getReverseZigZagIndexes({block.size(0)*block.size(2),block.size(1)*block.size(3)})});
    //std::cout<<"original coeffs: "<<coeffs.index({at::indexing::Slice(0,10)})<<std::endl;
    return getOrdered2DFromZigZagCoeffs(coeffs);
}
at::Tensor Block4D_::sgtFrom2DCoefficients(at::Tensor coefficients){
    at::Tensor indexes = getZigZagIndexes({coefficients.size(0),coefficients.size(1)});
    at::Tensor sgtCoefficients = coefficients.flatten().index({indexes});
    //std::cout<<" SGT Coeffs: "<<std::endl<<sgtCoefficients.index({at::indexing::Slice(0,10)})<<std::endl;
    //std::cout<<" SGT Orig Coeffs: "<<std::endl<<coefficients.flatten().index({at::indexing::Slice(0,10)})<<std::endl;

    return diagonalOrder4DBlock(sgtCoefficients);
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

    auto [peaks,w_k,w_m] = findLocalMaxima(block_dft,0.4);
    #if DEBUG == 1
    
    
    if(index == 70){
        //write_tensor(block_dft,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/frequency.png");
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
    //    if(index == 70){
    //     std::cout<<"peaks:"<<std::endl<<peaks<<std::endl;
    //     std::cout<<"w_k:"<<std::endl<<w_k_unq<<std::endl;
    //     std::cout<<"w_m:"<<std::endl<<w_m_unq<<std::endl;
    //     std::cout<<"w_th:"<<std::endl<<w_th<<std::endl;
    // }
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
    //write_tensor(ordering,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/anOrdering4.png");
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
    std::cout<<"ordinalH Size = " << ordinalH.sizes()<<std::endl;
    ordinalH = ordinalH.expand({ordinalFrequenciesV.size(1), ordinalFrequenciesH.size(1), 2});
    ordinalV = ordinalV.expand({ordinalFrequenciesV.size(1), ordinalFrequenciesH.size(1), 2});
    at::Tensor fullOrdinalFrequencies = torch::cat({ordinalV, ordinalH}, /*dim=*/2);
    at::Tensor indexes = torch::tensor({0,2,1,3},torch::kLong);
    fullOrdinalFrequencies = fullOrdinalFrequencies.index({at::indexing::Slice(),at::indexing::Slice(),indexes});
    fullOrdinalFrequencies = fullOrdinalFrequencies.flatten(0,1).t();
#if DEBUG == 1
    std::cout<<"Freq Sizes = "<<basisFrequenciesH.sizes()<<std::endl;
    std::cout<<"Ordinal Sizes = "<<ordinalFrequenciesH.sizes()<<std::endl;
    at::Tensor freqH = basisFrequenciesH.t().permute({1, 0}).unsqueeze(0);
    at::Tensor freqV = basisFrequenciesV.t().permute({1, 0}).unsqueeze(1);
    std::cout<<"ordinalH Size = " << ordinalH.sizes()<<std::endl;
    freqH = freqH.expand({basisFrequenciesV.t().size(1),basisFrequenciesV.t().size(1),2});
    freqV = freqV.expand({basisFrequenciesV.t().size(1),basisFrequenciesV.t().size(1),2});
    frequencies = torch::cat({freqV,freqH},2);
    frequencies = frequencies.index({at::indexing::Slice(),at::indexing::Slice(),indexes});
    frequencies = frequencies.flatten(0,1).t().round(1);
 
#endif 
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

        //write_tensor(vizTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/Frequency4D_"+std::to_string(j)+".png");
        // write_tensor(vizTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/Frequency4D_"+std::to_string(j)+".png",{0,vizTransform.max().item<double>()});
    }
}

at::Tensor Block4D_::orderCoefficientsByFrequency(const at::Tensor& coefficientBlock,const at::Tensor& sgtMatrixH,const at::Tensor& sgtMatrixV) const{
    //std::cout<<"Ordering Coefficients"<<std::endl;
    std::cout<<"MAX COEFFICIENT BIT SIZE: "<<log2(coefficientBlock.max().item<double>())<<std::endl;
    at::Tensor flatBlockDouble = coefficientBlock.to(at::kDouble);
    std::cout<<log2(1+(flatBlockDouble[70][0].abs().item<double>()))<<" "<<log2(1+(flatBlockDouble[0][70].abs().item<double>()))<<std::endl;
    double angleH = this->ssi.getAngleH();
    double angleV = this->ssi.getAngleV();
    at::Tensor basisFrequenciesH = getBasisFrequencies(sgtMatrixH,angleH);
    at::Tensor basisFrequenciesV = getBasisFrequencies(sgtMatrixV,angleV);
    std::cout<<"basisFrequencies: "<<basisFrequenciesH[70];
    //std::cout<<"Index of Max Frequency: "<<basisFrequenciesV.index({at::indexing::Slice(),0}).argmax().item<int64_t>()<<std::endl;
    at::Tensor basisFrequenciesViewTH = basisFrequenciesH.index({at::indexing::Slice(),0}).squeeze().repeat({basisFrequenciesV.size(0),1});
    at::Tensor basisFrequenciesViewTV = basisFrequenciesV.index({at::indexing::Slice(),0}).t().repeat({basisFrequenciesH.size(0),1}).t();
    //write_tensor(basisFrequenciesViewTH,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/BasisFrequenciesH.png");
    //write_tensor(basisFrequenciesViewTV,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/BasisFrequenciesV.png");

    at::Tensor ordinalFrequenciesH = getOrdinalFrequencies(basisFrequenciesH);
    at::Tensor ordinalFrequenciesV = getOrdinalFrequencies(basisFrequenciesV);

    std::cout<<"LOOK HERE: "<<ordinalFrequenciesH.t()[70]<<std::endl;
    //std::cout<<"RelevantH: "<<ordinalFrequenciesH.t()[0]<<std::endl;
 
    
        std::cout<<"Acquired Frquency Order of Basis"<<std::endl;
    at::Tensor frequencies;
    
    at::Tensor ordinalFrequenciesFull = getFullOrdinalFrequency(basisFrequenciesH, basisFrequenciesV,ordinalFrequenciesH,ordinalFrequenciesV, frequencies);
        std::cout<<"Acquired Full Coefficient Order"<<std::endl;

        #if DEBUG == 1
    //std::cout<<"Acquired Basis Frequencies"<<std::endl;
    //std::cout<<"RelevantH: "<<basisFrequenciesH[0]<<std::endl;
    //std::cout<<"RelevantV: "<<basisFrequenciesV[96]<<std::endl;
    //std::cout<<"RelevantV: "<<ordinalFrequenciesV.t()[96]<<std::endl;
    view4DFrequencies(ordinalFrequenciesFull, frequencies);
#endif   
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
    //write_tensor(log(1+(sortedSgt * sortedSgt)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/aFrequencyQuadTreeSgT.png");
    //write_tensor(log(1+(sgtCoefficients * sgtCoefficients)),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/anUnsortedSgT.png");

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
at::Tensor Block4D_::diagonalOrder4DSampling(at::Tensor tensor){
   int maxSum = 0;
   int nElems = 1;
   //std::array<int64_t,4> size = {tensor.size(0),tensor.size(1),tensor.size(2),tensor.size(3)};
    for(int i = 0; i < tensor.ndimension();i++){
        maxSum += tensor.size(i);
        nElems *= tensor.size(i);
        //std::cout<<i<<": "<<tensor.size(i)<<std::endl;
    }

    //std::cout<<"Max Sum = "<<maxSum<<" nElems = "<<nElems<<std::endl;
    at::Tensor coefficients = at::zeros({nElems},at::kDouble);
    int i = 0;
    for(int sum = 0; sum < maxSum; sum ++){
        for(int n = 0;n<tensor.size(0);n++){
            for(int m = 0;m<tensor.size(1);m++){
                int currentSum = m+n;
                if(sum == currentSum){
                    coefficients[i] = tensor[m][n] ;
                    i++;
                    //std::cout<<"     \r i = "<<i<< " Sum = "<<sum<<" maxSum = "<<maxSum<<std::endl;
                }
            }
        }
    }
    if( i != coefficients.size(0)){
        std::cerr<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<std::endl;
    }
    return coefficients;
}
at::Tensor Block4D_::diagonalOrder4DBlock(at::Tensor coefficients){
    int maxSum = 0;
    std::array<int64_t,4> size = {data.size(0),data.size(1),data.size(2),data.size(3)};
    for(int i = 0; i < 4; i++){
        maxSum += size[i];
    }
    //std::cout<<"Max Sum = "<<maxSum<<std::endl;
    at::Tensor tensor4D = at::zeros({size[0],size[1],size[2],size[3]});
    int i = 0;
    for(int sum = 0;sum<maxSum;sum++){
        for(int n = 0;n<size[0];n++){
            for(int m = 0;m<size[1];m++){
                for(int k = 0;k<size[3];k++){
                    for(int l = 0;l<size[2];l++){
                        int currentSum = m+n+k+l;
                        if(sum == currentSum){
                            tensor4D[m][n][k][l] = coefficients[i];
                            i++;
                        }
                    }
                }
            }
        }
    }
    if( i != coefficients.size(0)){
        std::cerr<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<std::endl;
    }
    return tensor4D;
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
    //std::cout<<covMat.sizes()<<std::endl;
    try{
        auto [L, Q] = torch::linalg::eigh(covMat, "U");
        eigVals = L.flip({-1});
        return Q.flip({-1});
    }
    catch (const c10::Error& e) {
        std::cerr << "Error computing eigen decomposition: " << e.what() << std::endl;
        std::cout<<"Covariance Matrix Size: "<<covMat.sizes()<<std::endl; 
        exit(-1);
    }
    // std::ofstream eigenValues;
    // eigenValues.open("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/eigenValues.m");
    // eigenValues<<"eigV = [";
    // for(int i = 0; i < eigVals.size(-1); i++){
    //     eigenValues<<eigVals[i].item()<<" ";
    // }
    // eigenValues<<"];"<<std::endl;

    // std::ofstream stuff;
    // stuff.open("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/covMat.m");
    // stuff<<"R_cpp = [";
    // for(int n = 0; n < covMat.size(0); n++){
    //     if(n != 0) stuff<<";"<<std::endl;
    //     for(int m = 0; m < covMat.size(1); m++){
    //         if (m != 0) stuff<<",";
    //         stuff<<covMat[n][m].item();
    //     }
    // }
    // stuff<<"];"<<std::endl;
     // flip such that coefficients are in DESCENDING order
}

at::Tensor Block4D_::getOrderH(){
    return orderH;
}
at::Tensor Block4D_::getOrderV(){
    return orderV;
}

at::Tensor Block4D_::sgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV,const at::Tensor& eigValsH,const at::Tensor& eigValsV) {
    at::Tensor block = flatBlock.to(at::kDouble);
    // for(int i = 0; i < eigValsH.size(0); i++){
    //     if (eigValsH[i].item<double>() < 0) {
    //         std::cerr<<"eigValsH < 0"<<std::endl;
    //        // exit(-2);
    //     }
    //     if (eigValsV[i].item<double>() < 0) {
    //         std::cerr<<"eigValsV < 0"<<std::endl;
    //         //exit(-2);
    //     }
    // }

    at::Tensor transform = at::mm(at::mm(sgtMatrixV.t(), block), sgtMatrixH);
    
    return transform;
}

at::Tensor Block4D_::isgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) {
    //std::cout<<"sgtMatrixV: "<<sgtMatrixV.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;
    at::Tensor transform = at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
    return transform;
}


at::Tensor Block4D_::squareTransform(at::Tensor transform, at::Tensor eigenValuesH, at::Tensor eigenValuesV){
    eigenValuesV = eigenValuesV.unsqueeze(1);
    eigenValuesH = eigenValuesH.unsqueeze(0);
    at::Tensor newTransform = at::zeros(transform.sizes(),transform.dtype());
    at::Tensor eigenValueMat = at::mm(eigenValuesV,eigenValuesH).flatten();

    auto [sorted, indices] = torch::sort(eigenValueMat,0,true);
    at::Tensor transformVec = transform.flatten();
    transformVec.index({indices}) = transformVec;
    int i = 0;
    
    for(int sum = 0; sum < std::max(transform.size(0),transform.size(1)); sum++){
        for(int n = 0; n < transform.size(0); n++){
            for(int m = 0; m < transform.size(1); m++){
                if(std::max(n,m) == sum){

                    newTransform[n][m] = transformVec[i++];
                }
            }
        }  
    }

    return newTransform; 
}
at::Tensor Block4D_::getReverseZigZagIndexes(std::array<int64_t,2> size){
    using namespace std;
    int64_t maxSize = max(size[0],size[1]);
    int64_t minSize = min(size[0],size[1]);
    at::Tensor cumm = at::zeros(size,at::kLong);
    int bias = 0;
    for (int64_t i = -size[0] + 1; i<size[1]; ++i){
        int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
        at::Tensor diag = at::zeros(diagSize,at::kLong);
        for (int64_t j = 0; j<diagSize; ++j){
            int64_t m = min(i+size[0] -1,size[0] -1) - j;
            int64_t n = j+max((int64_t)0,i);
            if(i%2 == 0){
                cumm[m][n] = bias + (diagSize - 1 - j);
            }else{
                cumm[m][n] = bias + j;
            }
        }
        bias = bias + diagSize;
    }
    //cout<<"CUMM: "<<endl<<cumm.flatten().index({at::indexing::Slice(0,7)})<<endl;
    return cumm.flatten();
}
at::Tensor Block4D_::squareSorting(std::array<int64_t,4> size, at::Tensor coefficients){
    using namespace std;
    int maxSum = 0;
    for(int i = 0; i < 4; i++){
        if(size[i] > maxSum){
            maxSum = size[i];
        }
    }
    maxSum -= 1;
    at::Tensor triangle = at::zeros(size,at::kDouble);
    int i = 0;
    for (int summ = 0; summ < maxSum; summ++){
        for (int k = 0; k< size[0]; k++){
            for( int l = 0; l < size[1]; l++){
                for (int m = 0; m < size[2]; m++){
                    for (int n = 0; n < size[3]; n++){
                        int currentSum = max(max(max(m,n),k),l);
                        if(summ == currentSum){
                            triangle[k][l][m][n] = coefficients[i];
                            i++;
                        }
                    }
                }
            }
        }
    } 
    return triangle.squeeze();
}
at::Tensor Block4D_::triangleSorting(std::array<int64_t,4> size, at::Tensor coefficients){
    using namespace std;
    int maxSum = std::accumulate(size.begin(),size.end(),0) - 4;
    at::Tensor triangle = at::zeros(size,at::kDouble);
    int i = 0;
    for (int summ = 0; summ < maxSum; summ++){
        for (int k = 0; k< size[0]; k++){
            for( int l = 0; l < size[1]; l++){
                for (int m = 0; m < size[2]; m++){
                    for (int n = 0; n < size[3]; n++){
                        int currentSum = m+n+k+l;
                        if(summ == currentSum){
                            triangle[k][l][m][n] = coefficients[i];
                            i++;
                        }
                    }
                }
            }
        }
    } 
    return triangle.squeeze();
}
at::Tensor Block4D_::getTriangleIndexes(std::array<int64_t,2> size){
    using namespace std;
    int64_t maxSize = max(size[0],size[1]);
    int64_t minSize = min(size[0],size[1]);
    at::Tensor cumm = at::zeros(size,at::kLong);
    int bias = 0;
    for (int64_t i = -size[0] + 1; i<size[1]; ++i){
        int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
        at::Tensor diag = at::zeros(diagSize,at::kLong);
        for (int64_t j = 0; j<diagSize; ++j){
            int64_t m = min(i+size[0] -1,size[0] -1) - j;
            int64_t n = j+max((int64_t)0,i);
            cumm[m][n] = bias + j;
        }
        bias = bias + diagSize;
    }
    return cumm.flatten();
}

at::Tensor Block4D_::zigZagTransformMatrix(at::Tensor transform,bool isHorizontal){
    std::array<int64_t,2> size;
    if(isHorizontal){
         size = {data.size(1),data.size(3)};
    }else{
        size = {data.size(0),data.size(2)};
    }
    at::Tensor zigZagIndices = getZigZagIndexes(size);
    return transform.index({at::indexing::Slice(),zigZagIndices});
}
at::Tensor Block4D_::triangleTransformMatrix(at::Tensor transform,bool isHorizontal){
    std::array<int64_t,2> size;
    if(isHorizontal){
         size = {data.size(1),data.size(3)};
    }else{
        size = {data.size(0),data.size(2)};
    }
    at::Tensor triangleIndices = getTriangleIndexes(size);
    return transform.index({at::indexing::Slice(),triangleIndices});
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
    at::Tensor padded_block = torch::zeros({size[0]*size[2],size[1]*size[3]}).to(at::kInt);
    std::cout<<"Padded Block Size: "<<padded_block.sizes()<<std::endl;
    std::cout<<"Flat Block Size: "<<flatBlock.sizes()<<std::endl;
    auto a = at::meshgrid({validPositions.valid_positions_v, validPositions.valid_positions_h},"ij");
    std::cout<<"a[0] size: "<<a[0].sizes()<<" "<<a[1].sizes()<<std::endl;
    std::cout<<"a[0]: "<<a[0].index({at::indexing::Slice(0,10),at::indexing::Slice(0,10)})<<std::endl;
    std::cout<<"a[1]: "<<a[1].index({at::indexing::Slice(0,10),at::indexing::Slice(0,10)})<<std::endl;
    padded_block = padded_block.index_put({a[0],a[1]},flatBlock);
    std::cout<<"Padded Block Size: "<<padded_block.sizes()<<std::endl;

    std::array<int64_t,4> sizeShifted = {size[1],size[3],size[0],size[2]};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = padded_block.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});
    std::cout<<"size unflattened: "<< unflattened_block.size(0)<<" "<<unflattened_block.size(1)<<" "<<unflattened_block.size(2)<<" "<<unflattened_block.size(3)<<std::endl;
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


double calcMonotonyCost(at::Tensor current, at::Tensor previous, at::Tensor& positiveOnly){
    at::Tensor diff = current - previous;
    double avg = diff.abs().mean().item<double>();
    at::Tensor positive = diff > 0.0*avg;
    //std::cout<<"BunchONumbers:"<<positive<<std::endl;
    positiveOnly = at::clamp_min(diff, 0.0);
    double cost = positive.sum().item<double>();
    //at::Tensor fullCost = (negativeOnly*negativeOnly) + positiveOnly;
    //double monotonyCost = -fullCost.sum().item<double>();
    return cost;
} 

at::Tensor Block4D_::orderSGTByMonotony( at::Tensor epiTransform,at::Tensor sgtTransform,at::Tensor& order){

    at::Tensor energy = epiTransform * epiTransform;
    at::Tensor newOrderTransform = sgtTransform.clone();
    //write_tensor(log(1+energy),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/old_energy.png");
    //write_tensor(newOrderTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/old_transform.png");
    int j = 0;
    order = at::arange(0,sgtTransform.size(1));
    while(j < epiTransform.size(1)-1){
        for(int i = 0; i < epiTransform.size(1)-1-j;i++){
            at::Tensor previous = energy.index({i,at::indexing::Slice()}).clone();
            at::Tensor current = energy.index({i+1,at::indexing::Slice()}).clone();
            at::Tensor positive;
            double cost = calcMonotonyCost(current, previous,positive);
            if(cost > epiTransform.size(1)/2){
                at::Tensor currentSigned = newOrderTransform.index({i+1,at::indexing::Slice()}).clone();
                at::Tensor previousSigned = newOrderTransform.index({i,at::indexing::Slice()}).clone();
                energy.index({i,at::indexing::Slice()}) = current;
                energy.index({i+1,at::indexing::Slice()}) = previous;
                newOrderTransform.index({i,at::indexing::Slice()}) = currentSigned;
                newOrderTransform.index({i+1,at::indexing::Slice()}) = previousSigned;
                int aux = order[i].item<int>();    
                order[i] = order[i+1];
                order[i+1] = aux;
            }   
        }
        j++;     
    }
    //write_tensor(log(1+energy),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/new_energy.png");
    //write_tensor(newOrderTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/new_transform.png");
    return newOrderTransform;
}

double optimizeMonotony(at::Tensor epiTransforms){
    int j = 0;
    double swapCount = 0;
    double totalSwapCost = 0;
    //epiTransforms = epiTransforms.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6)});
    //std::cout<<epiTransforms<<std::endl;
    //std::cout<<epiTransforms.sizes()<<std::endl;
    //saveTensorAsMatlabScript(epiTransforms,"epiTransformsBefore");

    
    while(j < epiTransforms.size(1)-1){
        for(int i = 0; i < epiTransforms.size(1)-1-j;i++){
            at::Tensor previous = epiTransforms.index({i,at::indexing::Slice()}).clone();
            
            at::Tensor current = epiTransforms.index({i+1,at::indexing::Slice()}).clone();
            //if(i == 0) std::cout<<previous.index({at::indexing::Slice(0,6)})<<std::endl;
            //if(i == 0) std::cout<<(current).index({at::indexing::Slice(0,6)})<<std::endl;
            at::Tensor positive;
            double cost = calcMonotonyCost(current, previous,positive);
            double swapCost = positive.sum().item<double>();

            //if(i == 43-j && j < 44)std::cout<<positiveOnly<<std::endl;
            //if(i == 43-j && j < 44)std::cout<<" i = "<<i<<" j = "<<j<<" cost = "<<cost<<std::endl;

            //std::cout<<j<<","<<i<<" cost:"<<cost<<std::endl;
            if(cost > epiTransforms.size(1)/2){
                epiTransforms.index({i,at::indexing::Slice()}) = current;
                epiTransforms.index({i+1,at::indexing::Slice()}) = previous;
                

                totalSwapCost+=swapCost;
                swapCount++;
                //if(j == 0) std::cout<<i<<":"<<swapCount<<std::endl;
            }            
        }
        //std::cout<<j<<" : "<<swapCount<<std::endl;

        j++;
        
    }
    //saveTensorAsMatlabScript(epiTransforms,"epiTransformsAfter");

    //std::cout<<epiTransforms<<std::endl;
    //std::cout<<"SWAP COUNT:"<<swapCount<<std::endl;
    return totalSwapCost;
}

void SgtSideInfo::estimateRhosFromMonotony(Block4D_ block){
    at::Tensor flatBlock = block.getFlatBlock().to(at::kDouble);
    //std::cout<<flatBlock.dtype()<<std::endl;

    // at::Tensor covFunH = Block4D_::normalizeCov(block.covFun(true));
    // at::Tensor covFunV = Block4D_::normalizeCov(block.covFun(false));
    // int64_t k_center = covFunH.size(0)/2;
    // int64_t m_center = covFunH.size(1)/2;     
    // int64_t l_center = covFunV.size(0)/2;
    // int64_t n_center = covFunV.size(1)/2;     
    // setRhoU(covFunH[k_center][m_center+1].item<double>());
    // setRhoV(covFunV[l_center][n_center+1].item<double>());
    // setRhoT(0.999);

    std::vector<double> rhoAngleVector = {0.5,0.9,0.99,0.999,0.9999,0.99999};
    //std::vector<double> rhoAngleVector = {0.999};
    double minimumCost = 1.7976931348623157E+308;
    double chosenRhoS = 0.999;
    for(int i = 0; i < rhoAngleVector.size();i++){
        setRhoS(rhoAngleVector[i]);
        at::Tensor modelCovMatH = block.calcModelCovMatrix(*this,true);
        at::Tensor eigValsH;
        at::Tensor sgtMatrixH = block.getSgtTransformMatrix(modelCovMatH,true,eigValsH);
        at::Tensor epiTransformH = at::mm(flatBlock,sgtMatrixH).t();
        double cost = optimizeMonotony(epiTransformH*epiTransformH);
        //std::cout<<rhoAngleVector[i]<<":"<<cost<<std::endl;
        if(cost < minimumCost){
            chosenRhoS = rhoAngleVector[i];
            minimumCost = cost;
        }        
    }
    setRhoS(chosenRhoS);

    minimumCost = 1.7976931348623157E+308;
    double chosenRhoT = 0.999;
    for(int i = 0; i < rhoAngleVector.size();i++){
        setRhoT(rhoAngleVector[i]);
        at::Tensor modelCovMatV = block.calcModelCovMatrix(*this,false);
        at::Tensor eigValsV;
        at::Tensor sgtMatrixV = block.getSgtTransformMatrix(modelCovMatV,false,eigValsV);
        at::Tensor epiTransformV = at::mm(sgtMatrixV.t(),flatBlock);
        double cost = optimizeMonotony(epiTransformV*epiTransformV);
        //std::cout<<rhoAngleVector[i]<<":"<<cost<<std::endl;

        if(cost < minimumCost){
            chosenRhoT = rhoAngleVector[i];
            minimumCost = cost;
        }        
    }
    setRhoT(chosenRhoT);
}


void SgtSideInfo::estimateAngleFromMonotony(Block4D_ block){
    at::Tensor flatBlock = block.getFlatBlock().to(at::kDouble);
    //std::cout<<flatBlock.dtype()<<std::endl;
    std::vector<double> monotonyCostHVec;
    std::vector<double> monotonyCostVVec;
    std::vector<double> thetaVec;

    double minimumCost = 1.7976931348623157E+308;
    double chosenAngleH = 0;
    double chosenAngleV = 0;


    double minimumCostH = 1.7976931348623157E+308;
    double minimumCostV = 1.7976931348623157E+308;
    for(double theta = this->angleRange[0];theta<=this->angleRange[1];theta+=PRECISION_ANGLE){
        setAngleH(theta);
        setAngleV(theta);
        at::Tensor modelCovMatV = block.calcModelCovMatrix(*this,false);
        at::Tensor modelCovMatH = block.calcModelCovMatrix(*this,true);

        at::Tensor eigValsV;
        at::Tensor eigValsH;

        at::Tensor sgtMatrixV = block.getSgtTransformMatrix(modelCovMatV,false,eigValsV);
        at::Tensor sgtMatrixH = block.getSgtTransformMatrix(modelCovMatH,true,eigValsH);

        at::Tensor epiTransformH = at::mm(flatBlock,sgtMatrixH).t();
        at::Tensor epiTransformV = at::mm(sgtMatrixV.t(),flatBlock);
        double costV = optimizeMonotony(epiTransformV*epiTransformV);
        double costH = optimizeMonotony(epiTransformH*epiTransformH);
        monotonyCostHVec.push_back(costH);
        monotonyCostVVec.push_back(costV);
        thetaVec.push_back(theta);
        //print();
        //std::cout<<theta<<":"<<costH<<std::endl;
        //write_tensor(epiTransformH*epiTransformH,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/epiAngleImages/"+std::to_string((int)theta)+".png");

        if(costH < minimumCostH){
            chosenAngleH = theta;
            minimumCostH = costH;
        }
        if(costV < minimumCostV){
            chosenAngleV = theta;
            minimumCostV = costH;
        }             
    }
    setAngleH(chosenAngleH);
    setAngleV(chosenAngleV);
    saveVectorAsMatlabScript(monotonyCostHVec,"monotonyCostH");
    saveVectorAsMatlabScript(monotonyCostHVec,"monotonyCostV");
    saveVectorAsMatlabScript(thetaVec,"thetaVec");

    
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
    //std::cout<<"In calcModelCovMatrix()"<<std::endl;
    //std::cout<<"Valid Positions H 42: "<<this->validPositions.valid_positions_h[42].item()<<std::endl;

    at::Tensor modelCovFun = calcModelCovFun(ssi, isHorizontal);


    //at::Tensor horizontalGrid = at::abs(u - ssi.getDisparity() * s);
    //saveTensorAsMatlabScript(horizontalGrid,"mGrid");

    // if(!isHorizontal){
    //     std::cout<<"Inside CovMatrix Function Vertical"<<std::endl;
    //     std::cout<<modelCovFun[8][32].item()<<std::endl;
    //     std::cout<<rhoSpt<<" "<<rhoAng<<std::endl;
    //     ssi.print();
    //     saveTensorAsMatlabScript(modelCovFun,"modelCovFunVerticalissimo");
    // }
    // if(isHorizontal){
    //     std::cout<<"Inside CovMatrix Function Horizontal"<<std::endl;
    //     std::cout<<modelCovFun[8][32].item()<<std::endl;      
    //     std::cout<<rhoSpt<<" "<<rhoAng<<std::endl;
    //     ssi.print();
    //     saveTensorAsMatlabScript(modelCovFun,"modelCovFunHorizontalisssimo");
    // }

    
    at::Tensor modelCovMat = covFun2Mat(modelCovFun,isHorizontal);
    

    
    
    //modelCovMat = modelCovMat.round(20);
    
    // if(ssi.getDCode() == 45 && ssi.getRhoUCode() == 99){
    // if(isHorizontal){
        
    //     std::cout<<"HORIZONTAL MODEL COV: "<<modelCovFun.sizes()<<std::endl;
    //     std::cout<<modelCovFun.index({at::indexing::Slice(8,15),at::indexing::Slice(63,69)})<<std::endl;
    // }else{
    //     std::cout<<"Vertical MODEL COV: "<<modelCovFun.sizes()<<std::endl;
    //     std::cout<<modelCovFun.index({at::indexing::Slice(8,15),at::indexing::Slice(63,69)})<<std::endl;
    // }
    // }

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
    // Check if block is valid (has non-zero data)
    if(this->validPositions.valid_positions_h.size(0) == 0 || this->validPositions.valid_positions_v.size(0) == 0){
        return std::numeric_limits<double>::max();
    }
    
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
    // Check if block is valid (has non-zero data)
    if(this->validPositions.valid_positions_h.size(0) == 0 || this->validPositions.valid_positions_v.size(0) == 0){
        return {0.0, 0.0};
    }
    
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
    //temp.setSpatialRhos(covFunH,covFunV);
    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2;     
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2;     
    temp.setRhoU(covFunH[k_center][m_center+1].item<double>());
    temp.setRhoV(covFunV[l_center][n_center+1].item<double>());
    //std::cout<<covFunH[k_center][m_center+1].item<double>()<<" "<<covFunV[l_center][n_center+1].item<double>()<<std::endl;
    //temp.setRhoU(0.99999);
    //temp.setRhoV(0.99999);
    temp.setAngularRhos();
    
    //std::cout<<"RhoU = "<<covFunH[k_center][m_center+1].item<double>()<<" RhoV = "<<covFunV[l_center][n_center+1].item<double>()<<std::endl;

    at::Tensor iSqrtCovMatH = block.iSqrtCovMat(true);
    //std::cout<<"iSqrt: "<<std::endl<<iSqrtCovMatH[0]<<std::endl<<std::endl<<std::endl;
    at::Tensor iSqrtCovMatV = block.iSqrtCovMat(false);
    //std::cout<<iSqrtCovMatH.sizes()<<std::endl;
    //std::cout<<iSqrtCovMatV.sizes()<<std::endl;
    //saveTensorAsMatlabScript(iSqrtCovMatH,"iSqrtCovMatH");
    //SgtSideInfo modelModel(25,25,this->disparityRange);
    //at::Tensor modelModeliCov = block.iSqrtCovMat(block.calcModelCovMatrix(modelModel,true),true);

    
   

    double min = std::numeric_limits<double>::max();
    double minV = std::numeric_limits<double>::max();
    double minH = std::numeric_limits<double>::max();
    double angleV = 0;
    double angleH = 0;
    //temp.print();
    
    for(double theta = this->angleRange[0];theta<=this->angleRange[1];theta+=PRECISION_ANGLE){
    //for(double theta = 20;theta<=20;theta+=PRECISION_ANGLE){
        temp.setAngleH(theta);
        temp.setAngleV(theta);
        //std::cout<<"PRINT COMING"<<std::endl;
        //temp.print();
        //double d = tan(acos(-1)/180 * theta)
        at::Tensor modelCovMatH = block.calcModelCovMatrix(temp,true);
        at::Tensor modelCovMatV = block.calcModelCovMatrix(temp,false);

        //saveTensorAsMatlabScript(modelCovMatH,"modelCovComparison/modelCovMatH"+std::to_string((int)theta));
        //saveTensorAsMatlabScript(modelCovMatV,"modelCovComparison/modelCovMatV"+std::to_string((int)theta));
         //Little Test
        //at::Tensor iSqrtModelCovMatH = block.iSqrtCovMat(modelCovMatH,true);


        // if(theta == 0){
        //     saveTensorAsMatlabScript(modelCovMatH,"modelCovMatH0");
        //     std::cout<<"TEST RESULT = "<<genDivergence(modelCovMatH,iSqrtModelCovMatH)<<std::endl;
        //     saveTensorAsMatlabScript(iSqrtModelCovMatH,"iSqrtModelCovMatH");

        // }
        // if(d == -3){
        //     std::cout<<modelCovMatH<<std::endl;
        // }
        double genDivH = genDivergence(modelCovMatH,iSqrtCovMatH);
        double genDivV = genDivergence(modelCovMatV,iSqrtCovMatV);
        //double genDivCheck = genDivergence(modelCovMatV,iSqrtModelCovMatH);
        //std::cout<<theta<<": "<<genDivCheck<<std::endl;
        //double testDiv = genDivergence(modelCovMatH,modelModeliCov);

        double genDiv = genDivH + genDivV;
        genDivHVec.push_back(genDivH);
        genDivVVec.push_back(genDivV);
        thetaVec.push_back(theta);
        ///std::cout<<theta<<": "<<testDiv<<std::endl;

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
        

        //std::cout<<"Angle Range divergence:" <<this->angleRange[0]<<" "<<this->angleRange[1]<<std::endl;
        //std::cout<<theta<<": "<<genDiv<<" "<<genDivH<<" "<<genDivV<< " "<<min<<std::endl;

        // if (genDiv < min){

        //     min = genDiv;
        //     setAngleV(theta);
        //     setAngleH(theta);
        // }
    }
    //std::cout<<"angleV: "<<angleV<<" angleH: "<<angleH<<std::endl;

    // setRhoS(temp.getRhoS());
    //     setRhoU(temp.getRhoU());
    //     setRhoV(temp.getRhoV());
    //     setRhoT(temp.getRhoT());
    //saveVectorAsMatlabScript(genDivHVec,"dummyReg_genDivH");
    //saveVectorAsMatlabScript(genDivVVec,"dummyReg_genDivV");
    //saveVectorAsMatlabScript(thetaVec,"dummyReg_thetaVec");
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
    // Handle empty covariance matrix
    if(covMat.numel() == 0 || covMat.size(-1) == 0){
        return at::empty({0}, covMat.dtype());
    }
    
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
at::Tensor Block4D_::iSqrtCovMat(bool isHorizontal ) const{
    //std::cout<<"is Horizontal: "<<isHorizontal<<std::endl;
    at::Tensor covFun = normalizeCov(this->covFun(isHorizontal));
    at::Tensor covMat = covFun2Mat(covFun,isHorizontal);
    //at::Tensor covMat = this->batchedCovMatrix(isHorizontal);  
    return iSqrtCovMat(covMat,isHorizontal );
}

at::Tensor Block4D_::covFun2Mat(const at::Tensor& covFun,bool isHorizontal) const{
    at::Tensor covMat;
    if(includesInvalidCorners){
        covMat = covFun2MatValid(covFun,isHorizontal);
    }else{
        //std::cout<<"Correctimundo"<<std::endl;
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
    //std::cout<<validIndexes.sizes()<<std::endl;
    //std::cout<<validIndexes.dtype()<<std::endl;
    at::Tensor covMat = covFun2MatAll(covFun);
    //std::cout<<"CovMat Size: "<<covMat.sizes()<<std::endl;
    //std::cout<<"Is Horizontal: "<<isHorizontal<<std::endl;
    //std::cout<<"validIndexes 42: "<<validIndexes[42].item()<<std::endl;
    //std::cout<<"Valid Positions H 42: "<<this->validPositions.valid_positions_h[42].item()<<std::endl;

    //std::cout<<"Valid Indexes: "<<validIndexes.unsqueeze(0)<<std::endl;
    //std::cout<<"Valid Positions H: "<<this->validPositions.valid_positions_h.unsqueeze(0)<<std::endl;

    covMat = covMat.index({at::indexing::Slice(), validIndexes});
        //std::cout<<"CovMat Size: "<<covMat.sizes()<<std::endl;

    covMat = covMat.index({validIndexes,at::indexing::Slice()});
        //std::cout<<"CovMat Size: "<<covMat.sizes()<<std::endl;

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
    //at::Tensor covFunH = Block4D_::normalizeCov(block.covFun(true));
    //at::Tensor covFunV = Block4D_::normalizeCov(block.covFun(false));
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


        //double rhoSpace = 0.8;
        //temp.setRhoU(rhoSpace);
        //temp.setRhoV(rhoSpace);
        

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
// void Block4D_::YCbCr2RGB_BT601(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Cb, Block4D_ const &Cr, int Scale) {
//     // std::cout<<"Data type = "<<Y.data.dtype()<<std::endl;
//     // std::cout<<"Decoding Scale = "<<Scale<<std::endl;
//     // std::cout<<"Chrominance Bias Correction: "<<((Scale+1)/2)<<std::endl;
//     Block4d_ Ytemp = Y.clone()data.to(at::kDouble);
//     auto CbTemp = (Cb.data - ((Scale+1)/2)).to(at::kDouble);
//     auto CrTemp = (Cr.data - ((Scale+1)/2)).to(at::kDouble);

//     //std::cout<<Ytemp[0][0][0][0].item()<<" "<<CbTemp[0][0][0][0].item()<<" "<<CrTemp[0][0][0][0].item()<<std::endl;
//     R = (Ytemp - 0.0000071525 * CbTemp + 1.4020 * CrTemp).round().to(at::kInt);
//     G = (Ytemp- 0.34413 * CbTemp - 0.71414 * CrTemp).round().to(at::kInt);
//     B = (Ytemp + 1.7720 * CbTemp - 0.000040249 * CrTemp).round().to(at::kInt);
//    //std::cout<<Ytemp[0][0][0][0].item()<<" "<<-0.34413 * CbTemp[0][0][0][0].item<double>()<<" "<<- 0.71414 *(CrTemp[0][0][0][0].item<double>())<<" "<<G.data[0][0][0][0]<<std::endl;

// }

// void Block4D_::YCoCg2RGB(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale) {
//     auto CoTemp = Co.data - (Scale+1)/2;
//     auto CgTemp = Cg.data - (Scale+1)/2;
//     auto t = Y - (CgTemp.bitwise_right_shift(1));
//     G = CgTemp + t;
//     B.data = t - (CoTemp.bitwise_right_shift(1));
//     R.data = B.data + CoTemp;          
// }
// void Block4D_::RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
//     static const auto Y_weights = at::tensor({0.299, 0.587, 0.114}, at::kDouble).reshape({3, 1});  
//     static const auto Cb_weights = at::tensor({-0.168736, -0.331264, 0.5}, at::kDouble).reshape({3, 1}); 
//     static const auto Cr_weights = at::tensor({0.5, -0.418688, -0.081312}, at::kDouble).reshape({3, 1}); 
//     static const int D = 1<<((int)log2(Scale+1)-8);
//     static const int Y8bitBias = 0;
//     static const int CbCr8bitBias = (1<<9);
//     //std::cout<<"D = "<<D<<" CbCr8bitBias = "<<CbCr8bitBias<<" Y8bitBias = "<<Y8bitBias<<std::endl;

//     auto Ey = R.data.to(at::kDouble)/Scale * Y_weights[0] + G.data.to(at::kDouble)/Scale * Y_weights[1] + B.data.to(at::kDouble)/Scale * Y_weights[2];
//     //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Y_weights[0].item()<<" Ey min = "<<Ey.min().item()<<" max = "<<Ey.max().item()<<std::endl;

//     Y.data = ((1023 * Ey + Y8bitBias)).round().to(at::kInt);

//     auto Ecb = R.data.to(at::kDouble)/Scale * Cb_weights[0] + G.data.to(at::kDouble)/Scale * Cb_weights[1] + B.data.to(at::kDouble)/Scale * Cb_weights[2];
//     //std::cout<<"ECb min = "<<Ecb.min().item()<<" max = "<<Ecb.max().item()<<std::endl;
//     Cb.data = ((1023 * Ecb + CbCr8bitBias)).round().to(at::kInt);
//     auto Ecr = R.data.to(at::kDouble)/Scale * Cr_weights[0] + G.data.to(at::kDouble)/Scale * Cr_weights[1] + B.data.to(at::kDouble)/Scale * Cr_weights[2];
//     //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Cr_weights[0]<<" Ecr min = "<<Ecr.min().item()<<" max = "<<Ecr.max().item()<<std::endl;
    
//     Cr.data = ((1023 * Ecr + CbCr8bitBias)).round().to(at::kInt); 
//     Y.validPositions = R.validPositions;
//     Cb.validPositions = R.validPositions;
//     Cr.validPositions = R.validPositions;
// }

// void Block4D_::RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
//     Co = R.data - B.data;
//     auto temp = B.data + Co.data.bitwise_right_shift(1);
//     Cg = G.data - temp;
//     Y = temp + Cg.data.bitwise_right_shift(1);
//     Co.data+= (Scale + 1)/2;
//     Cg.data+= (Scale + 1)/2;
//     Y.validPositions = R.validPositions;
//     Co.validPositions = R.validPositions;
//     Cg.validPositions = R.validPositions;        
// }

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
    this->validPositions = B.validPositions;
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
    if(this->includesInvalidCorners){
        newBlock.validPositions = ValidPositions(this->validPositions.valid_positions_h.clone(), this->validPositions.valid_positions_v.clone());
    }
    //newBlock.validPositions = ValidPositions(this->validPositions.valid_positions_h.clone(), this->validPositions.valid_positions_v.clone());
    newBlock.orderH = this->orderH;
    newBlock.orderV = this->orderV;
    newBlock.lightFieldPosition = this->lightFieldPosition;
    newBlock.lightField = this->lightField;

    return newBlock;
}

/********Static Functions******/

at::Tensor Block4D_::get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal){
  
    //std::cout<<"block_start: "<<block_start[0]<<" "<<block_start[1]<<" "<<block_start[2]<<" "<<block_start[3]<<std::endl;
    //std::cout<<"block_shape: "<<block_shape[0]<<" "<<block_shape[1]<<" "<<block_shape[2]<<" "<<block_shape[3]<<std::endl;
    //std::cout<<"lf_shape: "<<lf_shape[0]<<" "<<lf_shape[1]<<" "<<lf_shape[2]<<" "<<lf_shape[3]<<std::endl;
    
    if(block_start[2] >= lf_shape[2] || block_start[3] >= lf_shape[3]){
        if(block_shape[0] == 1){
            std::cerr<<"Block start is out of bounds! "<<block_start[2]<<" >= "<<lf_shape[2]<<" || "<<block_start[3]<<" >= "<<lf_shape[3]<<std::endl;

        }
        at::Tensor valid_positions = at::empty({0},at::kLong);
        return valid_positions;
    }
    int64_t view_coordinate = 0;
    int64_t spatial_coordinate = 2;
    if(is_horizontal){
        view_coordinate = 1;
        spatial_coordinate = 3;
    }
    int lf_extra_size = (int)(abs(round(adjustment_d*(lf_shape[view_coordinate]-1))));
    lf_shape[spatial_coordinate] = lf_shape[spatial_coordinate] - lf_extra_size;
    lf_shape[3] = lf_shape[3] - lf_extra_size;

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
        int n_start = floor(l_*true_alpha);
        int n_end = (lf_shape[spatial_coordinate]) + floor(l_*true_alpha);

        if(true_alpha < 0){
            n_start -= (lf_shape[view_coordinate]-1)*true_alpha;
            n_end -= (lf_shape[view_coordinate]-1)*true_alpha;
        }
        //std::cout<<l_<<": "<<n_start<<" "<<n_end<<std::endl;

        torch::TensorOptions options = torch::TensorOptions();
        at::Tensor indexes =  at::empty({0},options.dtype(at::kLong));
        if(n_start-block_start[spatial_coordinate] < block_shape[spatial_coordinate]){
            int64_t blk_n_start = std::max(n_start-block_start[spatial_coordinate],(int64_t)0);
            int64_t blk_n_end = std::min(n_end-block_start[spatial_coordinate],block_shape[spatial_coordinate]);
            
            if (blk_n_end - blk_n_start < 1) continue;
            indexes = (at::range(blk_n_start,blk_n_end-1,1)+block_shape[spatial_coordinate]*l_).to(at::kLong);
        }
        padding_coordinates.push_back(indexes);
         
    }
    
    at::Tensor vectorized_padding;
    if (padding_coordinates.empty()) {
        vectorized_padding = at::empty({0}, at::kLong);
    } else {
        vectorized_padding = torch::cat(padding_coordinates);
    }
    if(false){
    //if(block_start[3] == 512 && block_shape[3] == 64 && is_horizontal){
        std::cout<<"Block Shape: "<<block_shape[0]<<" "<<block_shape[1]<<" "<<block_shape[2]<<" "<<block_shape[3]<<std::endl;
        std::cout<<"Block Start: "<<block_start[0]<<" "<<block_start[1]<<" "<<block_start[2]<<" "<<block_start[3]<<std::endl;
        std::cout<<"vectorized_padding size: "<<vectorized_padding.sizes()<<std::endl;
        for(int i = 0; i<vectorized_padding.size(0);i++){
            std::cout<<"("<<(int) vectorized_padding[i].item<int64_t>()/block_shape[spatial_coordinate]<<" "<<vectorized_padding[i].item<int64_t>()%block_shape[spatial_coordinate]<<") ";
        }
        std::cout<<std::endl;
    }
    //std::cout<<"Initial Tenosr Type: "<<vectorized_padding.dtype()<<std::endl;
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




