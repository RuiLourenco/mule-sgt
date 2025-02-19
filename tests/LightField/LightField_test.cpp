#include <LightField/LightField.h>

//#include <LightField/Block4D_.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string.h>
#include <torch/torch.h>

using namespace testing;


using namespace std;

double mse(const at::Tensor& a, const at::Tensor& b) {
    //cout<<"MSE CALCULATION!"<<endl;
    auto mse =(a - b).pow(2).to(at::kDouble).mean(); 
    //cout<<"MSE CALCULATION Complete!"<<endl;

    return mse.item<double>();
};

double mse(const vector<int>& a, const vector<int>& b) {
   //cout<<"MSE CALCULATION!"<<endl;
   
    vector<int> difference(a.size());
    std::set_difference(a.begin(),a.end(),b.begin(),b.end(),difference.begin());
   // cout<<"difference size:" <<difference.size()<<endl;
    double mse_count = 0;
    std::for_each(difference.begin(), difference.end(), [&](int i) { mse_count += i*i;} );
   // cout<<mse_count<<endl;
    mse_count = mse_count/(double)difference.size();
    return mse_count;
};
// class LightFieldTest : public testing::Test {
//  protected:
//    LightField inputLF;



//   LightFieldTest():inputLF(9,9,512) {
//      //Upload Test Light Field Old Way
     
//     std::cout<<"pointer address = "<<inputLF.mViewCache<<std::endl;
//     std::cout<<inputLF.mViewCache[0][0].mLines<<std::endl;
//     std::cout<<inputLF.mViewCache[0][0].mColumns<<std::endl;
//     std::cout<<"("<<inputLF.mViewCache[0][0].mLines<<","<<inputLF.mViewCache[0][0].mColumns<<")"<<std::endl;

//     inputLF.mVerticalViewNumberOffset = 0;
//     inputLF.mHorizontalViewNumberOffset = 0;
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     cout<<inputDirectory<<endl;
//     string extension = ".ppm";
//     cout<<extension<<endl;
//     inputLF.OpenLightFieldPPM(strdup(inputDirectory.c_str()), strdup(extension.c_str()), 9, 9, 3, 3, 'r');
//     cout<<"Uploaded LightField The Old Way"<<endl;
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     cout<<"Uploaded LightField The New Way"<<endl;
//     //Block4D_ block_new = inputLF.ReadBlock4DfromLightField_({2,2,2,2,1},{0,0,0,0,0}); //this will crash

//   }
  
// };



// TEST(LightFieldTest, BlockFromLightField){
//   LightField inputLF(9,9,512);
//   inputLF.mVerticalViewNumberOffset = 0;
//   inputLF.mHorizontalViewNumberOffset = 0;
//   string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//   //cout<<inputDirectory<<endl;
//   string extension = ".ppm";
//   inputLF.OpenLightFieldPPM(strdup(inputDirectory.c_str()), strdup(extension.c_str()), 9, 9, 3, 3, 'r');
//   //cout<<"Uploaded LightField The Old Way"<<endl;
//   string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//   inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//   //cout<<"Uploaded LightField The New Way"<<endl;

//   Block4D block_old;
//   block_old.SetDimension(2,2,2,2);
//   inputLF.ReadBlock4DfromLightField(&block_old,0, 0, 0, 0);  
//   vector<int> blockOldVec(block_old.mPixelData,block_old.mPixelData+2*2*2*2);
//   Block4D_ block_new = inputLF.ReadBlock4DfromLightField_({2,2,2,2},{0,0,0,0},0);
//   at::Tensor blockNew = block_new.data.to(torch::kInt).cpu().contiguous().view({-1});
//   vector<int> blockNewVec(blockNew.data_ptr<int>(),blockNew.data_ptr<int>()+2*2*2*2);
//   //cout<<blockNewVec[0]<<" "<<blockOldVec[0]<<endl;
//   ASSERT_EQ(blockOldVec.size(), blockNewVec.size());
//   ASSERT_EQ(2*2*2*2, blockOldVec.size());
//   ASSERT_EQ(2*2*2*2, blockNewVec.size());
//  // cout<<"calculating mse"<<endl;
//   EXPECT_DOUBLE_EQ(0.0,mse(blockOldVec,blockNewVec));
// }

TEST(LightFieldTest,ReadWriteReadLoop){
  string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
  string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
  LightField inputLF(9,9,512);
  inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
  //cout<<"opened: "<<inputDirectory<<endl;
  std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/greek-copy/";
  //cout<<"writing!"<<endl;
  inputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
  //cout<<"wrote to: "<<outputDirectory<<endl;
  LightField readLF(9,9,512);
  readLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
  //cout<<" read from: "<<inputDirectory<<endl;
  //cout<<readLF.data.sizes()<<" "<<inputLF.data.sizes()<<endl;
  double mse_error = mse(readLF.data,inputLF.data);
 // cout<<"MSE: "<<mse_error<<endl;
  EXPECT_DOUBLE_EQ(0.0, mse_error);
  //cout<<"test passed"<<endl;
}




TEST(LightFieldTest,OnesBlocks){
  Block4D_ ones = Block4D_({2,2,2,2});
  ones.Ones();
  double sum = ones.data.sum().item<double>();
  EXPECT_DOUBLE_EQ(16.0,sum);
}
TEST(LightFieldTest,ZerosBlocks){
  Block4D_ zeros = Block4D_({2,2,2,2});
  zeros.Zeros();
  double sum = zeros.data.sum().item<double>();
  EXPECT_DOUBLE_EQ(0.0,sum);
}


TEST(LightFieldTest,BlockAdditionWithScalar){
  Block4D_ block = Block4D_({2,2,2,2});
  block.Ones();
  block = block + 7;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}
TEST(LightFieldTest,BlockMultiplicationWithScalar){
  Block4D_ block = Block4D_({2,2,2,2});
  block.Ones();
  block = block * 8;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}
TEST(LightFieldTest,BlockDivisionWithScalar){
  Block4D_ block = Block4D_({2,2,2,2});
  block.Ones();
  block = block * 8*2;
  block = block/2;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}

TEST(LightFieldTest,BlockSubtractionWithScalar){
  Block4D_ block = Block4D_({2,2,2,2});
  block.Ones();
  block = block + 8;
  block = block - 1;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}
TEST(LightFieldTest,BlockAdditionWithBlock){
  Block4D_ block = Block4D_({2,2,2,2});
  Block4D_ block2 = Block4D_({2,2,2,2});
  block.Ones();
  block2.Ones();
  block2 = block2 * 7;
  block = block + block2;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}
TEST(LightFieldTest,BlockMultiplicationWithBlock){
  Block4D_ block = Block4D_({2,2,2,2});
  Block4D_ block2 = Block4D_({2,2,2,2});
  block.Ones();
  block2.Ones();
  block2 = block2 * 8;
  block = block * block2;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}
TEST(LightFieldTest,BlockSubtractionWithBlock){
  Block4D_ block = Block4D_({2,2,2,2});
  Block4D_ block2 = Block4D_({2,2,2,2});
  block.Ones();
  block2.Ones();
  block = block * 10;
  block2 = block2 * 2;
  block = block - block2;
  int sum = block.data.sum().item<int>();
  EXPECT_EQ(128.0,sum);
}

TEST(LightFieldTest,BlockShiftView){
  Block4D_ block = Block4D_({2,2,2,2});
  block.Ones();
  block = block + 7;
  block.Shift_UVPlane(2,0,0);
  int new_value = block.data[0][0][0][0].item<int>();
  EXPECT_EQ(new_value,8<<2);
  block.Shift_UVPlane(-2,0,1);
  int new_value1 = block.data[0][1][0][0].item<int>();

  EXPECT_EQ(new_value1,8>>2);
}

TEST(LightFieldTest,FromRValueTensor){
  Block4D_ block = Block4D_({2,2,2,2});
  Block4D_ block2 = Block4D_({2,2,2,2});
  block.Ones();
  block2.Ones();
  Block4D_ block3 = Block4D_(block + block2);
  int sum = block3.data.sum().item<int>();
  EXPECT_EQ(sum, 2*2*2*2*2);

}

TEST(LightFieldTest,BlockFrom4Sublocks){
  Block4D_ B00 = Block4D_({2,2,2,2});
  Block4D_ B01 = Block4D_({2,2,2,2});
  Block4D_ B11 = Block4D_({2,2,2,2});
  Block4D_ B10 = Block4D_({2,2,2,2});

  B00.Zeros();
  B01.Ones();
  B11.Ones();
  B10.Ones();
  B11 = B11*2;
  B10 = B10*3;
  Block4D_ bigDaddy = Block4D_(B00,B01,B10,B11,false);
  cout<<bigDaddy.data[0][0]<<endl;
  cout<<bigDaddy.data.sizes()<<endl;
  EXPECT_EQ(bigDaddy.data[0][0][0][0].item<int>(),0);
  EXPECT_EQ(bigDaddy.data[0][0][0][3].item<int>(),1);
  EXPECT_EQ(bigDaddy.data[0][0][3][3].item<int>(),0);
  EXPECT_EQ(bigDaddy.data[0][0][3][0].item<int>(),1);
  
}

TEST(LightFieldTest,LightFieldFromBlocks){
  LightField lightField({9,9,20,20,3});

  Block4D_ B00 = Block4D_({9,9,10,10});
  Block4D_ B01 = Block4D_({9,9,10,10});
  Block4D_ B11 = Block4D_({9,9,10,10});
  Block4D_ B10 = Block4D_({9,9,10,10});

  B00.Zeros();
  B01.Ones();
  B11.Zeros();
  B10.Ones();

  for(int c = 0; c < 3; c++){
    lightField.WriteBlock4DtoLightField_(B00,{0,0,0,0,c});
    lightField.WriteBlock4DtoLightField_(B10,{0,0,10,0,c});
    lightField.WriteBlock4DtoLightField_(B01,{0,0,0,10,c});
    lightField.WriteBlock4DtoLightField_(B11,{0,0,10,10,c});
  }

  for( int l = 0; l < 9; l++){
    for( int k = 0; k < 9; k++){
      for( int c = 0; c < 3; c++){
        EXPECT_EQ(lightField.data[l][k][0][0][c].item<short>(),0);
        EXPECT_EQ(lightField.data[l][k][9][9][c].item<short>(),0);
        
        EXPECT_EQ(lightField.data[l][k][10][10][c].item<short>(),0);
        EXPECT_EQ(lightField.data[l][k][19][19][c].item<short>(),0);
        
        EXPECT_EQ(lightField.data[l][k][10][0][c].item<short>(),1);
        EXPECT_EQ(lightField.data[l][k][19][9][c].item<short>(),1);
        
        EXPECT_EQ(lightField.data[l][k][0][10][c].item<short>(),1);
        EXPECT_EQ(lightField.data[l][k][9][19][c].item<short>(),1);
      }
    }
  }

}

TEST(SGTTEST,isqrt){
    string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
    string pattern = R"((?P<V>.*)_(?P<U>.*)\.ppm)";
    LightField inputLF(9,9,512);
    inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
    Block4D_ block = inputLF.ReadBlock4DfromLightField_({9,9,16,16},{0,0,0,0},0);
    at::Tensor iSqrtCovMatH = block.iSqrtCovMat(true);
    cout<<iSqrtCovMatH<<endl;
}

TEST(SGTTest,CalcDisparity){
  string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
  string pattern = R"((?P<V>.*)_(?P<U>.*)\.ppm)";
  LightField inputLF(9,9,512);
  inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
  Block4D_ block = inputLF.ReadBlock4DfromLightField_({9,9,16,16},{0,0,0,0},0);

  SgtSideInfo ssi;
  double startingDisparity = ssi.getDisparityV();
  cout<<startingDisparity<<endl;
  ssi.estimateDisparity(block);
  cout<<ssi.getDisparityV()<<endl;
  EXPECT_NEAR(2.75,ssi.getDisparityV(),0.3);
}


TEST(SGTTest,SideInfoCalc){
  string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
  string pattern = R"((?P<V>.*)_(?P<U>.*)\.ppm)";
  LightField inputLF(9,9,512);
  inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
  Block4D_ block = inputLF.ReadBlock4DfromLightField_({9,9,16,16},{0,0,0,0},0);

  SgtSideInfo ssi(block,{-0.1,0.1});
  cout<<ssi.getDisparityV()<<endl;
}



TEST(SGTTest,CompressDecompressHolyGrail){
  string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
  string pattern = R"((?P<V>.*)_(?P<U>.*)\.ppm)";
  LightField inputLF(9,9,512);
  inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
  Block4D_ block = inputLF.ReadBlock4DfromLightField_({9,9,16,16},{0,0,0,0},0);
  Block4D_ blockStart = block.clone();
  block.sgtTransform(1<<17,{-0.1,0.1});
  cout<<"weight: "<<(1<<17)<<endl;
  cout<<"FIRST ELEMENT: "<<block.data[0][0][0][0].item()<<endl;
  block.isgtTransform(1<<17,block.ssi);
  cout<<block.data.dtype()<<" "<<blockStart.data.dtype()<<endl;
  double mseTransform = mse(block.data,blockStart.data); 
  cout<<"Error"<<mseTransform<<endl;
  EXPECT_NEAR(0.0,mseTransform,1e-20);
}
TEST(SGTTest,NoQuantizationScale){
  Block4D_ ones({9,9,16,16});
  ones.Ones();
  ones = ones * 1023;
  SgtSideInfo ssi;
  ssi.setAngleV(1.00);
  ones.ssi = ssi;
  ones.sgtTransform(1);
  cout<<ones.data[0][0]<<endl;
  cout<<"Potential Max Coef = "<<ones.data[0][0][0][0].item()<<endl;

}

void RGB2YCbCr_BT601_test(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
  static const auto Y_weights = at::tensor({0.299, 0.587, 0.114}, at::kDouble).reshape({3, 1});  
  static const auto Cb_weights = at::tensor({-0.168736, -0.331264, 0.5}, at::kDouble).reshape({3, 1}); 
  static const auto Cr_weights = at::tensor({0.5, -0.418688, -0.081312}, at::kDouble).reshape({3, 1}); 
   at::Tensor Rtemp = R.data.to(at::kDouble);
   at::Tensor Gtemp = G.data.to(at::kDouble);
   at::Tensor Btemp = B.data.to(at::kDouble) ;

   Y.data = (Y_weights[0] * Rtemp + Y_weights[1] * Gtemp + Y_weights[2] * Btemp).round().to(at::kInt);
   Cb.data = (Cb_weights[0] * Rtemp + Cb_weights[1] * Gtemp + Cb_weights[2] * Btemp).round().to(at::kInt);
   Cr.data = (Cr_weights[0] * Rtemp + Cr_weights[1] * Gtemp + Cr_weights[2] * Btemp).round().to(at::kInt);
   Cb.data += (Scale-1)/2;
   Cr.data += (Scale-1)/2;
}
TEST(ColorTransformTests,YCbCr2RGB_BT601_RED){
  
    std::array<int64_t,4> length = {9,9,16,16};
    Block4D_ R(length);
    Block4D_ G (length);
    Block4D_ B (length);

    R.Ones();
    R.data = R.data * ((1<<10) - 1);
    G.Zeros();
    B.Zeros();

    Block4D_ Y(length);
    Block4D_ Cb(length);
    Block4D_ Cr(length);
    Block4D_ R_rec(length);
    Block4D_ B_rec(length);
    Block4D_ G_rec(length);

    //Block4D_::RGB2YCbCr_BT601(Y,Cb,Cr,R,G,B,((1<<10) - 1));
    RGB2YCbCr_BT601_test(Y,Cb,Cr,R,G,B,(1<<10)-1);
    Block4D_::YCbCr2RGB_BT601(R_rec,G_rec,B_rec,Y,Cb,Cr,((1<<10) - 1));

    cout<<"R min = "<<R.data.min().item()<<" max = "<<R.data.max().item()<<endl;
    cout<<"G min = "<<G.data.min().item()<<" max = "<<G.data.max().item()<<endl;
    cout<<"B min = "<<B.data.min().item()<<" max = "<<B.data.max().item()<<endl<<endl;
    cout<<"Y min = "<<Y.data.min().item()<<" max = "<<Y.data.max().item()<<endl;
    cout<<"Cb min = "<<Cb.data.min().item()<<" max = "<<Cb.data.max().item()<<endl;
    cout<<"Cr min = "<<Cr.data.min().item()<<" max = "<<Cr.data.max().item()<<endl<<endl;
    cout<<"R rec min = "<<R_rec.data.min().item()<<" max = "<<R_rec.data.max().item()<<endl;
    cout<<"G rec min = "<<G_rec.data.min().item()<<" max = "<<G_rec.data.max().item()<<endl;
    cout<<"B rec min = "<<B_rec.data.min().item()<<" max = "<<B_rec.data.max().item()<<endl;
    double mse_error_R = mse(R_rec.data,R.data);
    double mse_error_B = mse(B_rec.data,B.data);
    double mse_error_G = mse(G_rec.data,G.data);
    EXPECT_DOUBLE_EQ(0.0, mse_error_R);
    EXPECT_DOUBLE_EQ(0.0, mse_error_B);
    EXPECT_DOUBLE_EQ(0.0, mse_error_G);
}

TEST(ColorTransformTests,YCbCr2RGB_BT601){
    string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
    string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    LightField inputLF(9,9,512);
    std::array<int64_t,4> length = {9,9,16,16};
    inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
    Block4D_ R = inputLF.ReadBlock4DfromLightField_(length,{0,0,23,439},0);
    Block4D_ G = inputLF.ReadBlock4DfromLightField_(length,{0,0,23,439},1);
    Block4D_ B = inputLF.ReadBlock4DfromLightField_(length,{0,0,23,439},2);
    Block4D_ Y(length);
    Block4D_ Cb(length);
    Block4D_ Cr(length);
    Block4D_ R_rec(length);
    Block4D_ B_rec(length);
    Block4D_ G_rec(length);

    Block4D_::RGB2YCoCg(Y,Cb,Cr,R,G,B,inputLF.mPGMScale);
    Block4D_::YCoCg2RGB(R_rec,G_rec,B_rec,Y,Cb,Cr,inputLF.mPGMScale);

    cout<<"R min = "<<R.data.min().item()<<" max = "<<R.data.max().item()<<endl;
    cout<<"G min = "<<G.data.min().item()<<" max = "<<G.data.max().item()<<endl;
    cout<<"B min = "<<B.data.min().item()<<" max = "<<B.data.max().item()<<endl<<endl;
    cout<<"Y min = "<<Y.data.min().item()<<" max = "<<Y.data.max().item()<<endl;
    cout<<"Cb min = "<<Cb.data.min().item()<<" max = "<<Cb.data.max().item()<<endl;
    cout<<"Cr min = "<<Cr.data.min().item()<<" max = "<<Cr.data.max().item()<<endl<<endl;
    cout<<"R rec min = "<<R_rec.data.min().item()<<" max = "<<R_rec.data.max().item()<<endl;
    cout<<"G rec min = "<<G_rec.data.min().item()<<" max = "<<G_rec.data.max().item()<<endl;
    cout<<"B rec min = "<<B_rec.data.min().item()<<" max = "<<B_rec.data.max().item()<<endl;
    double mse_error_R = mse(R_rec.data,R.data);
    double mse_error_B = mse(B_rec.data,B.data);
    double mse_error_G = mse(G_rec.data,G.data);
    EXPECT_DOUBLE_EQ(0.0, mse_error_R);
    EXPECT_DOUBLE_EQ(0.0, mse_error_B);
    EXPECT_DOUBLE_EQ(0.0, mse_error_G);
}



// TEST(LightFieldTest,LightFieldFromBlocks){

// }

// TEST(LightFieldTest,BlockColorTransforms){

// }
// TEST(LightFieldTest,CopySubblockFrom){

// }
// TEST(LightFieldTest, BlockFrom4Subblocks){

// }


// int main(int argc, char** argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }