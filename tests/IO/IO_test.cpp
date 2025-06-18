#include <IO/io.h>
#include <gtest/gtest.h>
#include <string.h>
#include <torch/torch.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>



namespace fs = boost::filesystem;
namespace bio = boost::iostreams;


double mse(const at::Tensor& a, const at::Tensor& b) {
    auto mse =(a - b).pow(2).to(at::kDouble).mean(); 
    return mse.item<double>();
};


TEST(IO_Tests, ReadPPM) {

    std::string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/000_000";
    std::string extension = ".ppm";
    std::ifstream is{inputDirectory+extension};
    at::Tensor image = io::read_ppm(is);
    EXPECT_EQ(image.size(0), 512);
    EXPECT_EQ(image.size(1), 512);
    EXPECT_EQ(image.size(2), 3);    
}

TEST(IO_Tests, ReadPPM_Collection) {
    std::string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
    std::string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    int scale;
    at::Tensor collection = io::read_collection(inputDirectory,pattern,scale);
    EXPECT_EQ(collection.size(0), 9);
    EXPECT_EQ(collection.size(1), 9);
    EXPECT_EQ(collection.size(2), 512);
    EXPECT_EQ(collection.size(3), 512);
    EXPECT_EQ(collection.size(4), 3);
    
}

TEST(IO_Tests,Read_WRITE_READ_PPM_View){
    std::string inputView = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/000_000.ppm";
    std::string outputView = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/test_view.ppm";
    fs::ifstream is(inputView,std::ios::in | std::ios::binary);
    at::Tensor image = io::read_ppm(is);
    fs::ofstream os;
    os.open(outputView, std::ios::out | std::ios::binary);
    io::write_ppm(image,os);
    fs::ifstream is1(outputView,std::ios::in | std::ios::binary);

    at::Tensor read_view = io::read_ppm(is1);
    EXPECT_DOUBLE_EQ(0.0, mse(image, read_view));
}
TEST(IO_Tests, Write_PPM_Collection){
    std::string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
    std::string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    int scale;
    at::Tensor collection = io::read_collection(inputDirectory,pattern,scale);
    std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/greek-copy/";
    io::write_collection(outputDirectory, collection,{0,0});
    //std::cout<<"collection_written"<<std::endl;
    at::Tensor copy_collection = io::read_collection(outputDirectory,pattern,scale);
    //std::cout<<"copy_collection_read"<<std::endl;
    //std::cout<<copy_collection[0][0][0][0][0]<<" "<<collection[0][0][0][0][0]<<std::endl;
    EXPECT_DOUBLE_EQ(0.0, mse(copy_collection,collection));
}
TEST(IO_Tests, WritePPM) {
    std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/example.ppm";
    fs::ofstream os;
    os.open(outputDirectory, std::ios::out | std::ios::binary);
    at::Tensor image = (torch::rand({512, 512, 3})*1024).to(at::kLong);
    ASSERT_EQ(image.size(0), 512);
    ASSERT_EQ(image.size(1), 512);
    ASSERT_EQ(image.size(2), 3);
    io::write_ppm(image, os);
    //std::cout<<"Done Writing"<<std::endl;
    fs::ifstream is(outputDirectory,std::ios::in | std::ios::binary);

    at::Tensor read_image =io::read_ppm(is);
    //std::cout<<"Done Reading"<<std::endl;

    EXPECT_EQ(read_image.size(0), 512);
    EXPECT_EQ(read_image.size(1), 512);
    EXPECT_EQ(read_image.size(2), 3);   
    long example_value = image[0][1][2].item<int64_t>();
    long read_value = read_image[0][1][2].item<int64_t>();
    //std::cout<<example_value<<" "<<read_value<<std::endl;
    EXPECT_EQ(example_value, read_value);
    EXPECT_DOUBLE_EQ(0.0,mse(read_image,image));

}