#include "LightField/LightField.h"
#include "LightField/Block4D.h"
#include "Encoder/Hierarchical4DEncoder.h"
#include "Encoder/TransformPartition.h"
#include <boost/program_options.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <array>
#include <algorithm>
#include <filesystem>
#include <cctype>

using namespace std;
using namespace filesystem;


bool is_all_whitespace(const std::string& str) {
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return std::isspace(c);
    });
}

class EncoderParameters;
enum ExtensionType { REPEAT_LAST, CYCLIC, NONE};
enum ColorTransformType {BT601,YCOCG};
void ExtendBlock4D(Block4D &extendedblock, ExtensionType extensionMethod, int extensionLength, char direction);
void RGB2YCbCr_BT601(Block4D &Y, Block4D &Cb, Block4D &Cr, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
int readProgramOptions(int argc, char **argv, EncoderParameters &par);
void conflicting_options(const boost::program_options::variables_map & vm,
                         const std::string & opt1, const std::string & opt2);
void BigEndianUnsignedIntegerWrite(unsigned long int value, int precision, FILE *outputFilePointer);
void BigEndianSignedIntegerWrite(long int value, int precision, FILE *outputFilePointer);




class EncoderParameters {
public:
    double Lambda = 1.0;
    std::array<int64_t,4> minPartitionSize = {13,13,15,15};
    std::array<int64_t,4> maxPartitionSize = {4,4,4,4};
    std::array<int64_t,2> viewSize = {13,13};
    std::array<int64_t,2> firstView = {0,0}; //TO DO
    bool isLenslet13x13 = false; //uNSURE WHAT THE FUCK THIS DOES
    std::string inputDirectory = "./ExampleLightField/";
    std::string outputFileName = "out.comp";
    std::string configFile = "";
    std::array<double,2> disparityRange = {-3.5,3.5};
    ExtensionType extensionMethod = REPEAT_LAST;
    double transformGain = 1;        
    ColorTransformType colorTransformType = BT601; 
    int verbosity = false; //TO DO
    void ReadConfigurationFile(std::string parametersFileName); 
    void DisplayConfiguration(void);
};
void EncoderParameters :: ReadConfigurationFile(std::string parametersFileName) {
    std::cout<<"HELLO!"<<std::endl;
    std::ifstream parametersFile(parametersFileName);
    // FILE *parametersFilepointer;
    
    // if((parametersFilepointer = fopen(parametersFileName.c_str(), "r")) == NULL) {
    //     printf("ERROR: unable to open configuration file %s\n", parametersFileName);
    //     exit(0);
    // } 
    if( ! parametersFile ) {
        std::cerr << "Error opening input file" << std::endl ;
        return ;
    }
    while(!parametersFile.eof()){
        std::string command;
        parametersFile>>command;
        if(!command.compare("-lambda")){
            parametersFile>>Lambda;
        }
        if(!command.compare("-l")){
            parametersFile>>maxPartitionSize[0]>>maxPartitionSize[1]>>maxPartitionSize[2]>>maxPartitionSize[3];
        }
        if(!command.compare("-m")){
            parametersFile>>minPartitionSize[0]>>minPartitionSize[1]>>minPartitionSize[2]>>minPartitionSize[3];
        }
        if(!command.compare("-r")){
            parametersFile>>disparityRange[0]>>disparityRange[1];
        }
        if(!command.compare("-u")){
            parametersFile>>maxPartitionSize[3];
        }
        if(!command.compare("-v")){
            parametersFile>>maxPartitionSize[2];
        }
        if(!command.compare("-s")){
            parametersFile>>maxPartitionSize[1];
        }
        if(!command.compare("-t")){
            parametersFile>>maxPartitionSize[0];
        }
        if(!command.compare("-d")){
            parametersFile>>inputDirectory;
        }
        if(!command.compare("-o")){
            parametersFile>>outputFileName;
        }
        if(!command.compare("-nv")){
            parametersFile>>viewSize[0];
        }
        if(!command.compare("-nh")){
            parametersFile>>viewSize[1];
        }
        if(!command.compare("-off_v")){
            parametersFile>>firstView[0];
        }
        if(!command.compare("-off_h")){
            parametersFile>>firstView[1];
        }
        if(!command.compare("-lenslet13x13")){
            isLenslet13x13 = true;
        }
        if(!command.compare("-extension_repeat")){
            extensionMethod = REPEAT_LAST;
        }
        if(!command.compare("-extension_none")){
            extensionMethod =  NONE;
        }
        if(!command.compare("-extension_cyclic")){
            extensionMethod =  CYCLIC;
        }
        if(!command.compare("-t_gain")){
            parametersFile>>transformGain;
        }
        if(!command.compare("-min_t")){
            parametersFile>>minPartitionSize[0];
        }
        if(!command.compare("-min_s")){
            parametersFile>>minPartitionSize[1];
        }
        if(!command.compare("-min_v")){
            parametersFile>>minPartitionSize[2];
        }
        if(!command.compare("-min_u")){
            parametersFile>>minPartitionSize[3];
        }
        if(!command.compare("-bt601")){
            colorTransformType = BT601;
        }
        if(!command.compare("-ycocg")){
            colorTransformType = YCOCG;
        }
        if(!command.compare("-VV")){
            verbosity = true;
        }               
    }
}


void EncoderParameters :: DisplayConfiguration(void) {
    cout<<"Lambda = "<<Lambda<<endl;
    cout<<"Max Partition Size (t,s,v,u) = ("<<maxPartitionSize[0]<<","
                                            <<maxPartitionSize[1]<<","
                                            <<maxPartitionSize[2]<<","
                                            <<maxPartitionSize[3]<<")"<<endl;
    cout<<"Min Partition Size (t,s,v,u) = ("<<minPartitionSize[0]<<","
                                            <<minPartitionSize[1]<<","
                                            <<minPartitionSize[2]<<","
                                            <<minPartitionSize[3]<<")"<<endl;
    cout<<"View Size (t,s) = ("<<viewSize[0]<<","<<viewSize[1]<<")"<<endl;
    cout<<"First View Offset (t,s) = ("<<firstView[0]<<","<<firstView[1]<<")"<<endl;
    cout<<"Transform Gain = "<<transformGain<<endl;
    cout<<"Disparity Range = [ "<<disparityRange[0]<<","<<disparityRange[1]<<"]"<<endl;
    cout<<"Input Directory = "<<inputDirectory<<endl;
    cout<<"Output Directory = "<<outputFileName<<endl;
    cout<<"Lenslet 13x13 = "<<isLenslet13x13<<endl;
    cout<<"Extension Method = "<<extensionMethod<<endl;
    cout<<"Color Transform Type = "<<colorTransformType<<endl;
    cout<<"Verbosity = "<<verbosity<<endl;
}
   

void conflicting_options(const boost::program_options::variables_map & vm,
                         const std::string & opt1, const std::string & opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() &&
        vm.count(opt2) && !vm[opt2].defaulted())
    {
        throw std::logic_error(std::string("Conflicting options '") +
                               opt1 + "' and '" + opt2 + "'.");
    }
}
int readProgramOptions(int argc, char **argv, EncoderParameters &par) {
    namespace po = boost::program_options;
      po::options_description options;
    options.add_options()
        ("help,h", "Show help message")
        ("config-file,c", po::value<string>(&par.configFile), "Path to Configuration File")
        ("light-field-dir,d", po::value<string>(&par.inputDirectory), "Path to Light Field Collection Directory")
        ("output-dir,o", po::value<string>(&par.outputFileName), "Path to Compressed Output File")
        ("lambda", po::value<double>(&par.Lambda),  "Regex pattern for view positioning")
        ("maximum-partition-size,l", po::value<std::vector<int64_t>>()->multitoken(), "Maximum Partition Length t s v u") 
        ("minimum-partition-size,m", po::value<std::vector<int64_t>>()->multitoken(), "Minimum Partition Length t s v u") 
        ("disp-range,r", po::value<std::vector<double>>()->multitoken(), "Disparity Range (-5,5) is a good compromise for most LFs") 
        ("transform-gain,g", po::value<double>(&par.transformGain),  "Transform Gain")
        ("num-views,v", po::value<std::vector<std::int64_t>>()->multitoken(),  "Number of Views: T S")
        ("view-offset,b", po::value<std::vector<std::int64_t>>()->multitoken(),  "Index of First View: T S")
        ("extension-repeat", po::bool_switch()->default_value(false),  "Sets Extention to Repeat")
        ("extension-cyclic", po::bool_switch()->default_value(false),  "Sets Extention to Cyclic")
        ("extension-none", po::bool_switch()->default_value(false),  "Sets Extention to None")
        ("bt601", po::bool_switch()->default_value(false),  "Sets Color Transform to YCbCr BT601")
        ("ycocg", po::bool_switch()->default_value(false),  "Sets Color Transform to YCOCG")
        ("isLenslet13x13",po::bool_switch()->default_value(false), "Increases Brightness of Edge Views")
        ("verbosity,V", po::bool_switch()->default_value(false),  "Sets Verbosity to true");



    po::variables_map vm;
    
    po::store(po::parse_command_line(argc, argv,options),vm);
    if (vm.count("help")){
        cout << options << endl;
        return 1;
    }
    conflicting_options(vm,"bt601","ycocg");
    conflicting_options(vm,"extension-repeat","extention-cyclic");
    conflicting_options(vm,"extension-cyclic","extention-none");
    conflicting_options(vm,"extension-repeat","extention-none");
    try {
        po::notify(vm);
    } catch(exception& e) {
        cout << e.what() << endl;
        return -1;
    }
    if(vm["isLenslet13x13"].as<bool>()) par.isLenslet13x13 = true;
    if(vm["bt601"].as<bool>()) par.colorTransformType = BT601;
    if(vm["ycocg"].as<bool>()) par.colorTransformType = YCOCG;
    if(vm["extension-repeat"].as<bool>()) par.extensionMethod = REPEAT_LAST;
    if(vm["extension-cyclic"].as<bool>()) par.extensionMethod = CYCLIC;
    if(vm["extension-none"].as<bool>()) par.extensionMethod = NONE;
    if(vm.count("maximum-partition-size")){
        std::vector<int64_t> data = vm["maximum-partition-size"].as<std::vector<int64_t>>();
        if (data.size() != 4) throw std::invalid_argument("maximum-partition-size must have 4 elements");
        std::copy(data.begin(), data.end(), par.maxPartitionSize.begin());
    }
    if(vm.count("minimum-partition-size")){
        std::vector<int64_t> data = vm["minimum-partition-size"].as<std::vector<int64_t>>();
        if (data.size() != 4) throw std::invalid_argument("minimum-partition-size must have 4 elements");
        std::copy(data.begin(), data.end(), par.minPartitionSize.begin());
    }
    if(vm.count("disp-range")){
        std::vector<double> data = vm["disp-range"].as<std::vector<double>>();
        if (data.size() != 2) throw std::invalid_argument("disp-range must have 2 elements");
        std::copy(data.begin(), data.end(), par.disparityRange.begin());
    }
    if(vm.count("num-views")){
        std::vector<int64_t> data = vm["num-views"].as<std::vector<int64_t>>();
        if (data.size() != 2) throw std::invalid_argument("num-views must have 2 elements");
        std::copy(data.begin(), data.end(), par.viewSize.begin());
    }
    if(vm.count("view-offset")){
        std::vector<int64_t> data = vm["view-offset"].as<std::vector<int64_t>>();
        if (data.size() != 2) throw std::invalid_argument("view-offset must have 2 elements");
        std::copy(data.begin(), data.end(), par.firstView.begin());
    }
    par.verbosity = vm["verbosity"].as<bool>();
    return 0;
}

int main(int argc, char **argv) {
    
    torch::InferenceMode guard;
    // torch::set_num_threads(1);
    // at::set_num_threads(1);
    //DEFAULT Encoder
    EncoderParameters par;

    int v = readProgramOptions(argc, argv, par);
    if(v != 0){
        return v;
    }
    if(par.configFile.compare("") != 0){
        par.ReadConfigurationFile(par.configFile);
    } 


    if(par.verbosity > 0){
        par.DisplayConfiguration();
    }

    par.Lambda *= par.transformGain*par.transformGain;


    

    Block4D yBlock,cbBlock,crBlock; 

    


 
    std::array<int64_t,4> extensionLength;

    std::cout<<"Opening Stuff and things:"<<std::endl;
    //std::string folder = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/DebugData/";
    //std::string experiment = "DebugPrinting/";
    std::string filename = filesystem::path(par.outputFileName).filename().string();
    std::string stem = filesystem::path(par.outputFileName).stem().string();
    std::string path = filesystem::path(par.outputFileName).parent_path().string()+"/";
    std::string infoPath;
    std::cout<<"Path: "<<path<<" "<<filename<<std::endl;
    std::cout<<"Path True: "<<filesystem::absolute(path)<<std::endl;
    //std::string experiment = "Greek/64-4-angle3/";
    //std::string experiment = "Greek/64-8/";
    //std::string path = folder + experiment;
    int counter = 1;
    std::string originalPath = path;
    if (path == "/"){
        par.outputFileName = filename;
        path = "";
        infoPath =  stem + "_info.json";
        std::cout<<"info: "<<filesystem::absolute(infoPath)<<std::endl;
        std::cout<<"file: "<<filesystem::absolute(par.outputFileName)<<std::endl;
    }else{
        if (originalPath.back() == '/') {
            originalPath.pop_back(); // Remove trailing slash if present
        }
        while (exists(path)) {
            path = originalPath + "-" + std::to_string(counter)+"/";
            counter++;
        }
    
        std::cout<<filesystem::absolute(path)<<std::endl;
        std::cout<<filesystem::absolute(path+filename)<<std::endl;
        create_directories(path);
        par.outputFileName = path + filename;
        infoPath =  path + "info.json";
    }
   


    LightField inputLF;
    string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    inputLF.OpenLightFieldPPM_(par.inputDirectory,pattern,par.firstView,par.viewSize);  


    for(int n = 0; n < 4; n++) {
        extensionLength[n] = inputLF.data.size(n) % par.maxPartitionSize[n];
    }
    
       
    FILE *outputFileNamePointer;
    if((outputFileNamePointer = fopen(par.outputFileName.c_str(), "wb")) == NULL) {
        cout<<"Error: Could not open output file: "<<par.outputFileName<<std::endl;
        return -1;
    }

    Hierarchical4DEncoder hdt(par.maxPartitionSize[0] * par.maxPartitionSize[2], par.maxPartitionSize[1]* par.maxPartitionSize[3]);


    //writes the superior bit plane value
    BigEndianUnsignedIntegerWrite(hdt.mSuperiorBitPlane, 2, outputFileNamePointer);
    std::cout<<"Superior bit plane = "<<hdt.mSuperiorBitPlane<<std::endl;
    
    //writes the maximum transform sizes
    for(int n = 0; n < 4; n++) {
        BigEndianUnsignedIntegerWrite(par.maxPartitionSize[n], 2, outputFileNamePointer);
    }
    std::cout<<"Max partition size = "<<par.maxPartitionSize[0]<<","<<par.maxPartitionSize[1]<<","<<par.maxPartitionSize[2]<<","<<par.maxPartitionSize[3]<<std::endl;
    //Writes the total size of the LF
    for(int n = 0; n < 4; n++) {
        BigEndianUnsignedIntegerWrite(inputLF.data.size(n), 2, outputFileNamePointer);
    }
    
    //Writes an Integer Encoded Disparity Range of the LF
    for(int n = 0; n < 2; n++) {
        BigEndianSignedIntegerWrite((int)(par.disparityRange[n]*1000), 3, outputFileNamePointer);
    }
    std::cout<<"Disparity Range: "<<par.disparityRange[0]<<" "<<par.disparityRange[1]<<std::endl;

  

    //writes the bit precision of each component of the pixels of the views
    BigEndianUnsignedIntegerWrite(inputLF.mPGMScale, 2, outputFileNamePointer);


    //cout<<"mPGM scale = "<<inputLF.mPGMScale<<endl;
    TransformPartition tp(par.minPartitionSize,hdt,par.disparityRange,par.transformGain);
    tp.mEntropyCoder.StartEncoder(outputFileNamePointer);

    std::array<double,3> error = {0,0,0};
    bool second_half = false;
    double size = 0;
    for(int verticalView = 0; verticalView < inputLF.data.size(0); verticalView += par.maxPartitionSize[0]) {
        for(int horizontalView = 0; horizontalView < inputLF.data.size(1); horizontalView += par.maxPartitionSize[1]) {
            //for(int viewLine = 64; viewLine < 64 + par.maxPartitionSize[2]; viewLine += par.maxPartitionSize[2]) {
            //for(int viewLine = 0*par.maxPartitionSize[2]; viewLine < 0*par.maxPartitionSize[2] + 2*par.maxPartitionSize[2]; viewLine += par.maxPartitionSize[2]) {
            for(int viewLine = 0; viewLine < inputLF.data.size(2); viewLine += par.maxPartitionSize[2]) {
                //for(int viewColumn = 192; viewColumn <192  + par.maxPartitionSize[3]; viewColumn += par.maxPartitionSize[3]) {
                //for(int viewColumn = 0*par.maxPartitionSize[3]; viewColumn <0*par.maxPartitionSize[3]  + 2*par.maxPartitionSize[3]; viewColumn += par.maxPartitionSize[3]) {
                //for(int viewColumn = 0*par.maxPartitionSize[3]; viewColumn <0*par.maxPartitionSize[3]  + 2*par.maxPartitionSize[3]; viewColumn += par.maxPartitionSize[3]) {
                for(int viewColumn = 0; viewColumn < inputLF.data.size(3); viewColumn += par.maxPartitionSize[3]) {
                    printf("transforming the 4D block at position (%d %d %d %d)\n", verticalView, horizontalView, viewLine, viewColumn);
                    std::array<int64_t,4> blockPosition = {verticalView,horizontalView,viewLine,viewColumn};


                    Block4D rBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,0);
                    Block4D gBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,1);
                    Block4D bBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,2);
                    std::cout<<" Read Block 4D"<<std::endl;
        
                    if(par.isLenslet13x13 == 1) {
                        //Correcting the values of the edge views of the light field by multiplying them by 4.
                        if(verticalView == 0) {
                            if(horizontalView == 0) {
                                rBlock.Shift_UVPlane(2, 0, 0);
                                gBlock.Shift_UVPlane(2, 0, 0);
                                bBlock.Shift_UVPlane(2, 0, 0);
                            }

                            if((horizontalView + par.maxPartitionSize[1] >= inputLF.data.size(1))&&(horizontalView <= inputLF.data.size(1))) {
                                int lastViewH = inputLF.data.size(1)-horizontalView-1;
                                rBlock.Shift_UVPlane(2, 0, lastViewH);
                                gBlock.Shift_UVPlane(2, 0, lastViewH);
                                bBlock.Shift_UVPlane(2, 0, lastViewH);
                            }
                        }

                        if((verticalView + par.maxPartitionSize[0] >= inputLF.data.size(0))&&(verticalView <= inputLF.data.size(0))) {

                            int lastViewV = inputLF.data.size(0)-verticalView-1;
                            if(horizontalView == 0) {
                                rBlock.Shift_UVPlane(2, lastViewV, 0);
                                gBlock.Shift_UVPlane(2, lastViewV, 0);
                                bBlock.Shift_UVPlane(2, lastViewV, 0);
                            }
                            if((horizontalView + par.maxPartitionSize[1] >= inputLF.data.size(1))&&(horizontalView <= inputLF.data.size(1))) {
                                int lastViewH = inputLF.data.size(1)-horizontalView-1;
                                rBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                                gBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                                bBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                            }
                        }
                    }
                    if(par.colorTransformType == BT601){
                        std::cout<<" Attempting BT601 Color Transformation"<<std::endl;
                        yBlock = rBlock.clone();
                        cbBlock = gBlock.clone();
                        crBlock = bBlock.clone();
                        std::cout<<"Includes Invalid Corners: "<<yBlock.includesInvalidCorners<<std::endl;
                        std::cout<<"ValidSize: "<< rBlock.validPositions.valid_positions_v.sizes()<<" x "<<rBlock.validPositions.valid_positions_h.sizes()<<std::endl;
                        std::cout<<"ValidSize: "<< yBlock.validPositions.valid_positions_v.sizes()<<" x "<<yBlock.validPositions.valid_positions_h.sizes()<<std::endl;

                        std::cout<<"Includes Invalid Corners: "<<cbBlock.includesInvalidCorners<<std::endl;
                        std::cout<<"Includes Invalid Corners: "<<crBlock.includesInvalidCorners<<std::endl;

                        std::cout<<rBlock.data.max()<<std::endl;
                        RGB2YCbCr_BT601(yBlock, cbBlock, crBlock, rBlock, gBlock, bBlock, inputLF.mPGMScale);
                        std::cout<<" Completed BT601 Color Transformation"<<std::endl;

                    }
                    if(par.colorTransformType == YCOCG){
                        RGB2YCoCg(yBlock, cbBlock, crBlock, rBlock, gBlock, bBlock, inputLF.mPGMScale);
                        std::cout<<" Completed YCOCG Color Transformation"<<std::endl;

                    }

                    for(int spectralComponent = 0; spectralComponent < 3; spectralComponent++) {
                        if(par.verbosity > 0) printf("\nProcessing spectral component %d\n", spectralComponent);
                        Block4D lfBlock;
                        if(spectralComponent == 0){
                            lfBlock = yBlock;
                        }
                        if(spectralComponent == 1){
                            lfBlock = cbBlock;
                        }
                        if(spectralComponent == 2){
                            lfBlock = crBlock;
                        }
                        lfBlock = lfBlock - (inputLF.mPGMScale+1)/2;
                        for(int n = 0; n<4; n++){
                            if(blockPosition[n] + par.maxPartitionSize[n] > inputLF.data.size(n)){
                                ExtendBlock4D(lfBlock, par.extensionMethod, extensionLength[n], n);
                            }
                        } 
                        std::cout<<"Ready to Encode"<<std::endl;
                        if(par.verbosity > 0) {
                            //std::cout<<"YUV Block"<<std::endl;
                            //std::cout<<lfBlock.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
                        }
                                                                                              
                        tp.RDoptimizeTransform(lfBlock, par.Lambda);
                        tp.EncodePartition(par.Lambda);
                        //std::cout<<"Size Channel "<<spectralComponent<<": "<<tp.mCodingPartitionInfo.getTotalSize()<<std::endl;
                        
                        std::cout<<"Encoding Successful!"<<std::endl;
                        // std::cout<<"Encoded"<<std::endl;
                        int sizeV = std::min(par.maxPartitionSize[2],inputLF.data.size(2)-viewLine);
                        int sizeH = std::min(par.maxPartitionSize[3],inputLF.data.size(3)-viewColumn);
                        
                        std::cout<<"Block Size: "<<sizeH<<" "<<sizeV<<std::endl;
                        
                    }
                }
            }
        }
    }
    std::cout<<"Total Distortion: "<<error[0]<<" "<<error[1]<<" "<<error[2]<<std::endl;
    double mseY = error[0]/(inputLF.data.size(0)*inputLF.data.size(1)*inputLF.data.size(2)*inputLF.data.size(3));
    double mseCb = error[1]/(inputLF.data.size(0)*inputLF.data.size(1)*inputLF.data.size(2)*inputLF.data.size(3));
    double mseCr = error[2]/(inputLF.data.size(0)*inputLF.data.size(1)*inputLF.data.size(2)*inputLF.data.size(3));

    double PSNR_Y = 10*log10((1024*1024)/mseY);
    double PSNR_Cb = 10*log10((1024*1024)/mseCb);
    double PSNR_Cr = 10*log10((1024*1024)/mseCr);
    std::cout<<"Predicted PSNR-Y: "<<PSNR_Y<<std::endl;
    std::cout<<"Predicted PSNR-YUV:"<<(6*PSNR_Y+PSNR_Cb+PSNR_Cr)/8<<std::endl;
    std::cout<<"Total Rate: "<<size<<std::endl;

    
    tp.mEntropyCoder.DoneEncoding();
    
    fclose(outputFileNamePointer);
    return 0;
}

void ExtendBlock4D(Block4D &extendedBlock, ExtensionType extensionMethod, int extensionLength, char direction) {
    
    if(extensionMethod == REPEAT_LAST) {
        
        if(direction == 't') 
            extendedBlock.Extend_T(extensionLength);
        if(direction == 's') 
            extendedBlock.Extend_S(extensionLength);
        if(direction == 'v') 
            extendedBlock.Extend_V(extensionLength);
        if(direction == 'u') 
            extendedBlock.Extend_U(extensionLength);
        
    }
    if(extensionMethod == CYCLIC) {
        
        if(direction == 't') 
            extendedBlock.CopySubblockFrom(extendedBlock, {0, 0, 0, 0}, {extensionLength, 0, 0, 0});
        if(direction == 's') 
            extendedBlock.CopySubblockFrom(extendedBlock, {0, 0, 0, 0},{ 0, extensionLength, 0, 0});
        if(direction == 'v') 
            extendedBlock.CopySubblockFrom(extendedBlock,{ 0, 0, 0, 0},{0, 0, extensionLength, 0});
        if(direction == 'u') 
            extendedBlock.CopySubblockFrom(extendedBlock,{0, 0, 0, 0},{0, 0, 0, extensionLength});
    }
    if(extensionMethod == NONE) {
            
    }
}

void RGB2YCbCr_BT601(Block4D &Y, Block4D &Cb, Block4D &Cr, Block4D const &R, Block4D const &G, Block4D const &B, int Scale) {
    int* Y_data = Y.data.data_ptr<int>();
    int* Cb_data = Cb.data.data_ptr<int>();
    int* Cr_data = Cr.data.data_ptr<int>();
    int* R_data = R.data.data_ptr<int>();
    int* G_data = G.data.data_ptr<int>();
    int* B_data = B.data.data_ptr<int>();
    for(int n = 0; n < R.size[0]*R.size[1]*R.size[2]*R.size[3]; n++) {
        double pixel =  0.299 * R_data[n] + 0.587 * G_data[n] + 0.114 * B_data[n];
        Y_data[n] = (int) round(pixel);
        pixel = -0.16875 *(double) R_data[n] -0.33126 *(double) G_data[n] + 0.5 * (double)B_data[n];
        Cb_data[n] = (int) round(pixel) + (Scale + 1)/2;
        pixel = 0.5 *(double) R_data[n] -0.41869 * (double) G_data[n] -0.08131  * (double)B_data[n];
        Cr_data[n] = (int) round(pixel) + (Scale + 1)/2;
    }
}



void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale) {
    Co = R - B;
    auto temp = B + Co.data.bitwise_right_shift(1);
    Cg = G - temp;
    Y = temp + Cg.data.bitwise_right_shift(1);
    Co.data+= (Scale + 1)/2;
    Cg.data+= (Scale + 1)/2;
    Y.validPositions = R.validPositions;
    Co.validPositions = R.validPositions;
    Cg.validPositions = R.validPositions;        
}

void BigEndianUnsignedIntegerWrite(unsigned long int value, int precision, FILE *outputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char output_byte;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        output_byte = (value >> (8*byte_index))&255;
        fwrite(&output_byte, 1, 1, outputFilePointer);
    }
    
}
void BigEndianSignedIntegerWrite(long int value, int precision, FILE *outputFilePointer) {
    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    unsigned char sign = (value < 0) ? 1 : 0;
    fwrite(&sign,1,1,outputFilePointer);
    unsigned int magnitude = abs(value);
    unsigned char output_byte;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        output_byte = (magnitude >> (8*byte_index))&255;
        fwrite(&output_byte, 1, 1, outputFilePointer);
    }
    
}
