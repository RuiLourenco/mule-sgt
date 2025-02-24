#include "LightField/LightField.h"
#include "LightField/Block4D_.h"
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

using namespace std;
using namespace filesystem;



class EncoderParameters;
enum ExtensionType { REPEAT_LAST, CYCLIC, NONE};
enum ColorTransformType {BT601,YCOCG};
void ExtendDCT(Matrix &extendedDCT, ExtensionType extensionMethod, int transformLength, int extensionLength);
void ExtendBlock4D(Block4D_ &extendedblock, ExtensionType extensionMethod, int extensionLength, char direction);
void RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
void RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
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
        if(!command.compare("-lf")){
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
            extensionMethod = extensionMethod = NONE;
        }
        if(!command.compare("-extension_cyclic")){
            extensionMethod = extensionMethod = CYCLIC;
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


    LightField inputLF;
    string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    inputLF.OpenLightFieldPPM_(par.inputDirectory,pattern,'r');
       
    Block4D_ lfBlock(par.maxPartitionSize);  
    Block4D_ rBlock(par.maxPartitionSize); 
    Block4D_ gBlock(par.maxPartitionSize); 
    Block4D_ bBlock(par.maxPartitionSize);
    Block4D_ yBlock(par.maxPartitionSize); 
    Block4D_ cbBlock(par.maxPartitionSize);
    Block4D_ crBlock(par.maxPartitionSize);
    
    Hierarchical4DEncoder hdt;
    TransformPartition tp;
    tp.mlength_t_min = par.minPartitionSize[0];
    tp.mlength_s_min = par.minPartitionSize[1];
    tp.mlength_v_min = par.minPartitionSize[2];
    tp.mlength_u_min = par.minPartitionSize[3];

 
    std::array<int64_t,4> extensionLength;

    std::cout<<"Opening Stuff and things:"<<std::endl;
    std::string folder = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/DebugData/";
    //std::string experiment = "TimingConsiderations/";
    std::string experiment = "Sideboard/32-8-angle10/";
    // std::string experiment = "Greek/64-8/";
    std::string path = folder + experiment;
    create_directory(path);

    for(int n = 0; n < 4; n++) {
        extensionLength[n] = inputLF.data.size(n) % par.maxPartitionSize[n];
    }
    
       
    FILE *outputFileNamePointer;
    if((outputFileNamePointer = fopen(par.outputFileName.c_str(), "wb")) == NULL) {
        cout<<"Error: Could not open output file: "<<par.outputFileName<<std::endl;
        return -1;
    }

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
        BigEndianSignedIntegerWrite((int)(par.disparityRange[n]*100), 2, outputFileNamePointer);
    }


    //writes the bit precision of each component of the pixels of the views
    BigEndianUnsignedIntegerWrite(inputLF.mPGMScale, 2, outputFileNamePointer);
    //cout<<"mPGM scale = "<<inputLF.mPGMScale<<endl;
    at::Tensor lfEnergy = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfRhoS = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfRhoT = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfRhoU = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfRhoV = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfAngleV = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfAngleH = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfRate = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    at::Tensor lfDistortion = at::zeros({inputLF.data.size(2),inputLF.data.size(3),3},at::kDouble);
    //std::cout<<inputLF.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4),0})<<std::endl<<std::endl;
    //std::cout<<inputLF.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4),1})<<std::endl<<std::endl;;
    //std::cout<<inputLF.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4),2})<<std::endl<<std::endl;;
    
    
    hdt.StartEncoder(outputFileNamePointer);
    for(int verticalView = 0; verticalView < inputLF.data.size(0); verticalView += par.maxPartitionSize[0]) {
        for(int horizontalView = 0; horizontalView < inputLF.data.size(1); horizontalView += par.maxPartitionSize[1]) {
            for(int viewLine = 0; viewLine < inputLF.data.size(2); viewLine += par.maxPartitionSize[2]) {
                for(int viewColumn = 0; viewColumn < inputLF.data.size(3); viewColumn += par.maxPartitionSize[3]) {
                    if(true)
                        printf("transforming the 4D block at position (%d %d %d %d)\n", verticalView, horizontalView, viewLine, viewColumn);
                    std::array<int64_t,4> blockPosition = {verticalView,horizontalView,viewLine,viewColumn};

                    rBlock.Zeros();
                    gBlock.Zeros();
                    bBlock.Zeros();
                    rBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,0);
                    gBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,1);
                    bBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,2);
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
                    if(par.verbosity > 0) {
                        // std::cout<<"R"<<std::endl;
                        // std::cout<<rBlock.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
                        // std::cout<<"G"<<std::endl;
                        // std::cout<<gBlock.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
                        // std::cout<<"B"<<std::endl;
                        // std::cout<<bBlock.data.index({4,4,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
                    }
                    if(par.colorTransformType == BT601){
                        RGB2YCbCr_BT601(yBlock, cbBlock, crBlock, rBlock, gBlock, bBlock, inputLF.mPGMScale);
                        std::cout<<" Completed BT601 Color Transformation"<<std::endl;

                    }
                    if(par.colorTransformType == YCOCG){
                        RGB2YCoCg(yBlock, cbBlock, crBlock, rBlock, gBlock, bBlock, inputLF.mPGMScale);
                        std::cout<<" Completed YCOCG Color Transformation"<<std::endl;

                    }

                    for(int spectralComponent = 0; spectralComponent < 3; spectralComponent++) {
                        if(par.verbosity > 0) printf("\nProcessing spectral component %d\n", spectralComponent);
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
                                                                                              

                        hdt.RestartProbabilisticModel();
                        tp.RDoptimizeTransform_(lfBlock, hdt,par.disparityRange,par.transformGain, par.Lambda);
                        tp.EncodePartition_(hdt, par.Lambda);
                        // std::cout<<"Encoded"<<std::endl;
                        // std::cout<<tp.costImage.sizes()<<std::endl;
                        lfEnergy.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.costImage;
                        // // std::cout<<"Cost Image Fine"<<std::endl;
                        lfRhoS.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.rhoSImage;
                        // // std::cout<<"RhoS Fine"<<std::endl;

                        lfRhoT.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.rhoTImage;
                        // // std::cout<<"RhoT Fine"<<std::endl;
                        lfRhoU.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.rhoUImage;
                        // // std::cout<<"RhoU Fine"<<std::endl;
                        lfRhoV.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.rhoVImage;
                        // //std::cout<<"RhoV Fine"<<std::endl;
                        lfAngleV.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.angleImageV;
                        //std::cout<<"AngleV Fine"<<std::endl;
                        lfAngleH.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.angleImageH;
                        //std::cout<<"AngleH Fine"<<std::endl;
                        lfRate.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.rateImage;
                        lfDistortion.index({at::indexing::Slice(viewLine,viewLine+par.maxPartitionSize[2]),at::indexing::Slice(viewColumn,viewColumn+par.maxPartitionSize[3]),spectralComponent}) = tp.distortionImage;
                        //std::cout<<"Rate Fine"<<std::endl;
                    }            
                }
            }
        }
    }
    //write_tensor(hdt.ignored[0][0],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/ignored.png");
    std::ofstream energy;
    std::ofstream rhoS;
    std::ofstream rhoT;
    std::ofstream rhoU;
    std::ofstream rhoV;
    std::ofstream angleH;
    std::ofstream angleV;
    std::ofstream rate;
    std::ofstream distortion;

    energy.open(path + "energy.m");
    rhoS.open(path + "rhoS.m");
    rhoT.open(path + "rhoT.m");
    rhoU.open(path + "rhoU.m");
    rhoV.open(path + "rhoV.m");
    angleH.open(path + "angleH.m");
    angleV.open(path + "angleV.m");
    rate.open(path + "rate.m");
    distortion.open(path + "distortion.m");
    std::cout<<"Printing Images"<<endl;
    energy<<"energy_cpp = zeros("<<lfEnergy.size(0)<<","<<lfEnergy.size(1)<<","<<lfEnergy.size(2)<<");"<<std::endl;
    rhoS<<"rhoS_cpp = zeros("<< lfRhoS.size(0)<<","<< lfRhoS.size(1)<<","<< lfRhoS.size(2)<<");"<<std::endl;
    rhoT<<"rhoT_cpp = zeros("<<lfRhoT.size(0)<<","<<lfRhoT.size(1)<<","<<lfRhoT.size(2)<<");"<<std::endl;
    rhoU<<"rhoU_cpp = zeros("<<lfRhoU.size(0)<<","<<lfRhoU.size(1)<<","<<lfRhoU.size(2)<<");"<<std::endl;
    rhoV<<"rhoV_cpp = zeros("<<lfRhoV.size(0)<<","<<lfRhoV.size(1)<<","<<lfRhoV.size(2)<<");"<<std::endl;
    angleH<<"angleH_cpp = zeros("<<lfAngleH.size(0)<<","<<lfAngleH.size(1)<<","<<lfAngleH.size(2)<<");"<<std::endl;
    angleV<<"angleV_cpp = zeros("<<lfAngleV.size(0)<<","<<lfAngleV.size(1)<<","<<lfAngleV.size(2)<<");"<<std::endl;
    rate<<"rate_cpp = zeros("<<lfRate.size(0)<<","<<lfRate.size(1)<<","<<lfRate.size(2)<<");"<<std::endl;
    distortion<<"distortion_cpp = zeros("<<lfDistortion.size(0)<<","<<lfDistortion.size(1)<<","<<lfDistortion.size(2)<<");"<<std::endl;
    for(int n = 0; n < lfEnergy.size(0); n++){
        for(int m = 0; m < lfEnergy.size(1); m++){
            energy<<"energy_cpp("<<n+1<<","<<m+1<<",1) = "<<lfEnergy[n][m][0].item()<<";";
            energy<<"energy_cpp("<<n+1<<","<<m+1<<",2) = "<<lfEnergy[n][m][1].item()<<";";
            energy<<"energy_cpp("<<n+1<<","<<m+1<<",3) = "<<lfEnergy[n][m][2].item()<<";";
            
            rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoS[n][m][0].item()<<";";
            rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoS[n][m][1].item()<<";";
            rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoS[n][m][2].item()<<";";
            
            rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoT[n][m][0].item()<<";";
            rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoT[n][m][1].item()<<";";
            rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoT[n][m][2].item()<<";";
            
            rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoU[n][m][0].item()<<";";
            rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoU[n][m][1].item()<<";";
            rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoU[n][m][2].item()<<";";
           
            rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoV[n][m][0].item()<<";";
            rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoV[n][m][1].item()<<";";
            rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoV[n][m][2].item()<<";";
           
            angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",1) = "<<lfAngleH[n][m][0].item()<<";";
            angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",2) = "<<lfAngleH[n][m][1].item()<<";";
            angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",3) = "<<lfAngleH[n][m][2].item()<<";";
           
            angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",1) = "<<lfAngleV[n][m][0].item()<<";";
            angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",2) = "<<lfAngleV[n][m][1].item()<<";";
            angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",3) = "<<lfAngleV[n][m][2].item()<<";";
           
            rate<<"rate_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRate[n][m][0].item()<<";";
            rate<<"rate_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRate[n][m][1].item()<<";";
            rate<<"rate_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRate[n][m][2].item()<<";";
           
            distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",1) = "<<lfDistortion[n][m][0].item()<<";";
            distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",2) = "<<lfDistortion[n][m][1].item()<<";";
            distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",3) = "<<lfDistortion[n][m][2].item()<<";";
        }
        energy<<std::endl;
        rhoS<<std::endl;
        rhoT<<std::endl;
        rhoU<<std::endl;
        rhoV<<std::endl;
        angleH<<std::endl;
        angleV<<std::endl;
        rate<<std::endl;
        distortion<<std::endl;
    }
    
    cout<<"The file pointer is not null right?"<< outputFileNamePointer <<endl;
     hdt.DoneEncoding();
    fclose(outputFileNamePointer);
    cout<<"I'm exiting, the rest is just bullshit"<<endl;
    inputLF.CloseLightField();

}

void ExtendBlock4D(Block4D_ &extendedBlock, ExtensionType extensionMethod, int extensionLength, char direction) {
    
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

void RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    int* Y_data = Y.data.data_ptr<int>();
    int* Cb_data = Cb.data.data_ptr<int>();
    int* Cr_data = Cr.data.data_ptr<int>();
    int* R_data = R.data.data_ptr<int>();
    int* G_data = G.data.data_ptr<int>();
    int* B_data = B.data.data_ptr<int>();
    for(int n = 0; n < R.size[0]*R.size[1]*R.size[2]*R.size[3]; n++) {
        double pixel =  0.299 * R_data[n] + 0.587 * G_data[n] + 0.114 * B_data[n];
        Y_data[n] = round(pixel);
        pixel = -0.16875 * R_data[n] -0.33126 * G_data[n] + 0.5 * B_data[n];
        Cb_data[n] = round(pixel) + (Scale + 1)/2;
        pixel = 0.5 * R_data[n] -0.41869 * G_data[n] -0.08131  * B_data[n];
        Cr_data[n] = round(pixel) + (Scale + 1)/2;
    }
}

void RGB2YCbCr_BT601_old(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    static const auto Y_weights = at::tensor({0.299, 0.587, 0.114}, at::kDouble).reshape({3, 1});  
    static const auto Cb_weights = at::tensor({-0.16875, -0.33126, 0.5}, at::kDouble).reshape({3, 1}); 
    static const auto Cr_weights = at::tensor({0.5, -0.41869, -0.08131}, at::kDouble).reshape({3, 1}); 
    static const int D = 1<<((int)log2(Scale+1)-8);
    static const int Y8bitBias = 0;
    static const int CbCr8bitBias = (1<<7);

    auto Ey = R.data.to(at::kDouble)/Scale * Y_weights[0] + G.data.to(at::kDouble)/Scale * Y_weights[1] + B.data.to(at::kDouble)/Scale * Y_weights[2];
    Y = ((255 * Ey + Y8bitBias) * D).round().to(at::kInt)/D;
    auto Ecb = R.data.to(at::kDouble)/Scale * Cb_weights[0] + G.data.to(at::kDouble)/Scale * Cb_weights[1] + Cb.data.to(at::kDouble)/Scale * Y_weights[2];
    Cb = ((255 * Ecb + CbCr8bitBias)*D).round().to(at::kInt)/D;
    auto Ecr = R.data.to(at::kDouble)/Scale * Cr_weights[0] + G.data.to(at::kDouble)/Scale * Cr_weights[1] + Cr.data.to(at::kDouble)/Scale * Y_weights[2];
    Cr = ((255 * Ecr + CbCr8bitBias)*D).round().to(at::kInt)/D; 
    Y.validPositions = R.validPositions;
    Cb.validPositions = R.validPositions;
    Cr.validPositions = R.validPositions;
}

void RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
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
