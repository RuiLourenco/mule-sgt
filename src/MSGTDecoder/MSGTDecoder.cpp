#include "LightField/LightField.h"
#include "LightField/Block4D.h"
#include "Decoder/Hierarchical4DDecoder.h"
#include "Decoder/PartitionDecoder.h"
#include <boost/program_options.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <array>
#include <algorithm>


using namespace std;

class DecoderParameters;
enum ExtensionType { REPEAT_LAST, CYCLIC, NONE};
enum ColorTransformType {BT601,YCOCG};
void ExtendBlock4D(Block4D &extendedblock, ExtensionType extensionMethod, int extensionLength, char direction);
void YCbCr2RGB_BT601(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Cb, Block4D const &Cr, int Scale);
void YCoCg2RGB(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Co, Block4D const &Cg, int Scale);
void RGB2YCbCr_BT601(Block4D &Y, Block4D &Cb, Block4D &Cr, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
unsigned long int BigEndianUnsignedIntegerRead(int precision, FILE *inputFilePointer);
long int BigEndianSignedIntegerRead(int precision, FILE *inputFilePointer) ;

class DecoderParameters {
public:
    array<int64_t, 2> viewSize;
    array<int64_t,2> firstView;
    array<int64_t,2> stride = {1,1};
    array<double,2> disparityRange;
    string outputDirectory;
    string inputFileName;
    string inputLightFieldDirectory;
    string configFile;
    bool isLenslet13x13;
    ExtensionType extensionMethod;
    ColorTransformType colorTransformType = BT601; 
    double transformGain;
    bool verbosity;
    void ReadConfigurationFile(string parametersFileName);
    void DisplayConfiguration();
};

void DecoderParameters :: ReadConfigurationFile(string parametersFileName) {
    std::ifstream parametersFile(parametersFileName);
    if( ! parametersFile ) {
        std::cerr << "Error opening input file" << std::endl ;
        return ;
    }
    while(!parametersFile.eof()) {
        std::string command;
        parametersFile >> command;
        if(command == "-nv") {
            parametersFile >> viewSize[0] >> viewSize[1];
        } else if(command == "-off") {
            parametersFile >> firstView[0] >> firstView[1];
        } else if(command == "-stride") {
            parametersFile >> stride[0] >> stride[1];
        } else if(command == "-lf") {
            parametersFile >> outputDirectory;
        } else if(command == "-i") {
            parametersFile >> inputFileName;
        } else if(command == "-d") {
            parametersFile >> inputLightFieldDirectory;
        } else if(command == "-lenslet13x13") {
            isLenslet13x13 = true;
        } else if(command == "-extension-repeat") {
        extensionMethod = REPEAT_LAST;
        } else if(command == "-extension-cyclic") {
        extensionMethod = CYCLIC;
        } else if(command == "-extension-none") {
        extensionMethod = NONE;
        } else if(command == "-t_gain") {
            parametersFile >> transformGain;
        } 
    }
}


void DecoderParameters :: DisplayConfiguration() {
    cout << "viewSize = " << viewSize[0] << " " << viewSize[1] << endl;
    cout << "firstView = " << firstView[0] << " " << firstView[1] << endl;
    cout << "stride = " << stride[0] << " " << stride[1] << endl;
    cout << "outputDirectory = " << outputDirectory << endl;
    cout << "inputFileName = " << inputFileName << endl;
    cout << "inputLightFieldDirectory = " << inputLightFieldDirectory << endl;
    cout << "isLenslet13x13 = " << isLenslet13x13 << endl;
    cout << "extensionMethod = " << extensionMethod << endl;
    cout << "transformGain = " << transformGain << endl;
    cout << "verbosity = " << verbosity << endl;
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

int readProgramOptions(int argc, char** argv, DecoderParameters& par){
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    ("config-file,c", po::value<string>(&par.configFile), "configuration file")
    ("num-views,v", po::value<vector<int64_t>>()->multitoken(), "view size")
    ("view-offset,b", po::value<vector<int64_t>>()->multitoken(), "first view")
    ("view-stride,s", po::value<vector<int64_t>>()->multitoken(), "view stride")
    ("output-dir,o", po::value<string>(&par.outputDirectory), "output directory")
    ("input-file,i", po::value<string>(&par.inputFileName), "input file")
    ("light-field-dir,d", po::value<string>(&par.inputLightFieldDirectory), "Path to Input Light Field Collection Directory")
    ("lenslet13x13", po::bool_switch()->default_value(false), "lenslet 13x13")
    ("extension-repeat", po::bool_switch()->default_value(false), "extension repeat")
    ("extension-cyclic", po::bool_switch()->default_value(false), "extension cyclic")
    ("extension-none", po::bool_switch()->default_value(false), "extension none")
    ("bt601", po::bool_switch()->default_value(false), "bt601")
    ("ycocg", po::bool_switch()->default_value(false), "ycocg")
    ("t_gain", po::value<double>(&par.transformGain), "transform gain")
    ("verbosity,V", po::bool_switch()->default_value(false), "verbosity");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv,desc),vm);
    if (vm.count("help")){
        cout << desc << endl;
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
    par.isLenslet13x13 = vm["lenslet13x13"].as<bool>();
    if(vm["bt601"].as<bool>()) par.colorTransformType = BT601;
    if(vm["ycocg"].as<bool>()) par.colorTransformType = YCOCG;
    if(vm["extension-repeat"].as<bool>()) par.extensionMethod = REPEAT_LAST;
    if(vm["extension-cyclic"].as<bool>()) par.extensionMethod = CYCLIC;
    if(vm["extension-none"].as<bool>()) par.extensionMethod = NONE;
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
    if(vm.count("view-stride")){
        std::vector<int64_t> data = vm["view-stride"].as<std::vector<int64_t>>();
        if (data.size() != 2) throw std::invalid_argument("view-stride must have 2 elements");
        std::copy(data.begin(), data.end(), par.stride.begin());
    }
    par.verbosity = vm["verbosity"].as<bool>();
    return 0;
}

int main(int argc, char **argv) {

    DecoderParameters par;
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
    par.DisplayConfiguration();
    
    // Load LightField from inputLightFieldDirectory if provided
    LightField reconstructedLF;
    if(!par.inputLightFieldDirectory.empty()) {
        string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
        reconstructedLF.OpenLightFieldPPM_(par.inputLightFieldDirectory, pattern, par.firstView, par.viewSize);
    }
    
    Hierarchical4DDecoder hdt;
    FILE *inputFileNamePointer;
    if((inputFileNamePointer = fopen(par.inputFileName.c_str(), "rb")) == NULL) {
        printf("Error: input file %s not found\n", par.inputFileName.c_str());
        exit(0);
    }
    std::cout<<"attempting to read superior bit plane"<<std::endl;
    hdt.mSuperiorBitPlane = BigEndianUnsignedIntegerRead(2, inputFileNamePointer);
    std::cout<<"SuperiorBitPlane: "<<hdt.mSuperiorBitPlane<<std::endl;

    //reads the maximum Partition sizes
    array<int64_t, 4> maxPartitionSize;
    for (int n = 0; n < 4; n++) {
        maxPartitionSize[n] = BigEndianUnsignedIntegerRead(2, inputFileNamePointer);
    }
    std::cout<<"MaxPartitionSize: "<<maxPartitionSize[0]<<" "<<maxPartitionSize[1]<<" "<<maxPartitionSize[2]<<" "<<maxPartitionSize[3]<<std::endl;
    array<int64_t,5> lfSize = {0,0,0,0,3};
    //reads the LightField Size
    for(int n = 0; n < 4; n ++ ){
        lfSize[n] = BigEndianUnsignedIntegerRead(2, inputFileNamePointer);
    }
    std::cout<<"LightField Size: "<<lfSize[0]<<" "<<lfSize[1]<<" "<<lfSize[2]<<" "<<lfSize[3]<<std::endl;

    //reads disparity range
    //Writes an Integer Encoded Disparity Range of the LF
    for(int n = 0; n < 2; n++) {
        par.disparityRange[n] =(double) BigEndianSignedIntegerRead( 3, inputFileNamePointer)/1000.0;
    }
    std::cout<<"DisparityRange: "<<par.disparityRange[0]<<" "<<par.disparityRange[1]<<std::endl;

    int PGMScale =BigEndianUnsignedIntegerRead( 2, inputFileNamePointer);
    std::cout<<"PGMScale: "<<PGMScale<<std::endl;
    hdt.StartDecoder(inputFileNamePointer);
    LightField outputLF(lfSize);
    outputLF.mPGMScale = PGMScale;
    Block4D lfBlock, yBlock,cbBlock,crBlock, rBlock, gBlock, bBlock; 



    array<int64_t,4> totalSize;
    for (int n = 0; n < 4; n++) {
        totalSize[n] = (lfSize[n]%maxPartitionSize[n] == 0) ? lfSize[n] : maxPartitionSize[n]*(lfSize[n]/maxPartitionSize[n]) + maxPartitionSize[n];
    }
    std::array<int64_t,4> extensionLength;

    for(int n = 0; n < 4; n++) {
        extensionLength[n] = lfSize[n] % maxPartitionSize[n];
    }
    PartitionDecoder pd(par.transformGain);

    at::Tensor lfEntropy = at::zeros(lfSize,at::kDouble);
    
    std::cout<<"LOOP WILL START"<<std::endl;
    for(int verticalView = 0; verticalView < lfSize[0]; verticalView+= maxPartitionSize[0]){
        for(int horizontalView = 0; horizontalView < lfSize[1]; horizontalView+=maxPartitionSize[1]){
            for(int viewLine = 0; viewLine <lfSize[2]; viewLine+=maxPartitionSize[2]){
                for(int viewColumn = 0; viewColumn < lfSize[3]; viewColumn+=maxPartitionSize[3]){
                    std::array<int64_t,4> blockPosition = {verticalView,horizontalView,viewLine,viewColumn};
                    
                    // Load reconstructed blocks from input light field if available
                    Block4D rReconstructedBlock, gReconstructedBlock, bReconstructedBlock;
                    Block4D yReconstructedBlock, cbReconstructedBlock, crReconstructedBlock;
                    
                    
                    rReconstructedBlock = reconstructedLF.ReadBlock4DfromLightField_(maxPartitionSize, blockPosition, 0);
                    gReconstructedBlock = reconstructedLF.ReadBlock4DfromLightField_(maxPartitionSize, blockPosition, 1);
                    bReconstructedBlock = reconstructedLF.ReadBlock4DfromLightField_(maxPartitionSize, blockPosition, 2);
                    std::cout<<rReconstructedBlock.data.index({0,0,at::indexing::Slice(0,10),at::indexing::Slice(0,10)})<<std::endl;
                    if(par.isLenslet13x13 == 1) {
                        //Correcting the values of the edge views of the light field by multiplying them by 4.
                        if(verticalView == 0) {
                            if(horizontalView == 0) {
                                rReconstructedBlock.Shift_UVPlane(2, 0, 0);
                                gReconstructedBlock.Shift_UVPlane(2, 0, 0);
                                bReconstructedBlock.Shift_UVPlane(2, 0, 0);
                            }

                            if((horizontalView + maxPartitionSize[1] >= reconstructedLF.data.size(1))&&(horizontalView <= reconstructedLF.data.size(1))) {
                                int lastViewH = reconstructedLF.data.size(1)-horizontalView-1;
                                rReconstructedBlock.Shift_UVPlane(2, 0, lastViewH);
                                gReconstructedBlock.Shift_UVPlane(2, 0, lastViewH);
                                bReconstructedBlock.Shift_UVPlane(2, 0, lastViewH);
                            }
                        }

                        if((verticalView + maxPartitionSize[0] >= reconstructedLF.data.size(0))&&(verticalView <= reconstructedLF.data.size(0))) {

                            int lastViewV = reconstructedLF.data.size(0)-verticalView-1;
                            if(horizontalView == 0) {
                                rReconstructedBlock.Shift_UVPlane(2, lastViewV, 0);
                                gReconstructedBlock.Shift_UVPlane(2, lastViewV, 0);
                                bReconstructedBlock.Shift_UVPlane(2, lastViewV, 0);
                            }
                            if((horizontalView + maxPartitionSize[1] >= reconstructedLF.data.size(1))&&(horizontalView <= reconstructedLF.data.size(1))) {
                                int lastViewH = reconstructedLF.data.size(1)-horizontalView-1;
                                rReconstructedBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                                gReconstructedBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                                bReconstructedBlock.Shift_UVPlane(2, lastViewV, lastViewH);
                            }
                        }
                    }
                    if(par.colorTransformType == BT601){
                        yReconstructedBlock = rReconstructedBlock.clone();
                        cbReconstructedBlock = gReconstructedBlock.clone();
                        crReconstructedBlock = bReconstructedBlock.clone(); 
                        RGB2YCbCr_BT601(yReconstructedBlock, cbReconstructedBlock, crReconstructedBlock, rReconstructedBlock, gReconstructedBlock, bReconstructedBlock, reconstructedLF.mPGMScale);

                    }
                    if(par.colorTransformType == YCOCG){
                        RGB2YCoCg(yReconstructedBlock, cbReconstructedBlock, crReconstructedBlock, rReconstructedBlock, gReconstructedBlock, bReconstructedBlock, reconstructedLF.mPGMScale);
                        std::cout<<" Completed YCOCG Color Transformation"<<std::endl;

                    }
                    
                    
                    for(int spectralComponent = 0; spectralComponent < 3; spectralComponent++){
                        if(par.verbosity > 0) cout<<"decoding spectral component "<<spectralComponent<<endl;
                        std::array<int64_t,5> currLfPosition = {verticalView,horizontalView,viewLine,viewColumn,spectralComponent};

                        if(par.verbosity > 0) 
                            printf("Decoding 4D block at position (%d %d %d %d)\n", verticalView, horizontalView, viewLine, viewColumn);
                        pd.mPartitionData = Block4D(maxPartitionSize,blockPosition,&outputLF);

                        // Get the appropriate reconstructed block for this spectral component
                        Block4D reconstructedBlock;
                        if(!par.inputLightFieldDirectory.empty()) {
                            if(spectralComponent == 0) {
                                reconstructedBlock = yReconstructedBlock;
                            } else if(spectralComponent == 1) {
                                reconstructedBlock = cbReconstructedBlock;
                            } else if(spectralComponent == 2) {
                                reconstructedBlock = crReconstructedBlock;
                            }
                        }
                        reconstructedBlock = reconstructedBlock - (outputLF.mPGMScale+1)/2;
                        hdt.RestartProbabilisticModel();
                        pd.DecodePartition(reconstructedBlock,hdt,par.disparityRange);
                                      
                
        
                        lfBlock = pd.mPartitionData;

                        for(int n = 0; n < 4; n++) {
                            if(blockPosition[n] + maxPartitionSize[n] > lfSize[n]) {

                                ExtendBlock4D(lfBlock,par.extensionMethod,extensionLength[n],n); 

                                cout<<"Block Extended!"<<endl;
                            } 
                        }
                        lfBlock = lfBlock + (outputLF.mPGMScale + 1)/2;
                        if(spectralComponent == 0) {
                            yBlock = lfBlock;
                        }
                        if(spectralComponent == 1) {
                            cbBlock = lfBlock;
                        }
                        if(spectralComponent == 2) {
                            crBlock = lfBlock;
                        }
                    }

                    if(par.colorTransformType == BT601){
                        rBlock = Block4D(yBlock.size,yBlock.lightFieldPosition,yBlock.lightField);
                        gBlock = Block4D(yBlock.size,yBlock.lightFieldPosition,yBlock.lightField);
                        bBlock = Block4D(yBlock.size,yBlock.lightFieldPosition,yBlock.lightField);
                        YCbCr2RGB_BT601( rBlock, gBlock, bBlock,yBlock, cbBlock, crBlock, outputLF.mPGMScale);
                    }
                    if(par.colorTransformType == YCOCG){
                        YCoCg2RGB(rBlock, gBlock, bBlock, yBlock, cbBlock, crBlock, outputLF.mPGMScale);
                    }
                    
                    if(par.isLenslet13x13 == 1) {
                    //Correcting the values of the edge views of the light field by multiplying them by 4.
                        if(verticalView == 0) {
                            if(horizontalView == 0) {
                                rBlock.Shift_UVPlane(-2, 0, 0);
                                gBlock.Shift_UVPlane(-2, 0, 0);
                                bBlock.Shift_UVPlane(-2, 0, 0);
                            }

                            if((horizontalView + maxPartitionSize[1] >= lfSize[1])&&(horizontalView <= lfSize[1])) {
                                int lastViewH = lfSize[1]-horizontalView-1;
                                rBlock.Shift_UVPlane(-2, 0, lastViewH);
                                gBlock.Shift_UVPlane(-2, 0, lastViewH);
                                bBlock.Shift_UVPlane(-2, 0, lastViewH);
                            }
                        }
                        if((verticalView + maxPartitionSize[0] >= lfSize[0])&&(verticalView <= lfSize[0])) {
                            int lastViewV = lfSize[0]-verticalView-1;
                            if(horizontalView == 0) {
                                rBlock.Shift_UVPlane(-2, lastViewV, 0);
                                gBlock.Shift_UVPlane(-2, lastViewV, 0);
                                bBlock.Shift_UVPlane(-2, lastViewV, 0);
                            }
                            if((horizontalView + maxPartitionSize[1] >= lfSize[1])&&(horizontalView <= lfSize[1])) {
                                int lastViewH = lfSize[1]-horizontalView-1;
                                rBlock.Shift_UVPlane(-2, lastViewV, lastViewH);
                                gBlock.Shift_UVPlane(-2, lastViewV, lastViewH);
                                bBlock.Shift_UVPlane(-2, lastViewV, lastViewH);
                            }
                        }
                        if(par.verbosity > 0) cout<<"Completed 13x13 luminosity adjustment"<<endl;
                    }
                    //std::cout<<"RED BLOCK: "<<rBlock.data.min().item()<<" "<<rBlock.data.max().item()<<std::endl;
                    //std::cout<<"GREEN BLOCK: "<<gBlock.data.min().item()<<" "<<gBlock.data.max().item()<<std::endl;
                    //std::cout<<"BLUE BLOCK: "<<bBlock.data.min().item()<<" "<<bBlock.data.max().item()<<std::endl;
                    rBlock.clip(0,outputLF.mPGMScale);
                    gBlock.clip(0,outputLF.mPGMScale);
                    bBlock.clip(0,outputLF.mPGMScale);
                    //std::cout<<"RED BLOCK: "<<rBlock.data.min().item()<<" "<<rBlock.data.max().item()<<std::endl;
                    //std::cout<<"GREEN BLOCK: "<<gBlock.data.min().item()<<" "<<gBlock.data.max().item()<<std::endl;
                    //std::cout<<"BLUE BLOCK: "<<bBlock.data.min().item()<<" "<<bBlock.data.max().item()<<std::endl;
                    outputLF.WriteBlock4DtoLightField_(rBlock,{blockPosition[0],blockPosition[1],blockPosition[2],blockPosition[3],0});
                    outputLF.WriteBlock4DtoLightField_(gBlock,{blockPosition[0],blockPosition[1],blockPosition[2],blockPosition[3],1});
                    outputLF.WriteBlock4DtoLightField_(bBlock,{blockPosition[0],blockPosition[1],blockPosition[2],blockPosition[3],2});
                }
            }
        }
    }






    hdt.DoneDecoding();
    std::error_code ec;
    if (std::filesystem::create_directories(par.outputDirectory, ec)) {
        std::cout << "Created new directory(ies) in path" << std::endl;
    } else if (ec) {
        std::cerr << "Error creating directories: " << ec.message() << std::endl;
    }
    outputLF.OpenLightFieldPPM_(par.outputDirectory, "", 'w', par.firstView, par.stride);
    fclose(inputFileNamePointer);
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

// void YCbCr2RGB_BT601_old(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Cb, Block4D const &Cr, int Scale) {
//     Block4D CbTemp = Cb - ((Scale+1)/2);
//     Block4D CrTemp = Cr - ((Scale+1)/2);
//     R = Y - CbTemp * 0.0000071525  +CrTemp * 1.4020 ;
//     G = Y -CbTemp * 0.34413  - CrTemp * 0.71414;
//     B = Y + 1.7720 * CbTemp - 0.000040249 * CrTemp;
// }

void YCoCg2RGB(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Co, Block4D const &Cg, int Scale) {
    auto CoTemp = Co- (Scale+1)/2;
    auto CgTemp = Cg - (Scale+1)/2;
    auto t = Y - (CgTemp.data.bitwise_right_shift(1));
    G = CgTemp + t;
    B.data = t - (CoTemp.data.bitwise_right_shift(1));
    R.data = B.data + CoTemp;          
}

void YCbCr2RGB_BT601(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Cb, Block4D const &Cr, int Scale) {
    int* Y_data = Y.data.data_ptr<int>();
    int* Cb_data = Cb.data.data_ptr<int>();
    int* Cr_data = Cr.data.data_ptr<int>();
    int* R_data = R.data.data_ptr<int>();
    int* G_data = G.data.data_ptr<int>();
    int* B_data = B.data.data_ptr<int>();
    for(int n = 0; n < R.size[0]*R.size[1]*R.size[2]*R.size[3]; n++) {
        double pixel_Y = Y_data[n];
        double pixel_Cb = Cb_data[n]-(Scale+1)/2;
        double pixel_Cr = Cr_data[n]-(Scale+1)/2;
        R_data[n] = round(pixel_Y - 0.0000071525 * pixel_Cb + 1.4020 * pixel_Cr);
        G_data[n] = round(pixel_Y -0.34413 * pixel_Cb - 0.71414 * pixel_Cr);
        B_data[n] = round(pixel_Y + 1.7720 * pixel_Cb - 0.000040249 * pixel_Cr);
    }
}

unsigned long int BigEndianUnsignedIntegerRead(int precision, FILE *inputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char input_byte;
    unsigned long int value = 0;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        int a = fread(&input_byte, 1, 1, inputFilePointer);
        value = (value << 8);
        value += input_byte;
    }
    return (value);
}

long int BigEndianSignedIntegerRead(int precision, FILE *inputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char input_byte;
    unsigned long int value = 0;
    int sign;
    int a = fread(&input_byte, 1, 1, inputFilePointer);
    if(input_byte) {
        sign = -1;
    }else{
        sign = 1;
    }
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        int a = fread(&input_byte, 1, 1, inputFilePointer);
        value = (value << 8);
        value += input_byte;
    }
    return (value*sign);
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