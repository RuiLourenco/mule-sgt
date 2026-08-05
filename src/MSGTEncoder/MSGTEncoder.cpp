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
#include <cctype>

using namespace std;
using namespace filesystem;


bool is_all_whitespace(const std::string& str) {
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return std::isspace(c);
    });
}

std::string g_outputFileName = "";

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
    int preSlantTan = 0;
    SearchMethodType searchMethod = SearchMethodType::REFINE_STRUCTURE_TENSOR;
    bool searchMethodSet = false;
    double logdetAngleStep = 1.0;
    double gridSearchAngleStep = 1.0;
    double refineStructureTensorRange = 10.0;
    double refineStructureTensorStep = 0.5;
    double refineGridSearchInitialStep = 1.0;
    double refineGridSearchRange = 0.9;
    double refineGridSearchStep = 0.1;

    void setMethod(SearchMethodType method) {
        if (searchMethodSet) {
            std::cerr << "Error: Multiple search methods specified in configuration file!" << std::endl;
            exit(1);
        }
        searchMethod = method;
        searchMethodSet = true;
    }
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
        if(!command.compare("-preSlantTan")){
            parametersFile>>preSlantTan;
        }
        if(!command.compare("-structure_tensor") || !command.compare("-structure-tensor")){
            setMethod(SearchMethodType::STRUCTURE_TENSOR);
        }
        if(!command.compare("-logdet")){
            setMethod(SearchMethodType::LOGDET);
            auto pos = parametersFile.tellg();
            std::string peekToken;
            if (parametersFile >> peekToken) {
                if (!peekToken.empty() && peekToken[0] != '-') {
                    logdetAngleStep = std::stod(peekToken);
                } else {
                    parametersFile.seekg(pos);
                }
            }
        }
        if(!command.compare("-grid_search") || !command.compare("-grid-search")){
            setMethod(SearchMethodType::GRID_SEARCH);
            auto pos = parametersFile.tellg();
            std::string peekToken;
            if (parametersFile >> peekToken) {
                if (!peekToken.empty() && peekToken[0] != '-') {
                    gridSearchAngleStep = std::stod(peekToken);
                } else {
                    parametersFile.seekg(pos);
                }
            }
        }
        if(!command.compare("-covariance")){
            setMethod(SearchMethodType::COVARIANCE);
        }
        if(!command.compare("-all_heuristics") || !command.compare("-all-heuristics")){
            setMethod(SearchMethodType::ALL_HEURISTICS);
        }
        if(!command.compare("-zero")){
            setMethod(SearchMethodType::ZERO);
        }
        if(!command.compare("-refine_structure_tensor") || !command.compare("-refine-structure-tensor")){
            setMethod(SearchMethodType::REFINE_STRUCTURE_TENSOR);
            std::string peekToken;
            int paramsRead = 0;
            while (paramsRead < 2) {
                auto pos = parametersFile.tellg();
                if (parametersFile >> peekToken) {
                    if (!peekToken.empty() && peekToken[0] != '-') {
                        if (paramsRead == 0) refineStructureTensorRange = std::stod(peekToken);
                        if (paramsRead == 1) refineStructureTensorStep = std::stod(peekToken);
                        paramsRead++;
                    } else {
                        parametersFile.seekg(pos);
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        if(!command.compare("-refine_grid_search") || !command.compare("-refine-grid-search")){
            setMethod(SearchMethodType::REFINE_GRID_SEARCH);
            std::string peekToken;
            int paramsRead = 0;
            while (paramsRead < 3) {
                auto pos = parametersFile.tellg();
                if (parametersFile >> peekToken) {
                    if (!peekToken.empty() && peekToken[0] != '-') {
                        if (paramsRead == 0) refineGridSearchInitialStep = std::stod(peekToken);
                        if (paramsRead == 1) refineGridSearchRange = std::stod(peekToken);
                        if (paramsRead == 2) refineGridSearchStep = std::stod(peekToken);
                        paramsRead++;
                    } else {
                        parametersFile.seekg(pos);
                        break;
                    }
                } else {
                    break;
                }
            }
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
    cout<<"Pre Slant Tan = "<<preSlantTan<<endl;
    cout<<"Search Method Enum = "<<static_cast<int>(searchMethod)<<endl;
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
        ("pre-slant-tan", po::value<int>(&par.preSlantTan), "Pre Slant Tangent")
        ("structure-tensor", po::bool_switch(), "Structure Tensor Heuristic")
        ("logdet", po::value<std::vector<double>>()->multitoken(), "Logdet Heuristic [angleStep]")
        ("grid-search", po::value<std::vector<double>>()->multitoken(), "Grid Search Heuristic [angleStep]")
        ("covariance", po::bool_switch(), "Covariance Heuristic")
        ("all-heuristics", po::bool_switch(), "All Heuristics")
        ("zero", po::bool_switch(), "Zero Heuristic")
        ("refine-structure-tensor", po::value<std::vector<double>>()->multitoken(), "Refine Structure Tensor [refinementRange refinementStep]")
        ("refine-grid-search", po::value<std::vector<double>>()->multitoken(), "Refine Grid Search [initialStep refinementRange refinementStep]")
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
    
    // Process search methods
    std::vector<std::string> searchMethodFlags = {
        "structure-tensor", "logdet", "grid-search", "covariance", "all-heuristics", "zero", "refine-structure-tensor", "refine-grid-search"
    };
    int setMethods = 0;
    for (const auto& flag : searchMethodFlags) {
        if (vm.count(flag)) {
            if (vm[flag].value().type() == typeid(bool)) {
                if (vm[flag].as<bool>()) setMethods++;
            } else {
                setMethods++;
            }
        }
    }
    if (setMethods > 1) {
        throw std::logic_error("Conflicting options: Multiple search methods specified.");
    }

    if (vm.count("structure-tensor") && vm["structure-tensor"].as<bool>()) par.setMethod(SearchMethodType::STRUCTURE_TENSOR);
    if (vm.count("logdet")) {
        par.setMethod(SearchMethodType::LOGDET);
        auto data = vm["logdet"].as<std::vector<double>>();
        if (data.size() > 0) par.logdetAngleStep = data[0];
    }
    if (vm.count("grid-search")) {
        par.setMethod(SearchMethodType::GRID_SEARCH);
        auto data = vm["grid-search"].as<std::vector<double>>();
        if (data.size() > 0) par.gridSearchAngleStep = data[0];
    }
    if (vm.count("covariance") && vm["covariance"].as<bool>()) par.setMethod(SearchMethodType::COVARIANCE);
    if (vm.count("all-heuristics") && vm["all-heuristics"].as<bool>()) par.setMethod(SearchMethodType::ALL_HEURISTICS);
    if (vm.count("zero") && vm["zero"].as<bool>()) par.setMethod(SearchMethodType::ZERO);
    if (vm.count("refine-structure-tensor")) {
        par.setMethod(SearchMethodType::REFINE_STRUCTURE_TENSOR);
        auto data = vm["refine-structure-tensor"].as<std::vector<double>>();
        if (data.size() > 0) par.refineStructureTensorRange = data[0];
        if (data.size() > 1) par.refineStructureTensorStep = data[1];
    }
    if (vm.count("refine-grid-search")) {
        par.setMethod(SearchMethodType::REFINE_GRID_SEARCH);
        auto data = vm["refine-grid-search"].as<std::vector<double>>();
        if (data.size() > 0) par.refineGridSearchInitialStep = data[0];
        if (data.size() > 1) par.refineGridSearchRange = data[1];
        if (data.size() > 2) par.refineGridSearchStep = data[2];
    }

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


    

    Block4D_ yBlock,cbBlock,crBlock; 

    


 
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
        while (exists(path + filename)) {
            path = originalPath + "-" + std::to_string(counter)+"/";
            counter++;
        }
    
        std::cout<<filesystem::absolute(path)<<std::endl;
        std::cout<<filesystem::absolute(path+filename)<<std::endl;
        create_directories(path);
        par.outputFileName = path + filename;
        infoPath =  path + "info.json";
    }
   
    g_outputFileName = par.outputFileName;


    LightField inputLF;
    string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    inputLF.OpenLightFieldPPM_(par.inputDirectory,pattern,par.firstView,par.viewSize);  
    std::cout<<"LightField Size: "<<inputLF.data.sizes()<<std::endl;     
    inputLF.slantLightField(par.preSlantTan);
    std::cout<<"LightField Size: "<<inputLF.data.sizes()<<std::endl;  
    std::array<int64_t,2> stride = {2,3};
    // inputLF.OpenLightFieldPPM_("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant-st-fixed/set2_slanted", "", 'w',{0,2} , stride);
    // std::cout << "Press Enter to continue..." << std::endl;
    // std::cin.get();
    write_tensor(inputLF.data.index({inputLF.data.size(0)/2,at::indexing::Slice(),inputLF.data.size(2)/2,at::indexing::Slice(),0}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant-st-fixed/results/Set2/eval/epi.png");
    inputLF.computeTopHalfGradients();

    //inputLF.computeBottomHalfGradients();
    //inputLF.computeGradients();
    //inputLF.changePadding("Fill", 999999.0);
    // write_tensor(inputLF.gradients.index({4,4,at::indexing::Slice(),at::indexing::Slice(),2}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/v.png");
    // write_tensor(inputLF.gradients.index({4,4,at::indexing::Slice(),at::indexing::Slice(),1}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/s.png");
    // write_tensor(inputLF.gradients.index({4,4,at::indexing::Slice(),at::indexing::Slice(),0}),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/t.png");

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

    BigEndianSignedIntegerWrite(par.preSlantTan, 1, outputFileNamePointer);
    std::cout<<"Pre-Slant Tangent: "<<par.preSlantTan<<std::endl;

  

    //writes the bit precision of each component of the pixels of the views
    BigEndianUnsignedIntegerWrite(inputLF.mPGMScale, 2, outputFileNamePointer);


    //cout<<"mPGM scale = "<<inputLF.mPGMScale<<endl;
    std::vector<CodingPartitionInfo> codingPartitionInfos;
    TransformPartition tp(par.minPartitionSize,hdt,par.disparityRange,par.transformGain,par.searchMethod,
                          par.logdetAngleStep, par.gridSearchAngleStep, par.refineStructureTensorRange,
                          par.refineStructureTensorStep, par.refineGridSearchInitialStep,
                          par.refineGridSearchRange, par.refineGridSearchStep);
    tp.mEntropyCoder.StartEncoder(outputFileNamePointer);

    std::array<double,3> error = {0,0,0};
    bool second_half = false;
    double size = 0;
    for(int verticalView = 0; verticalView < inputLF.data.size(0); verticalView += par.maxPartitionSize[0]) {
        for(int horizontalView = 0; horizontalView < inputLF.data.size(1); horizontalView += par.maxPartitionSize[1]) {
            for(int viewLine = 0; viewLine < inputLF.data.size(2); viewLine += par.maxPartitionSize[2]) {
                for(int viewColumn = 0; viewColumn < inputLF.data.size(3); viewColumn += par.maxPartitionSize[3]) {
                    if(viewLine >= inputLF.secondHalfBias){
                        if(!inputLF.secondHalfGradientsComputed){
                            std::cout<<"Starting Bottom Half Gradient Computation"<<std::endl;
                            inputLF.changePadding("RepeatBorders");
                            inputLF.computeBottomHalfGradients();
                            //inputLF.changePadding("Fill", 999999.0);
                        }
                    }
                    if(true)
                        printf("transforming the 4D block at position (%d %d %d %d)\n", verticalView, horizontalView, viewLine, viewColumn);
                    std::array<int64_t,4> blockPosition = {verticalView,horizontalView,viewLine,viewColumn};

                    //if (verticalView != 0 || horizontalView != 0 || viewLine != 1152 || viewColumn != 0) continue;

                    Block4D_ rBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,0);
                    Block4D_ gBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,1);
                    Block4D_ bBlock = inputLF.ReadBlock4DfromLightField_(par.maxPartitionSize,blockPosition,2);
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
                        Block4D_ lfBlock;
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
                                                                                              

                        tp.mCodingPartitionInfo = CodingPartitionInfo(lfBlock.lightFieldPosition,lfBlock.size);
                        tp.mSpectralComponent = spectralComponent;
                        tp.RDoptimizeTransform_(lfBlock, par.Lambda);
                        tp.EncodePartition();
                        error[spectralComponent] += tp.mCodingPartitionInfo.getTotalDistortion();
                        size += tp.mCodingPartitionInfo.getTotalSize();
                        //std::cout<<"Size Channel "<<spectralComponent<<": "<<tp.mCodingPartitionInfo.getTotalSize()<<std::endl;
                        
                        std::cout<<"Encoding Successful!"<<std::endl;
                        // std::cout<<"Encoded"<<std::endl;
                        int sizeV = std::min(par.maxPartitionSize[2],inputLF.data.size(2)-viewLine);
                        int sizeH = std::min(par.maxPartitionSize[3],inputLF.data.size(3)-viewColumn);
                        
                        std::cout<<"Block Size: "<<sizeH<<" "<<sizeV<<std::endl;
                        
                        codingPartitionInfos.push_back(tp.mCodingPartitionInfo);
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

    CodingPartitionInfo::printVectorToJsonFile(codingPartitionInfos,infoPath);           

    //write_tensor(hdt.ignored[0][0],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/ignored.png");
    // std::ofstream energy;
    // std::ofstream rhoS;
    // std::ofstream rhoT;
    // std::ofstream rhoU;
    // std::ofstream rhoV;
    // std::ofstream angleH;
    // std::ofstream angleV;
    // std::ofstream rate;
    // std::ofstream distortion;

    // energy.open(path + "energy.m");
    // rhoS.open(path + "rhoS.m");
    // rhoT.open(path + "rhoT.m");
    // rhoU.open(path + "rhoU.m");
    // rhoV.open(path + "rhoV.m");
    // angleH.open(path + "angleH.m");
    // angleV.open(path + "angleV.m");
    // rate.open(path + "rate.m");
    // distortion.open(path + "distortion.m");
    // std::cout<<"Printing Images"<<endl;
    // energy<<"energy_cpp = zeros("<<lfEnergy.size(0)<<","<<lfEnergy.size(1)<<","<<lfEnergy.size(2)<<");"<<std::endl;
    // rhoS<<"rhoS_cpp = zeros("<< lfRhoS.size(0)<<","<< lfRhoS.size(1)<<","<< lfRhoS.size(2)<<");"<<std::endl;
    // rhoT<<"rhoT_cpp = zeros("<<lfRhoT.size(0)<<","<<lfRhoT.size(1)<<","<<lfRhoT.size(2)<<");"<<std::endl;
    // rhoU<<"rhoU_cpp = zeros("<<lfRhoU.size(0)<<","<<lfRhoU.size(1)<<","<<lfRhoU.size(2)<<");"<<std::endl;
    // rhoV<<"rhoV_cpp = zeros("<<lfRhoV.size(0)<<","<<lfRhoV.size(1)<<","<<lfRhoV.size(2)<<");"<<std::endl;
    // angleH<<"angleH_cpp = zeros("<<lfAngleH.size(0)<<","<<lfAngleH.size(1)<<","<<lfAngleH.size(2)<<");"<<std::endl;
    // angleV<<"angleV_cpp = zeros("<<lfAngleV.size(0)<<","<<lfAngleV.size(1)<<","<<lfAngleV.size(2)<<");"<<std::endl;
    // rate<<"rate_cpp = zeros("<<lfRate.size(0)<<","<<lfRate.size(1)<<","<<lfRate.size(2)<<");"<<std::endl;
    // distortion<<"distortion_cpp = zeros("<<lfDistortion.size(0)<<","<<lfDistortion.size(1)<<","<<lfDistortion.size(2)<<");"<<std::endl;
    // for(int n = 0; n < lfEnergy.size(0); n++){
    //     for(int m = 0; m < lfEnergy.size(1); m++){
    //         energy<<"energy_cpp("<<n+1<<","<<m+1<<",1) = "<<lfEnergy[n][m][0].item()<<";";
    //         energy<<"energy_cpp("<<n+1<<","<<m+1<<",2) = "<<lfEnergy[n][m][1].item()<<";";
    //         energy<<"energy_cpp("<<n+1<<","<<m+1<<",3) = "<<lfEnergy[n][m][2].item()<<";";
            
    //         rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoS[n][m][0].item()<<";";
    //         rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoS[n][m][1].item()<<";";
    //         rhoS<<"rhoS_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoS[n][m][2].item()<<";";
            
    //         rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoT[n][m][0].item()<<";";
    //         rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoT[n][m][1].item()<<";";
    //         rhoT<<"rhoT_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoT[n][m][2].item()<<";";
            
    //         rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoU[n][m][0].item()<<";";
    //         rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoU[n][m][1].item()<<";";
    //         rhoU<<"rhoU_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoU[n][m][2].item()<<";";
           
    //         rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRhoV[n][m][0].item()<<";";
    //         rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRhoV[n][m][1].item()<<";";
    //         rhoV<<"rhoV_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRhoV[n][m][2].item()<<";";
           
    //         angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",1) = "<<lfAngleH[n][m][0].item()<<";";
    //         angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",2) = "<<lfAngleH[n][m][1].item()<<";";
    //         angleH<<"angleH_cpp("<<n+1<<","<<m+1<<",3) = "<<lfAngleH[n][m][2].item()<<";";
           
    //         angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",1) = "<<lfAngleV[n][m][0].item()<<";";
    //         angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",2) = "<<lfAngleV[n][m][1].item()<<";";
    //         angleV<<"angleV_cpp("<<n+1<<","<<m+1<<",3) = "<<lfAngleV[n][m][2].item()<<";";
           
    //         rate<<"rate_cpp("<<n+1<<","<<m+1<<",1) = "<<lfRate[n][m][0].item()<<";";
    //         rate<<"rate_cpp("<<n+1<<","<<m+1<<",2) = "<<lfRate[n][m][1].item()<<";";
    //         rate<<"rate_cpp("<<n+1<<","<<m+1<<",3) = "<<lfRate[n][m][2].item()<<";";
           
    //         distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",1) = "<<lfDistortion[n][m][0].item()<<";";
    //         distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",2) = "<<lfDistortion[n][m][1].item()<<";";
    //         distortion<<"distortion_cpp("<<n+1<<","<<m+1<<",3) = "<<lfDistortion[n][m][2].item()<<";";
    //     }
    //     energy<<std::endl;
    //     rhoS<<std::endl;
    //     rhoT<<std::endl;
    //     rhoU<<std::endl;
    //     rhoV<<std::endl;
    //     angleH<<std::endl;
    //     angleV<<std::endl;
    //     rate<<std::endl;
    //     distortion<<std::endl;
    // }
    
    tp.mEntropyCoder.DoneEncoding();
    
    fclose(outputFileNamePointer);
    return 0;
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
    std::cout << "Y data type: " << Y.data.dtype() << std::endl;
    std::cout << "Cb data type: " << Cb.data.dtype() << std::endl;
    std::cout << "Cr data type: " << Cr.data.dtype() << std::endl;
    std::cout << "R data type: " << R.data.dtype() << std::endl;
    std::cout << "G data type: " << G.data.dtype() << std::endl;
    std::cout << "B data type: " << B.data.dtype() << std::endl;
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

// void RGB2YCbCr_BT601_old(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
//     static const std::array<double,3> Y_weights = {0.299, 0.587, 0.114});  
//     static const auto Cb_weights = at::tensor({-0.16875, -0.33126, 0.5}, at::kDouble).reshape({3, 1}); 
//     static const auto Cr_weights = at::tensor({0.5, -0.41869, -0.08131}, at::kDouble).reshape({3, 1}); 
//     static const int D = 1<<((int)log2(Scale+1)-8);
//     static const int Y8bitBias = 0;
//     static const int CbCr8bitBias = (1<<7);
//     R.data.to(at::kDouble);
//     G.data.to(at::kDouble);
//     B.data.to(at::kDouble);

//     auto Ey = R/Scale * Y_weights[0] + G.data.to(at::kDouble)/Scale * Y_weights[1] + B.data.to(at::kDouble)/Scale * Y_weights[2];
//     Y = ((255 * Ey + Y8bitBias) * D).round().to(at::kInt)/D;
//     auto Ecb = R.data.to(at::kDouble)/Scale * Cb_weights[0] + G.data.to(at::kDouble)/Scale * Cb_weights[1] + Cb.data.to(at::kDouble)/Scale * Y_weights[2];
//     Cb = ((255 * Ecb + CbCr8bitBias)*D).round().to(at::kInt)/D;
//     auto Ecr = R.data.to(at::kDouble)/Scale * Cr_weights[0] + G.data.to(at::kDouble)/Scale * Cr_weights[1] + Cr.data.to(at::kDouble)/Scale * Y_weights[2];
//     Cr = ((255 * Ecr + CbCr8bitBias)*D).round().to(at::kInt)/D; 
//     Y.validPositions = R.validPositions;
//     Cb.validPositions = R.validPositions;
//     Cr.validPositions = R.validPositions;
// }

void RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
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
