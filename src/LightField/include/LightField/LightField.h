#include <torch/torch.h>
#include "View.h"
#include "Block4D.h"
#include "Block4D_.h"
#include <array>



#ifndef LIGHTFIELD_H
#define LIGHTFIELD_H


class LightField {
public:    
    LightField() = default;
    void OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize);
    LightField(std::string rooth_path,std::string pattern);
    LightField(int numberOfCacheVerticalViews, int numberOfCacheHorizontalViews, int numberOfViewCacheLines);
    LightField(std::array<int64_t,5> sizes);
    int preSlantTan = 0;
    char *mViewFileNamePrefix;          /*!< lightfield view name: <mViewFileNamePrefix>_<horizontal index>_<vertical index>_<mViewFilenameSuffix>.pgm */
    char *mViewFileNameSuffix;          /*!< lightfield view name: <mViewFileNamePrefix>_<horizontal index>_<vertical index>_<mViewFilenameSuffix>.pgm */
    int mNumberOfHorizontalDigits;      /*!< number of digits used to represent the horizontal index in the view file names */
    int mNumberOfVerticalDigits;        /*!< number of digits used to represent the vertical index in the view file names */
    int mNumberOfHorizontalViews;      /*!< total number of horizontal views of the lightfield */
    int mNumberOfVerticalViews;         /*!< total number of vertical views of the lightfield */
    at::Tensor data;                  /*!< pointer to the two dimensional circular separable cache of views*/
    View **mViewCache;                  /*!< pointer to the two dimensional circular separable cache of views*/

    int mNumberOfCacheHorizontalViews;  /*!< number of horizontal views of the lightfield in the cache */
    int mNumberOfCacheVerticalViews;    /*!< number of vertical views of the lightfield in the cache */
    int mFirstCacheHorizontalView;      /*!< horizontal index of the first cached view */
    int mFirstCacheVerticalView;        /*!< vertical index of the first cached view */
    int mHorizontalIndexOffset;         /*!< position of the first horizontal view in the circular cache */
    int mVerticalIndexOffset;           /*!< position of the first vertical view in the circular cache */
    char mReadOrWriteLightField;        /*!< read or write flag */
    int mNumberOfViewLines;             /*!< vertical resolution of each view */
    int mNumberOfViewColumns;           /*!< horizontal resolution of each view */
    int mPGMScale;                      /*!< scale of the pgm files*/
    int mVerticalViewNumberOffset;      /*!< number of vertical views to skip */
    int mHorizontalViewNumberOffset;    /*!< number of horizontal views to skip */
    int mViewType;                      /*!< mViewType = 0 -> PGM, mViewType = 1 -> PPM */    
    LightField(int k_size, int l_size, int u_size, int v_size);
    ~LightField();
    void OpenLightFieldPGM(char *viewFileNamePrefix, char *viewFileNameSuffix, int numberOfVerticalViews, int numberOfHorizontalViews, int numberOfVerticalDigits, int numberOfHorizontalDigits, char readOrWriteLightField);
    void OpenLightFieldPPM(char *viewFileNamePrefix, char *viewFileNameSuffix, int numberOfVerticalViews, int numberOfHorizontalViews, int numberOfVerticalDigits, int numberOfHorizontalDigits, char readOrWriteLightField);
    void OpenLightFieldPPM_(std::string path,std::string pattern,char readOrWriteLightField);
    void CloseLightField();

    void ReadBlock4DfromLightField(Block4D *targetBlock, int position_t, int position_s, int position_v, int position_u, int component=0);
    Block4D_ ReadBlock4DfromLightField_(std::array<int64_t,4> size, std::array<int64_t,4> position_t,int64_t channel );
    void WriteBlock4DtoLightField(Block4D *targetBlock, int position_t, int position_s, int position_v, int position_u, int component=0);
    void WriteBlock4DtoLightField_(Block4D_ sourcelock, std::array<int64_t,5> position);
    int FindViewFileName(char *viewFileName, int index_t, int index_s);
    void SetViewVerbosity(char verbosity);

};

#endif

