#include "LightField/LightField.h"
#include <string.h>
#include <stdlib.h>
#include "IO/io.h"

/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/
LightField::LightField(std::string root_path,std::string pattern) {
    OpenLightFieldPPM_(root_path,pattern,'r');
    this->mNumberOfHorizontalViews = data.size(1);
    this->mNumberOfVerticalViews = data.size(0);
    this->mNumberOfViewLines = data.size(3);
    this->mNumberOfViewColumns = data.size(2);
}
LightField :: LightField(std::array<int,5> size){
    this->data = at::zeros({size[0],size[1],size[2],size[3],size[4]},at::kShort);
    mViewFileNamePrefix = NULL;
    mViewFileNameSuffix = NULL;
    mNumberOfHorizontalViews = 0;
    mNumberOfVerticalViews = 0;
    mNumberOfCacheHorizontalViews = size[1];
    mNumberOfCacheVerticalViews = size[0];
    this->mViewCache = new View * [mNumberOfCacheVerticalViews];
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
        mViewCache[index_t] = new View [mNumberOfCacheHorizontalViews];
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
            this->mViewCache[index_t][index_s].SetCacheSize(size[3]);
        }
    }

    
    
    mFirstCacheHorizontalView = 0;
    mFirstCacheVerticalView = 0;
    mHorizontalIndexOffset = 0;
    mVerticalIndexOffset = 0;
    mReadOrWriteLightField = 'r';
    mVerticalViewNumberOffset = 0;
    mHorizontalViewNumberOffset = 0;

}
LightField :: LightField(int numberOfCacheVerticalViews, int numberOfCacheHorizontalViews, int numberOfViewCacheLines) {
    
    mViewFileNamePrefix = NULL;
    mViewFileNameSuffix = NULL;
    mNumberOfHorizontalViews = 0;
    mNumberOfVerticalViews = 0;
    mNumberOfCacheHorizontalViews = numberOfCacheHorizontalViews;
    mNumberOfCacheVerticalViews = numberOfCacheVerticalViews;
    
    this->mViewCache = new View * [mNumberOfCacheVerticalViews];
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
        mViewCache[index_t] = new View [mNumberOfCacheHorizontalViews];
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
            this->mViewCache[index_t][index_s].SetCacheSize(numberOfViewCacheLines);
            
        }
    }

    
    
    mFirstCacheHorizontalView = 0;
    mFirstCacheVerticalView = 0;
    mHorizontalIndexOffset = 0;
    mVerticalIndexOffset = 0;
    mReadOrWriteLightField = 'r';
    mVerticalViewNumberOffset = 0;
    mHorizontalViewNumberOffset = 0;


}
LightField :: ~LightField() {
    
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
       delete [] mViewCache[index_t];
    }
    delete [] mViewCache;

}

void LightField :: OpenLightFieldPGM(char *viewFileNamePrefix, char *viewFileNameSuffix, int numberOfVerticalViews, int numberOfHorizontalViews, int numberOfVerticalDigits, int numberOfHorizontalDigits, char readOrWriteLightField) {
/*! opens lightfield pgm views for reading or writing */
    
    mViewFileNamePrefix = new char [strlen(viewFileNamePrefix)+1];
    mViewFileNameSuffix = new char [strlen(viewFileNameSuffix)+1];
    
    strcpy(mViewFileNamePrefix, viewFileNamePrefix);
    strcpy(mViewFileNameSuffix, viewFileNameSuffix);
    
    mNumberOfVerticalViews = numberOfVerticalViews;
    mNumberOfHorizontalViews = numberOfHorizontalViews;
    
    mNumberOfVerticalDigits = numberOfVerticalDigits;
    mNumberOfHorizontalDigits =  numberOfHorizontalDigits; 
    
    mReadOrWriteLightField = readOrWriteLightField;
    
    int endOfStrByte = 1;
    char *ViewFileName = new char[strlen(viewFileNamePrefix)+strlen(viewFileNameSuffix)+mNumberOfVerticalDigits+mNumberOfHorizontalDigits+1 + endOfStrByte];
    
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews && index_t < mNumberOfVerticalViews; index_t++) {
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews && index_s < mNumberOfHorizontalViews; index_s++) {
            
            FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
            //printf("Opening %s view file\n", ViewFileName);
            if(mReadOrWriteLightField == 'w') {
                mViewCache[index_t][index_s].mLines = mNumberOfViewLines;
                mViewCache[index_t][index_s].mColumns = mNumberOfViewColumns;
                mViewCache[index_t][index_s].mPGMScale = mPGMScale;
            }
            mViewCache[index_t][index_s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
        }
    }
    delete [] ViewFileName;
    if(mReadOrWriteLightField == 'r') {
        mNumberOfViewLines = mViewCache[0][0].mLines;
        mNumberOfViewColumns = mViewCache[0][0].mColumns;
        mPGMScale = mViewCache[0][0].mPGMScale;
    }
}


void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, char readOrWriteLightField ) {
    if(readOrWriteLightField == 'r'){
        this->data = io::read_collection(rootPath, pattern).to(torch::kInt16);
    }else{
        if(readOrWriteLightField == 'w'){
            io::write_collection(rootPath,this->data);
        }
    }
}

void LightField :: OpenLightFieldPPM(char *viewFileNamePrefix, char *viewFileNameSuffix, int numberOfVerticalViews, int numberOfHorizontalViews, int numberOfVerticalDigits, int numberOfHorizontalDigits, char readOrWriteLightField) {


    mViewType = 1;

    
    mViewFileNamePrefix = new char [strlen(viewFileNamePrefix)+1];
    mViewFileNameSuffix = new char [strlen(viewFileNameSuffix)+1];
    
    strcpy(mViewFileNamePrefix, viewFileNamePrefix);
    strcpy(mViewFileNameSuffix, viewFileNameSuffix);
    
    mNumberOfVerticalViews = numberOfVerticalViews;
    mNumberOfHorizontalViews = numberOfHorizontalViews;
    
    mNumberOfVerticalDigits = numberOfVerticalDigits;
    mNumberOfHorizontalDigits =  numberOfHorizontalDigits; 
    
    mReadOrWriteLightField = readOrWriteLightField;


    
    int endOfStrByte = 1;
    char *ViewFileName = new char[strlen(viewFileNamePrefix)+strlen(viewFileNameSuffix)+mNumberOfVerticalDigits+mNumberOfHorizontalDigits+1 + endOfStrByte];
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews && index_t < mNumberOfVerticalViews; index_t++) {
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews && index_s < mNumberOfHorizontalViews; index_s++) {
    
            FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);

            //printf("Opening %s view file\n", ViewFileName);
            if(mReadOrWriteLightField == 'w') {
                mViewCache[index_t][index_s].mLines = mNumberOfViewLines;
                mViewCache[index_t][index_s].mColumns = mNumberOfViewColumns;
                mViewCache[index_t][index_s].mPGMScale = mPGMScale;
            }


            mViewCache[index_t][index_s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
        }
    }
    
    delete [] ViewFileName;
    if(mReadOrWriteLightField == 'r') {
        mNumberOfViewLines = mViewCache[0][0].mLines;
        mNumberOfViewColumns = mViewCache[0][0].mColumns;
        mPGMScale = mViewCache[0][0].mPGMScale;
    }
}

void LightField :: CloseLightField() {
/*! closes all opened view files*/
    
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
            mViewCache[index_t][index_s].CloseViewFile();
        }
    }
    
    if(mViewFileNamePrefix != NULL) {
        delete [] mViewFileNamePrefix;
        mViewFileNamePrefix = NULL;
    }
    if(mViewFileNameSuffix != NULL) {
        delete [] mViewFileNameSuffix;
        mViewFileNameSuffix = NULL;
    }
        
    mNumberOfVerticalViews = 0;
    mNumberOfHorizontalViews = 0;
    
    mNumberOfVerticalDigits = 0;
    mNumberOfHorizontalDigits =  0;  
    
}

Block4D_ LightField::ReadBlock4DfromLightField_(std::array<int,5>size,std::array<int,5>position){
    at::Tensor blockData = this->data.index({at::indexing::Slice(position[0],position[0]+size[0]),
                                             at::indexing::Slice(position[1],position[1]+size[1]),
                                             at::indexing::Slice(position[2],position[2]+size[2]),
                                             at::indexing::Slice(position[3],position[3]+size[3]),
                                             at::indexing::Slice(position[4],position[4]+size[4])});
    
    
    
    return blockData;
}
void LightField :: ReadBlock4DfromLightField(Block4D *targetBlock, int position_t, int position_s, int position_v, int position_u, int component) {
/*! reads a 4 dimensional block from the lightfield at position (position_t,position_s,position_v,position_u). */
/*! position_t is the vertical view index; */
/*! position_s is the horizontal view index; */
/*! position_v is the line index; */
/*! position_u is the column index. */    

    //if((position_t >= mFirstCacheVerticalView)&&(position_t + targetBlock->mlength_t <= mFirstCacheVerticalView + mNumberOfCacheVerticalViews)&&(position_s >= mFirstCacheHorizontalView)&&(position_s +targetBlock->mlength_s <= mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews)) {
    if((position_t >= mFirstCacheVerticalView)&&((position_t + targetBlock->mlength_t < mNumberOfVerticalViews ? position_t + targetBlock->mlength_t : mNumberOfVerticalViews) <= mFirstCacheVerticalView + mNumberOfCacheVerticalViews)&&(position_s >= mFirstCacheHorizontalView)&&((position_s +targetBlock->mlength_s < mNumberOfHorizontalViews ? position_s +targetBlock->mlength_s : mNumberOfHorizontalViews ) <= mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews)) {
        //The requested block is inside the current cache of views. No need to open new ones
        for(int index_t = position_t-mFirstCacheVerticalView; index_t < position_t-mFirstCacheVerticalView+targetBlock->mlength_t; index_t++) {
            for(int index_s = position_s-mFirstCacheHorizontalView; index_s < position_s-mFirstCacheHorizontalView+targetBlock->mlength_s; index_s++) {
 		if((index_s + mFirstCacheHorizontalView < mNumberOfHorizontalViews)&&(index_t + mFirstCacheVerticalView < mNumberOfVerticalViews)) {
                    int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                    int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                    //mViewCache[v][u].GetBlock2D(targetBlock->mPixel[index_t][index_s],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u);
                    mViewCache[t][s].GetBlock2D(targetBlock->mPixel[index_t-position_t+mFirstCacheVerticalView][index_s-position_s+mFirstCacheHorizontalView],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u, component);
                }
            }
        }        
    }
    else {
	int endOfStrByte = 1;
        char *ViewFileName = new char[strlen(mViewFileNamePrefix)+strlen(mViewFileNameSuffix)+mNumberOfVerticalDigits+mNumberOfHorizontalDigits+1 + endOfStrByte];
        int numberOfVerticalViewsToRead;
        int numberOfHorizontalViewsToRead;
        int numberOfVerticalViewsToKeep;
        int numberOfHorizontalViewsToKeep;
        if(position_t + targetBlock->mlength_t > mFirstCacheVerticalView + mNumberOfCacheVerticalViews) {
            if(position_s + targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                //and mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
					numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
					numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if((index_t < numberOfVerticalViewsToRead)||(index_s < numberOfHorizontalViewsToRead))
							mViewCache[t][s].CloseViewFile();
                    }                   
                }
                
                mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews) mHorizontalIndexOffset = 0;
                mVerticalIndexOffset += numberOfVerticalViewsToRead;
                if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews) mVerticalIndexOffset = 0;
                
                
                for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                            if(mViewType == 0)
                                mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                            if(mViewType == 1)
                                mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                    }                   
                }
                for(int index_t = 0; index_t < numberOfVerticalViewsToKeep; index_t++) {
                    for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                            if(mViewType == 0)
                                mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                            if(mViewType == 1)
                                mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                    }                   
                }
            }
            else {
                if(position_s >= mFirstCacheHorizontalView) {
                    //new status after reading: mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                    numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                    for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            mViewCache[t][s].CloseViewFile();
                        }                   
                    }

                    mVerticalIndexOffset += numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews)
						mVerticalIndexOffset = 0;
                    
                    for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                  
                }
                else {
                    //new status after reading: mFirstCacheHorizontalView = position_s
                    //and mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                   
                    numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                    numberOfHorizontalViewsToRead = mFirstCacheHorizontalView - position_s;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                    mFirstCacheHorizontalView = position_s;
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if((index_t < numberOfVerticalViewsToRead)||(index_s >= numberOfHorizontalViewsToKeep))
				mViewCache[t][s].CloseViewFile();
                        }                   
                    }
                    
                    mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset < 0)
						mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;
                    mVerticalIndexOffset += numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews)
						mVerticalIndexOffset = 0;
                    
                    for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                    for(int index_t = 0; index_t < numberOfVerticalViewsToKeep; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
            }
        }
        else {
            if(position_t >= mFirstCacheVerticalView) {
                if(position_s +targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                    //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                    
                    numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                     for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            mViewCache[t][s].CloseViewFile();
                        }
                     }
                    
                    mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews)
						mHorizontalIndexOffset = 0;
                    
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
                else { //position_s < mFirstCacheHorizontalView
                    //new status after reading: mFirstCacheHorizontalView = position_s
                    
                    numberOfHorizontalViewsToRead = mFirstCacheHorizontalView - position_s;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheHorizontalView = position_s;
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset+numberOfHorizontalViewsToKeep)%mNumberOfHorizontalViews;
                            mViewCache[t][s].CloseViewFile();
                        }
                    }
                    
                    mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset < 0)
						mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;
                    
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
            }
            else {
                if(position_s +targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                    //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                    //and mFirstCacheVerticalView = position_t
                    numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                    numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                    mFirstCacheVerticalView = position_t;
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if((index_s < numberOfHorizontalViewsToRead)||(index_t >= numberOfVerticalViewsToKeep)) 
				mViewCache[t][s].CloseViewFile();
                        }                   
                    }
                    
                    mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset < 0)
						mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                    mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews)
						mHorizontalIndexOffset = mHorizontalIndexOffset%mNumberOfCacheHorizontalViews;
                    
                    for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                    for(int index_s = 0; index_s < numberOfHorizontalViewsToKeep; index_s++) {
                        for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }                     
                }
                else { 
                    if(position_s >= mFirstCacheHorizontalView) {
                        //and mFirstCacheVerticalView = position_t
                        numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                        if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
							numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                        numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                        mFirstCacheVerticalView = position_t;
                        for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                mViewCache[t][s].CloseViewFile();
                             }
                        }
                        
                        mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                        if(mVerticalIndexOffset < 0)
							mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                        
                        for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                            for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }
                       
                    }
                    else { //position_s < mFirstCacheHorizontalView
                        //new status after reading: mFirstCacheVerticalView = position_t
                        //new status after reading: mFirstCacheHorizontalView = position_s
                      
                        numberOfHorizontalViewsToRead =  mFirstCacheHorizontalView - position_s;
                        numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                        if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
							numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                        if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
							numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                        numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                        numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                        mFirstCacheHorizontalView = position_s;
                        mFirstCacheVerticalView = position_t;
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if((index_s >= numberOfHorizontalViewsToKeep)||(index_t >= numberOfVerticalViewsToKeep)) 
                                    mViewCache[t][s].CloseViewFile();
                            }                   
                        }

                        mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                        if(mVerticalIndexOffset < 0)
							mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                        mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                        if(mHorizontalIndexOffset < 0)
							mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;

                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            for(int index_t = numberOfVerticalViewsToRead; index_t < mNumberOfCacheVerticalViews; index_t++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }                     
                    }
                    
                }
            
            }
        }
        delete [] ViewFileName;
        
        for(int index_t = position_t-mFirstCacheVerticalView; index_t < position_t-mFirstCacheVerticalView+targetBlock->mlength_t; index_t++) {
            for(int index_s = position_s-mFirstCacheHorizontalView; index_s < position_s-mFirstCacheHorizontalView+targetBlock->mlength_s; index_s++) {
 		if((index_s + mFirstCacheHorizontalView < mNumberOfHorizontalViews)&&(index_t + mFirstCacheVerticalView < mNumberOfVerticalViews)) {
                    int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                    int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                    //mViewCache[v][u].GetBlock2D(targetBlock->mPixel[index_t][index_s],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u);
                    mViewCache[t][s].GetBlock2D(targetBlock->mPixel[index_t-position_t+mFirstCacheVerticalView][index_s-position_s+mFirstCacheHorizontalView],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u, component);
                }
            }
        }
        
    }
}
void LightField :: WriteBlock4DtoLightField(Block4D *targetBlock, int position_t, int position_s, int position_v, int position_u, int component) {
/*! writes a 4 dimensional block to the lightfield at position (position_t,position_s,position_v,position_u). */
/*! position_t is the vertical view index; */
/*! position_s is the horizontal view index; */
/*! position_v is the line index; */
/*! position_u is the column index. */    
    
    //if((position_t >= mFirstCacheVerticalView)&&(position_t + targetBlock->mlength_t <= mFirstCacheVerticalView + mNumberOfCacheVerticalViews)&&(position_s >= mFirstCacheHorizontalView)&&(position_s +targetBlock->mlength_s <= mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews)) {
    if((position_t >= mFirstCacheVerticalView)&&((position_t + targetBlock->mlength_t < mNumberOfVerticalViews ? position_t + targetBlock->mlength_t : mNumberOfVerticalViews) <= mFirstCacheVerticalView + mNumberOfCacheVerticalViews)&&(position_s >= mFirstCacheHorizontalView)&&((position_s +targetBlock->mlength_s < mNumberOfHorizontalViews ? position_s +targetBlock->mlength_s : mNumberOfHorizontalViews ) <= mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews)) {
        //The requested block is inside the current cache of views. No need to open new ones
        for(int index_t = position_t-mFirstCacheVerticalView; index_t < position_t-mFirstCacheVerticalView+targetBlock->mlength_t; index_t++) {
            for(int index_s = position_s-mFirstCacheHorizontalView; index_s < position_s-mFirstCacheHorizontalView+targetBlock->mlength_s; index_s++) {
 		if((index_s + mFirstCacheHorizontalView < mNumberOfHorizontalViews)&&(index_t + mFirstCacheVerticalView < mNumberOfVerticalViews)) {
                    int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                    int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                    //mViewCache[v][u].PutBlock2D(targetBlock->mPixel[index_t][index_s],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u);
                    mViewCache[t][s].PutBlock2D(targetBlock->mPixel[index_t-position_t+mFirstCacheVerticalView][index_s-position_s+mFirstCacheHorizontalView],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u, component);
                }
            }
        }        
    }
    else {
	int endOfStrByte = 1;
        char *ViewFileName = new char[strlen(mViewFileNamePrefix)+strlen(mViewFileNameSuffix)+mNumberOfVerticalDigits+mNumberOfHorizontalDigits+1 + endOfStrByte];
        int numberOfVerticalViewsToRead;
        int numberOfHorizontalViewsToRead;
        int numberOfVerticalViewsToKeep;
        int numberOfHorizontalViewsToKeep;
        if(position_t + targetBlock->mlength_t > mFirstCacheVerticalView + mNumberOfCacheVerticalViews) {
            if(position_s + targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                //and mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
					numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
					numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if((index_t < numberOfVerticalViewsToRead)||(index_s < numberOfHorizontalViewsToRead))
                            mViewCache[t][s].CloseViewFile();
                    }                   
                }
                
                mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews)
					mHorizontalIndexOffset = 0;
                mVerticalIndexOffset += numberOfVerticalViewsToRead;
                if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews)
					mVerticalIndexOffset = 0;
                
                
                for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                            if(mViewType == 0)
                                mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                            if(mViewType == 1)
                                mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                   }                   
                }
                for(int index_t = 0; index_t < numberOfVerticalViewsToKeep; index_t++) {
                    for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                        int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                        int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                        if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                            if(mViewType == 0)
                                mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                            if(mViewType == 1)
                                mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                    }                   
                }
            }
            else {
                if(position_s >= mFirstCacheHorizontalView) {
                    //new status after reading: mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                    numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                    for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            mViewCache[t][s].CloseViewFile();
                        }                   
                    }

                    mVerticalIndexOffset += numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews)
						mVerticalIndexOffset = 0;
                    
                    for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                  
                }
                else {
                    //new status after reading: mFirstCacheHorizontalView = position_s
                    //and mFirstCacheVerticalView = position_t + targetBlock->mlength_t - mNumberOfCacheVerticalViews
                   
                    numberOfVerticalViewsToRead = position_t + targetBlock->mlength_t - mFirstCacheVerticalView - mNumberOfCacheVerticalViews;
                    numberOfHorizontalViewsToRead = mFirstCacheHorizontalView - position_s;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheVerticalView += numberOfVerticalViewsToRead;
                    mFirstCacheHorizontalView = position_s;
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if((index_t < numberOfVerticalViewsToRead)||(index_s >= numberOfHorizontalViewsToKeep))
				mViewCache[t][s].CloseViewFile();
                        }                   
                    }
                    
                    mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset < 0)
						mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;
                    mVerticalIndexOffset += numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset >= mNumberOfCacheVerticalViews)
						mVerticalIndexOffset = 0;
                    
                    for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                    for(int index_t = 0; index_t < numberOfVerticalViewsToKeep; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
            }
        }
        else {
            if(position_t >= mFirstCacheVerticalView) {
                if(position_s +targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                    //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                    
                    numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                     for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            mViewCache[t][s].CloseViewFile();
                        }
                     }
                    
                    mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews)
						mHorizontalIndexOffset = 0;
                    
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
                else { //position_s < mFirstCacheHorizontalView
                    //new status after reading: mFirstCacheHorizontalView = position_s
                    
                    numberOfHorizontalViewsToRead = mFirstCacheHorizontalView - position_s;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    mFirstCacheHorizontalView = position_s;
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset+numberOfHorizontalViewsToKeep)%mNumberOfHorizontalViews;
                            mViewCache[t][s].CloseViewFile();
                        }
                    }
                    
                    mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset < 0)
						mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;
                    
                    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                }
            }
            else {
                if(position_s +targetBlock->mlength_s > mFirstCacheHorizontalView + mNumberOfCacheHorizontalViews) {
                    //new status after reading: mFirstCacheHorizontalView = position_s +targetBlock->mlength_s - mNumberOfCacheHorizontalViews
                    //and mFirstCacheVerticalView = position_t
                    numberOfHorizontalViewsToRead = position_s + targetBlock->mlength_s - mFirstCacheHorizontalView - mNumberOfCacheHorizontalViews;
                    numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                    if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
						numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                    if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
						numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                    numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                    numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                    mFirstCacheHorizontalView += numberOfHorizontalViewsToRead;
                    mFirstCacheVerticalView = position_t;
                    for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if((index_s < numberOfHorizontalViewsToRead)||(index_t >= numberOfVerticalViewsToKeep))
				mViewCache[t][s].CloseViewFile();
                        }                   
                    }
                    
                    mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                    if(mVerticalIndexOffset < 0)
						mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                    mHorizontalIndexOffset += numberOfHorizontalViewsToRead;
                    if(mHorizontalIndexOffset >= mNumberOfCacheHorizontalViews)
						mHorizontalIndexOffset = mHorizontalIndexOffset%mNumberOfCacheHorizontalViews;
                    
                    for(int index_s = numberOfHorizontalViewsToKeep; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                        for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }
                    for(int index_s = 0; index_s < numberOfHorizontalViewsToKeep; index_s++) {
                        for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                            //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                            int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                            int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                            if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                if(mViewType == 0)
                                    mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                if(mViewType == 1)
                                    mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                        }                   
                    }                     
                }
                else { 
                    if(position_s >= mFirstCacheHorizontalView) {
                        //and mFirstCacheVerticalView = position_t
                        numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                        if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
							numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                        numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                        mFirstCacheVerticalView = position_t;
                        for(int index_t = numberOfVerticalViewsToKeep; index_t < mNumberOfCacheVerticalViews; index_t++) {
                            for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                mViewCache[t][s].CloseViewFile();
                             }
                        }
                        
                        mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                        if(mVerticalIndexOffset < 0)
							mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                        
                        for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                            for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }
                       
                    }
                    else { //position_s < mFirstCacheHorizontalView
                        //new status after reading: mFirstCacheVerticalView = position_t
                        //new status after reading: mFirstCacheHorizontalView = position_s
                      
                        numberOfHorizontalViewsToRead =  mFirstCacheHorizontalView - position_s;
                        numberOfVerticalViewsToRead = mFirstCacheVerticalView - position_t;
                        if(numberOfHorizontalViewsToRead > mNumberOfCacheHorizontalViews)
							numberOfHorizontalViewsToRead = mNumberOfCacheHorizontalViews;
                        if(numberOfVerticalViewsToRead > mNumberOfCacheVerticalViews)
							numberOfVerticalViewsToRead = mNumberOfCacheVerticalViews;
                        numberOfHorizontalViewsToKeep = mNumberOfCacheHorizontalViews - numberOfHorizontalViewsToRead;
                        numberOfVerticalViewsToKeep = mNumberOfCacheVerticalViews - numberOfVerticalViewsToRead;
                        mFirstCacheHorizontalView = position_s;
                        mFirstCacheVerticalView = position_t;
                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if((index_s >= numberOfHorizontalViewsToKeep)||(index_t >= numberOfVerticalViewsToKeep))
									mViewCache[t][s].CloseViewFile();
                            }                   
                        }

                        mVerticalIndexOffset -= numberOfVerticalViewsToRead;
                        if(mVerticalIndexOffset < 0)
							mVerticalIndexOffset += mNumberOfCacheVerticalViews;
                        mHorizontalIndexOffset -= numberOfHorizontalViewsToRead;
                        if(mHorizontalIndexOffset < 0)
							mHorizontalIndexOffset += mNumberOfCacheHorizontalViews;

                        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
                            for(int index_t = 0; index_t < numberOfVerticalViewsToRead; index_t++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }
                        for(int index_s = 0; index_s < numberOfHorizontalViewsToRead; index_s++) {
                            for(int index_t = numberOfVerticalViewsToRead; index_t < mNumberOfCacheVerticalViews; index_t++) {
                                //FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView);
                                int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                                int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                                if(FindViewFileName(ViewFileName, index_t+mFirstCacheVerticalView, index_s+mFirstCacheHorizontalView))
                                    if(mViewType == 0)
                                        mViewCache[t][s].OpenViewFilePGM(ViewFileName, mReadOrWriteLightField);
                                    if(mViewType == 1)
                                        mViewCache[t][s].OpenViewFilePPM(ViewFileName, mReadOrWriteLightField);
                            }                   
                        }                     
                    }
                    
                }
            
            }
        }
        delete [] ViewFileName;
        
        for(int index_t = position_t-mFirstCacheVerticalView; index_t < position_t-mFirstCacheVerticalView+targetBlock->mlength_t; index_t++) {
            for(int index_s = position_s-mFirstCacheHorizontalView; index_s < position_s-mFirstCacheHorizontalView+targetBlock->mlength_s; index_s++) {
 		if((index_s + mFirstCacheHorizontalView < mNumberOfHorizontalViews)&&(index_t + mFirstCacheVerticalView < mNumberOfVerticalViews)) {
                    int s = (index_s+mHorizontalIndexOffset)%mNumberOfCacheHorizontalViews;
                    int t = (index_t+mVerticalIndexOffset)%mNumberOfCacheVerticalViews;
                    //mViewCache[v][u].PutBlock2D(targetBlock->mPixel[index_t][index_s],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u);
                    mViewCache[t][s].PutBlock2D(targetBlock->mPixel[index_t-position_t+mFirstCacheVerticalView][index_s-position_s+mFirstCacheHorizontalView],position_v,position_u,targetBlock->mlength_v,targetBlock->mlength_u, component);
                }
            }
        }
        
    }
}
void LightField :: WriteBlock4DtoLightField_(Block4D_ sourceBlock, std::array<int,5> position){
    std::cout<<sourceBlock.data.sizes()<<std::endl;
    std::cout<<this->data.sizes()<<std::endl;
    std::cout<<sourceBlock.data.size(3)<<std::endl;
    std::cout<<sourceBlock.data.dtype()<<std::endl;
    std::cout<<sourceBlock.data[0][0]<<std::endl;
    std::cout<<this->data.dtype()<<std::endl;
    at::Tensor a = this->data.index({at::indexing::Slice(position[0],position[0]+sourceBlock.data.size(0)),
                     at::indexing::Slice(position[1],position[1]+sourceBlock.data.size(1)),
                     at::indexing::Slice(position[2],position[2]+sourceBlock.data.size(2)),
                     at::indexing::Slice(position[3],position[3]+sourceBlock.data.size(3)),
                     at::indexing::Slice(position[4],position[4]+1)});
    a =sourceBlock.data;
    std::cout<<"Did this work?"<<std::endl;
    std::cout <<this->data[0][0][0][0][0]<<std::endl;
    std::cout<<a.sizes()<<std::endl;              
    std::cout<<a.size(3)<<std::endl;   
     std::cout<<sourceBlock.data.size(2)<<std::endl;         
     std::cout<<position[3]<<" "<<position[3]+sourceBlock.data.size(3)<<std::endl;
    this->data.index({at::indexing::Slice(position[0],position[0]+sourceBlock.data.size(0)),
                     at::indexing::Slice(position[1],position[1]+sourceBlock.data.size(1)),
                     at::indexing::Slice(position[2],position[2]+sourceBlock.data.size(2)),
                     at::indexing::Slice(position[3],position[3]+sourceBlock.data.size(3)),
                     position[4]}) = sourceBlock.data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()});
                    

                     std::cout<<"successfully did something!"<<std::endl;

}
int LightField :: FindViewFileName(char *viewFileName, int index_t, int index_s) {
/*! light field name conversion*/
    
    if((index_t >= mNumberOfVerticalViews) || (index_s >= mNumberOfHorizontalViews)) {
        return(0);
    }
    
    char *fileIndex = new char[mNumberOfVerticalDigits+mNumberOfHorizontalDigits+3];
    char *fileNameDigits = new char[mNumberOfVerticalDigits+mNumberOfHorizontalDigits+2];
        
    sprintf(fileIndex, "%d", index_s+mHorizontalViewNumberOffset);
    while(strlen(fileIndex) < mNumberOfVerticalDigits) {
        strcpy(fileNameDigits, "0");
        strcat(fileNameDigits, fileIndex);
        strcpy(fileIndex, fileNameDigits);
    }
    strcpy(viewFileName, mViewFileNamePrefix);
    strcat(viewFileName, fileIndex);
    strcat(viewFileName, "_");
    sprintf(fileIndex, "%d", index_t+mVerticalViewNumberOffset);
    while(strlen(fileIndex) < mNumberOfHorizontalDigits) {
        strcpy(fileNameDigits, "0");
        strcat(fileNameDigits, fileIndex);
        strcpy(fileIndex, fileNameDigits);
    }
    strcat(viewFileName, fileIndex);
    strcat(viewFileName, mViewFileNameSuffix);
        
    //delete [] viewFileNameindex_t;
    delete [] fileIndex;
    delete [] fileNameDigits;
    
    return(1);
   
}
void LightField :: SetViewVerbosity(char verbosity) {
/*! sets the verbosity of console messages */
    
    for(int index_t = 0; index_t < mNumberOfCacheVerticalViews; index_t++) {
        for(int index_s = 0; index_s < mNumberOfCacheHorizontalViews; index_s++) {
            mViewCache[index_t][index_s].mVerbosity = verbosity;
        }
    }
}
