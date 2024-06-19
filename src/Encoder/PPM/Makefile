CXX = g++
CXXFLAGS = -g -Wall -O3 -I ../ -I ../../
CPPFLAGS = 

LDFLAGS = 

LDLIBS = -lm

SOURCES = ../../Matrix.cpp ../../Block4D.cpp ../../MultiscaleTransform.cpp ../../View.cpp ../../LightField.cpp ../../ProbabilityModel.cpp ../../ABACoder.cpp ../TransformPartition.cpp ../Hierarchical4DEncoder.cpp encoder.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: compile clean all

compile: encoder 
clean:
	$(RM) $(RMFLAGS) $(OBJECTS) encoder
all: clean compile
 
encoder: LINK.o = $(LINK.cc)
encoder: $(OBJECTS)
