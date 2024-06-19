CXX = g++
CXXFLAGS = -g -Wall -O3 -I ../ -I ../../
CPPFLAGS = 

LDFLAGS = 

LDLIBS = -lm

SOURCES = ../../Matrix.cpp ../../Block4D.cpp ../../MultiscaleTransform.cpp ../../View.cpp ../../LightField.cpp ../../ProbabilityModel.cpp ../../ABADecoder.cpp ../PartitionDecoder.cpp ../Hierarchical4DDecoder.cpp decoder.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: compile clean all

compile: decoder 
clean:
	$(RM) $(RMFLAGS) $(OBJECTS) decoder
all: clean compile

decoder: LINK.o = $(LINK.cc)
decoder: $(OBJECTS)
