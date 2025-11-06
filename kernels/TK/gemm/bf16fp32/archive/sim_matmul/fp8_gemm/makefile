CXX = hipcc
CXXFLAGS = -std=c++17 -O3

SRC = gemm_fp8.cpp
OBJ = $(SRC:.cpp=.o)

TARGET = gemm_fp8

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -Rpass-analysis=kernel-resource-usage -o $@

clean:
	rm -f $(OBJ) $(TARGET)
