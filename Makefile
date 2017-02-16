CXX	= g++
CXXFLAGS= -std=c++11
LD	= g++

example: example.cpp cat.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o example
