#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}
