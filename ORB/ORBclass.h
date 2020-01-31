#ifndef ORB_CLASS_H
#define ORB_CLASS_H

#include <vector>

class Feature{
public:
    int   x;
    int   y;     // Location
    double orien;  // calculated by atan2

    Feature(int Px, int Py, int degree);
};

class FingerPrint{
public:
    std::vector<int>   	list;
    int 			length;

    FingerPrint();
    void pushKey(int key);
    int getKey(int index);
};
#endif