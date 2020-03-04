#ifndef ORB_CLASS_H
#define ORB_CLASS_H

#include <vector>

class Feature{
public:
	int level;
    int   x;
    int   y;     // Location
    double mag;
    double orien;  // calculated by atan2

    Feature(int level, int Px, int Py,double _mag,double degree);
	static bool compare(Feature a,Feature b);
};

class FingerPrint{
public:
	int	*data;
    int length;

    FingerPrint();
    void pushKey(int key);
    int getKey(int index);
	void clear();

};
#endif