#include <vector>
#include <assert.h> 
#include "ORBclass.h"

#define FP_LEN 256

Feature::Feature(int _level, int Px, int Py,double _mag, double degree){
    level=_level;
    x = Px;
    y = Py;
    mag = _mag;
    orien = degree;
    }

bool Feature::compare(Feature a,Feature b){
    return a.mag > b.mag;
}

FingerPrint::FingerPrint(){
	list.clear();
    length = 0;
    }

void FingerPrint::pushKey(int key){
	list.push_back(key);
	length++;
    }

int FingerPrint::getKey(int index){
	assert(length==256);
	assert(index<FP_LEN);
	int ans = list[index];
	return ans;
    }

