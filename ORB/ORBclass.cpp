#include <vector>
#include <assert.h> 
#include <stdlib.h>
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
	data = (int *)malloc(8*sizeof(int));
    length = 0;
    }

void FingerPrint::pushKey(int key){
	int bytepos = length/32;
	data[bytepos] = data[bytepos] * 2 +key; // key = 0 or 1
	length++;
    }
/*
int FingerPrint::getKey(int index){
	assert(length==FP_LEN);
	assert(index<FP_LEN);
	int ans = (int)list[index];
	return ans;
    }
*/
void FingerPrint::clear(){
	free(data);
}
