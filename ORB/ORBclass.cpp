#include <vector>
#include <assert.h> 
#include "ORBclass.h"

#define FP_LEN 256

Feature::Feature(int Px, int Py, int degree){
    x = Px;
    y = Py;
    orien = degree;
    }

FingerPrint::FingerPrint(){
    length = 0;
    }

void FingerPrint::pushKey(int key){
	assert(length<FP_LEN);
	assert(key==0 || key==1);
	if(length%8==0) list.push_back(key);
	else            list[length/8] = list[length/8]<<1 + key;
	   
	length++;
    }

int FingerPrint::getKey(int index){
	assert(index<FP_LEN);
	int listInd = index/32;
	int bitInd 	= 31-index%32;
	assert(bitInd > 0);
	int ans = (list[listInd]>>bitInd) & 1;
	assert(ans==0 || ans ==1);
	return ans;
    }

