#include "primary_classes.h"

// Constructor implementation
EdgeInfo::EdgeInfo(float foo, int A) : F(foo), a(A) {
}

// Method to get the capacity
float EdgeInfo::getCapacity() const {
    return F;
}

// Method to get the attempt number
int EdgeInfo::getAttemptNumber() const {
    return a;
}

// Method to set capacity
void EdgeInfo::setCapacity(float new_c){
    EdgeInfo::F = new_c;
}
