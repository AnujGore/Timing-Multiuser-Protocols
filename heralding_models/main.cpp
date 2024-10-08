#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include "primary_classes.h"
#include "primary_functions.h"

using namespace std;

int main()
{   
    int n = 10;
    float p = 0.6;
    vector<unique_ptr<EdgeInfo>> edgePtrVec;

    // Initialize the chain
    for (size_t i = 0; i < n; i++)
    {
        edgePtrVec.push_back(make_unique<EdgeInfo>(0., 0));
    }
    
    // Regardless of the scheme, we need to initially simulate the edges

    simulate_path(edgePtrVec, p);
    decohere(edgePtrVec);

    //For heralding all, the failed edges need to retry generating the edge

    for(auto& edge: edgePtrVec) {
        cout << edge->getCapacity() << " " << edge->getAttemptNumber() << endl;   
    }
    cout << endl;

    return 0;
}
