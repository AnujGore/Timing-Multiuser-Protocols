#ifndef PRIMARY_FUNCTIONS
#define PRIMARY_FUNCTIONS

#include "primary_classes.h"
#include <memory>
#include <vector>

using namespace std;

unique_ptr<EdgeInfo> simulate_edge(float p);

void simulate_path(vector<unique_ptr<EdgeInfo>>& this_ptr_vec, float p);

void decohere(const vector<unique_ptr<EdgeInfo>>& this_ptr_vec);

#endif