#include "primary_functions.h"
#include <memory.h>
#include <cstdlib>
#include <cmath>

using namespace std;

unique_ptr<EdgeInfo> simulate_edge(float p){
    auto success = ((double)rand()) / RAND_MAX;
    return make_unique<EdgeInfo>((success < p) ? 1. : 0., 1);
}

unique_ptr<EdgeInfo> decohere_edge(unique_ptr<EdgeInfo> this_ptr){
    auto edge_F = this_ptr->getCapacity();
    auto new_edge_F = edge_F*exp(0.1);
    this_ptr->setCapacity(new_edge_F);
    return this_ptr;
}

void simulate_path(vector<unique_ptr<EdgeInfo>>& this_ptr_vec, float p)
{
    for (size_t i = 0; i < this_ptr_vec.size(); i++)
    {
        this_ptr_vec[i] = simulate_edge(p);
    }
    
}

void decohere(const vector<unique_ptr<EdgeInfo>>& this_ptr_vec)
{
    for (size_t i = 0; i < this_ptr_vec.size(); i++)
    {   
        this_ptr_vec[i] = decohere_edge(this_ptr_vec[i]);
        // decohere_edge(this_ptr_vec[i]);
    }

}