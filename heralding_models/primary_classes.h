#ifndef PRIMARY_CLASSES_H
#define PRIMARY_CLASSES_H

class EdgeInfo 
{
public:
    // Constructor
    EdgeInfo(float foo, int A);

    // Getters
    float getCapacity() const;
    int getAttemptNumber() const;

    void setCapacity(float new_c);

private:
    float F; // Capacity
    int a;   // Attempt number
};



#endif