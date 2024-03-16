#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>


void icy::Point::Reset()
{
    Fe.setIdentity();
    velocity.setZero();
    Bp.setZero();
    q = 0;
    Jp_inv = 1;
}

void icy::Point::TransferToBuffer(real *buffer, const int pitch, const int point_index) const
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    ptr_intact[point_index] = q;

    short* ptr_grain = (short*)(&ptr_intact[pitch]);
    ptr_grain[point_index] = grain;

    buffer[point_index + pitch*icy::SimParams::idx_Jp_inv] = Jp_inv;

    for(int i=0; i<icy::SimParams::dim; i++)
    {
        buffer[point_index + pitch*(icy::SimParams::posx+i)] = pos[i];
        buffer[point_index + pitch*(icy::SimParams::velx+i)] = velocity[i];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            buffer[point_index + pitch*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)] = Fe(i,j);
            buffer[point_index + pitch*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)] = Bp(i,j);
        }
    }
}

Vector2r icy::Point::getPos(const real *buffer, const int pitch, const int point_index)
{
    Vector2r result;
    for(int i=0; i<icy::SimParams::dim;i++) result[i] = buffer[point_index + pitch*(icy::SimParams::posx+i)];
    return result;
}

char icy::Point::getQ(const real *buffer, const int pitch, const int point_index)
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    return ptr_intact[point_index];
}

double icy::Point::getJp_inv(const real *buffer, const int pitch, const int point_index)
{
    return buffer[point_index + pitch*icy::SimParams::idx_Jp_inv];
}

short icy::Point::getGrain(const real *buffer, const int pitch, const int point_index)
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    short* ptr_grain = (short*)(&ptr_intact[pitch]);
    short grain = ptr_grain[point_index];
    return grain;
}


/*
void PullFromBuffer(const real *buffer, const int pitch, const int point_index);

static Vector2r getVelocity(const real *buffer, const int pitch, const int point_index);
static void setPos_Q_Jpinv(Eigen::Vector2f _pos, float _Jp_inv, real *buff, const int pitch, const int pt_idx);
*/
