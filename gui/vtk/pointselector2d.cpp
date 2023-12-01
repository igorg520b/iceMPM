#include "pointselector2d.h"
#include <vtkObjectFactory.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>

vtkStandardNewMacro(PointSelector2D);

PointSelector2D::PointSelector2D()
{

}

void PointSelector2D::GetCurrentCoords(double &x, double &y)
{
    vtkRenderWindowInteractor* rwi = this->GetInteractor();
    int curPt[] = { 0, 0 };
    rwi->GetEventPosition(curPt);

    vtkRenderer* renderer = rwi->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
    vtkCamera* camera = renderer->GetActiveCamera();

    double camera_parallelScale = camera->GetParallelScale();
    int* renderer_getSize = renderer->GetSize();
    int renderer_getSize1 = renderer_getSize[1];

    double camPosX, camPosY, camPosZ;
    camera->GetPosition(camPosX,camPosY,camPosZ);

    double lastScale = 2.0 *  camera_parallelScale / renderer_getSize1;

    x = lastScale * (curPt[0]-renderer_getSize[0]/2)+camPosX;
    y = lastScale * (curPt[1]-renderer_getSize[1]/2)+camPosY;
}

void PointSelector2D::OnLeftButtonDown()
{
    double X, Y;
    GetCurrentCoords(X, Y);
    clicked_on_a_point(X,Y);

}

