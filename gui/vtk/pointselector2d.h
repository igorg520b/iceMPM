#ifndef POINTSELECTOR2D_H
#define POINTSELECTOR2D_H

#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleRubberBand2D.h>
#include <vtkActor.h>
#include <vtkNew.h>

#include <functional>


class PointSelector2D : public vtkInteractorStyleRubberBand2D
{
public:
    static PointSelector2D* New();
    vtkTypeMacro(PointSelector2D, vtkInteractorStyleRubberBand2D);

    PointSelector2D();

    std::function<void(double x, double y)> clicked_on_a_point;

    void OnLeftButtonDown() override;
    void OnLeftButtonUp() override {};
private:
    void GetCurrentCoords(double &x, double &y);
};

#endif // POINTSELECTOR2D_H
