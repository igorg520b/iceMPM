#include "vtk_representation.h"
#include "model.h"

#include "spdlog/spdlog.h"

icy::VisualRepresentation::VisualRepresentation()
{
    int nLut = sizeof lutArrayTemperatureAdj / sizeof lutArrayTemperatureAdj[0];
    hueLut->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut->SetTableValue(i, lutArrayTemperatureAdj[i][0],
                              lutArrayTemperatureAdj[i][1],
                              lutArrayTemperatureAdj[i][2], 1.0);

    nLut = sizeof lutArrayPastel / sizeof lutArrayPastel[0];
    hueLut_pastel->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_pastel->SetTableValue(i, lutArrayPastel[i][0],
                              lutArrayPastel[i][1],
                              lutArrayPastel[i][2], 1.0);
    hueLut_pastel->SetTableRange(0,39);


    cylinderMapper->SetInputConnection(cylinderSource->GetOutputPort());
    actor_cylinder->SetMapper(cylinderMapper);


    points_polydata->SetPoints(points);
    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
//    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(0,0,0);


}



void icy::VisualRepresentation::SynchronizeTopology()
{
    points->SetNumberOfPoints(model->points.size());
    SynchronizeValues();
    spdlog::info("void icy::MeshRepresentation::SynchronizeTopology() done");
}


void icy::VisualRepresentation::SynchronizeValues()
{
    for(int i=0;i<model->points.size();i++)
    {
        icy::Point &p = model->points[i];
        double x[3] {p.pos[0], p.pos[1], 0};
        points->SetPoint((vtkIdType)i, x);
    }
//    points->Modified();
    points_filter->Update();
}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{

    spdlog::info("icy::Model::ChangeVisualizationOption {}", option);
    VisualizingVariable = (VisOpt)option;

    SynchronizeTopology();
}
