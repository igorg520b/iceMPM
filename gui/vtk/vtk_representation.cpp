#include "vtk_representation.h"
#include "model.h"
#include "parameters_sim.h"
#include <omp.h>

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

    nLut = sizeof lutArrayMPMColors / sizeof lutArrayMPMColors[0];
    lutMPM->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        lutMPM->SetTableValue(i, lutArrayMPMColors[i][0],
                              lutArrayMPMColors[i][1],
                              lutArrayMPMColors[i][2], 1.0);

    hueLut_four->SetNumberOfColors(4);
    hueLut_four->SetTableValue(0, 0.3, 0.3, 0.3);
    hueLut_four->SetTableValue(1, 1.0, 0, 0);
    hueLut_four->SetTableValue(2, 0, 1.0, 0);
    hueLut_four->SetTableValue(3, 0, 0, 1.0);
    hueLut_four->SetTableRange(0,3);


    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);

    indenterSource->GeneratePolygonOff();
    indenterSource->SetNumberOfSides(50);

    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);
    actor_indenter->GetProperty()->LightingOff();
    actor_indenter->GetProperty()->EdgeVisibilityOn();
    actor_indenter->GetProperty()->VertexVisibilityOff();
    actor_indenter->GetProperty()->SetColor(0.1,0.1,0.1);
    actor_indenter->GetProperty()->SetEdgeColor(90.0/255.0, 90.0/255.0, 97.0/255.0);
    actor_indenter->GetProperty()->ShadingOff();
    actor_indenter->GetProperty()->SetInterpolationToFlat();
    actor_indenter->PickableOff();
    actor_indenter->GetProperty()->SetLineWidth(3);


    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(visualized_values);

    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    points_mapper->UseLookupTableScalarRangeOn();
    points_mapper->SetLookupTable(lutMPM);

    visualized_values->SetName("visualized_values");

    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(0,0,0);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();


    grid_mapper->SetInputData(structuredGrid);
    grid_mapper->SetLookupTable(hueLut);

//    mapper_structuredGrid->SetScalarRange(0, dataSize - 1);
//    mapper_structuredGrid->ScalarVisibilityOn();

    actor_grid->SetMapper(grid_mapper);
    actor_grid->GetProperty()->SetEdgeVisibility(true);
    actor_grid->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_grid->GetProperty()->LightingOff();
    actor_grid->GetProperty()->ShadingOff();
    actor_grid->GetProperty()->SetInterpolationToFlat();
    actor_grid->PickableOff();
    actor_grid->GetProperty()->SetColor(0.95,0.95,0.95);
}



void icy::VisualRepresentation::SynchronizeTopology()
{
    points->SetNumberOfPoints(model->points.size());
    visualized_values->SetNumberOfValues(model->points.size());
    points_polydata->GetPointData()->SetActiveScalars("visualized_values");
    points_mapper->ScalarVisibilityOn();
    points_mapper->SetColorModeToMapScalars();

    SynchronizeValues();

    // structured grid
    int &gx = model->prms.GridX;
    int &gy = model->prms.GridY;
    real &h = model->prms.cellsize;
    structuredGrid->SetDimensions(model->prms.GridX, model->prms.GridY, 1);

    grid_points->SetNumberOfPoints(gx*gy);
    for(int idx_y=0; idx_y<gy; idx_y++)
        for(int idx_x=0; idx_x<gx; idx_x++)
        {
            float x = idx_x * h;
            float y = idx_y * h;
            double pt_pos[3] {x, y, -1.0};
            grid_points->SetPoint((vtkIdType)(idx_x+idx_y*gx), pt_pos);
        }

    structuredGrid->SetPoints(grid_points);

    // indenter
    indenterSource->SetRadius(model->prms.IndDiameter/2.f);
}


void icy::VisualRepresentation::SynchronizeValues()
{
    const real &alpha0 = model->prms.NACC_alpha;
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);

    model->hostside_data_update_mutex.lock();
//#pragma omp parallel
    for(int i=0;i<model->points.size();i++)
    {
        const icy::Point &p = model->points[i];
        double x[3] {p.pos[0], p.pos[1], 0};
        points->SetPoint((vtkIdType)i, x);
      }

    if(VisualizingVariable == 0)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, 0);
    }
    else if(VisualizingVariable == 1)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, exp(model->points[i].NACC_alpha_p));
    }
    else if(VisualizingVariable == 2)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].q);
    }

    model->hostside_data_update_mutex.unlock();

    double centerVal = 0;
    double range = std::pow(10,value_range);

    if(VisualizingVariable == 2)
    {
        points_mapper->SetLookupTable(hueLut_four);
    }
    else
    {
        points_mapper->SetLookupTable(lutMPM);
    }


    if(VisualizingVariable == 1)
    {
        centerVal = exp(alpha0);
    }
//    float minmax[2];
//    visualized_values->GetValueRange(minmax);
    lutMPM->SetTableRange(centerVal-range, centerVal+range);

    points->Modified();
    visualized_values->Modified();
    points_filter->Update();
    indenterSource->SetCenter(model->indenter_x, model->indenter_y, 1);
}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}
