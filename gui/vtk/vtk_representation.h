#ifndef MESHREPRESENTATION_H
#define MESHREPRESENTATION_H

#include <QObject>

#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellType.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkNamedColors.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkVertexGlyphFilter.h>

#include <vtkRegularPolygonSource.h>
#include <vtkCylinderSource.h> // testing



namespace icy { class VisualRepresentation; class Model;}

class icy::VisualRepresentation : public QObject
{
    Q_OBJECT

public:
    VisualRepresentation();
    ~VisualRepresentation() = default;

    icy::Model *model;

    bool showPoints = true;
    bool showGrid = false;
    double limit_low = 0, limit_high = 1e7; // limits for color scale

    enum VisOpt { none, velocity, stress_mises, stress_maxP, stress_minP};
    Q_ENUM(VisOpt)

    void SynchronizeValues();
    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);  // invoked from GUI/main thread

    vtkNew<vtkLookupTable> hueLut;
    vtkNew<vtkActor> actor_points;
    vtkNew<vtkActor> actor_grid;
    vtkNew<vtkActor> actor_cylinder; // testing

private:
    VisOpt VisualizingVariable = VisOpt::none;

    vtkNew<vtkLookupTable> hueLut_pastel;

    // testing - cylinder
    vtkNew<vtkCylinderSource> cylinderSource;
    vtkNew<vtkPolyDataMapper> cylinderMapper;

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkCellArray> points_cells;
    vtkNew<vtkVertexGlyphFilter> points_filter;


    static constexpr float lutArrayTemperatureAdj[51][3] =
        {{0.770938, 0.951263, 0.985716}, {0.788065, 0.959241, 0.986878},
         {0.805191, 0.96722, 0.98804}, {0.822318, 0.975199, 0.989202},
         {0.839445, 0.983178, 0.990364}, {0.856572, 0.991157, 0.991526},
         {0.872644, 0.995552, 0.98386}, {0.887397, 0.995466, 0.965157},
         {0.902149, 0.99538, 0.946454}, {0.916902, 0.995294, 0.927751},
         {0.931655, 0.995208, 0.909049}, {0.946408, 0.995123, 0.890346},
         {0.961161, 0.995037, 0.871643}, {0.975913, 0.994951, 0.85294},
         {0.990666, 0.994865, 0.834237}, {0.996257, 0.991758, 0.815237},
         {0.994518, 0.986234, 0.795999}, {0.992779, 0.98071, 0.77676},
         {0.99104, 0.975186, 0.757522}, {0.989301, 0.969662, 0.738283},
         {0.987562, 0.964138, 0.719045}, {0.985823, 0.958614, 0.699807},
         {0.984084, 0.953089, 0.680568}, {0.982345, 0.947565, 0.66133},
         {0.97888, 0.936201, 0.641773}, {0.974552, 0.921917, 0.622058},
         {0.970225, 0.907633, 0.602342}, {0.965897, 0.893348, 0.582626},
         {0.961569, 0.879064, 0.562911}, {0.957242, 0.86478, 0.543195},
         {0.952914, 0.850496, 0.52348}, {0.948586, 0.836212, 0.503764},
         {0.944259, 0.821927, 0.484048}, {0.939066, 0.801586, 0.464871},
         {0.933626, 0.779513, 0.445847}, {0.928186, 0.757441, 0.426823},
         {0.922746, 0.735368, 0.4078}, {0.917306, 0.713296, 0.388776},
         {0.911866, 0.691223, 0.369752}, {0.906426, 0.669151, 0.350728},
         {0.900986, 0.647078, 0.331704}, {0.895546, 0.625006, 0.312681},
         {0.889975, 0.597251, 0.298625}, {0.884388, 0.568785, 0.285191},
         {0.8788, 0.54032, 0.271756}, {0.873212, 0.511855, 0.258322},
         {0.867625, 0.483389, 0.244888}, {0.862037, 0.454924, 0.231453},
         {0.856449, 0.426459, 0.218019}, {0.850862, 0.397993, 0.204584},
         {0.845274, 0.369528, 0.19115}};

    static constexpr float lutArrayPastel[40][3] = {
        {196/255.0,226/255.0,252/255.0}, // 0
        {136/255.0,119/255.0,187/255.0},
        {190/255.0,125/255.0,183/255.0},
        {243/255.0,150/255.0,168/255.0},
        {248/255.0,187/255.0,133/255.0},
        {156/255.0,215/255.0,125/255.0},
        {198/255.0,209/255.0,143/255.0},
        {129/255.0,203/255.0,178/255.0},
        {114/255.0,167/255.0,219/255.0},
        {224/255.0,116/255.0,129/255.0},
        {215/255.0,201/255.0,226/255.0},  // 10
        {245/255.0,212/255.0,229/255.0},
        {240/255.0,207/255.0,188/255.0},
        {247/255.0,247/255.0,213/255.0},
        {197/255.0,220/255.0,204/255.0},
        {198/255.0,207/255.0,180/255.0},
        {135/255.0,198/255.0,233/255.0},
        {179/255.0,188/255.0,221/255.0},
        {241/255.0,200/255.0,206/255.0},
        {145/255.0,217/255.0,213/255.0},
        {166/255.0,200/255.0,166/255.0},  // 20
        {199/255.0,230/255.0,186/255.0},
        {252/255.0,246/255.0,158/255.0},
        {250/255.0,178/255.0,140/255.0},
        {225/255.0,164/255.0,195/255.0},
        {196/255.0,160/255.0,208/255.0},
        {145/255.0,158/255.0,203/255.0},
        {149/255.0,217/255.0,230/255.0},
        {193/255.0,220/255.0,203/255.0},
        {159/255.0,220/255.0,163/255.0},
        {235/255.0,233/255.0,184/255.0},  // 30
        {237/255.0,176/255.0,145/255.0},
        {231/255.0,187/255.0,212/255.0},
        {209/255.0,183/255.0,222/255.0},
        {228/255.0,144/255.0,159/255.0},
        {147/255.0,185/255.0,222/255.0},  // 35
        {158/255.0,213/255.0,194/255.0},  // 36
        {177/255.0,201/255.0,139/255.0},  // 37
        {165/255.0,222/255.0,141/255.0},  // 38
        {244/255.0,154/255.0,154/255.0}   // 39
    };
};
#endif
