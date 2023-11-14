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
#include <vtkFloatArray.h>
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
#include <vtkStructuredGrid.h>

#include <vtkRegularPolygonSource.h>
#include <vtkCylinderSource.h>



namespace icy { class VisualRepresentation; class Model;}

class icy::VisualRepresentation : public QObject
{
    Q_OBJECT

public:
    VisualRepresentation();

    icy::Model *model;

    double value_range = 0.01;

    enum VisOpt { none, NACC_alpha, NACC_case };
    Q_ENUM(VisOpt)

    void SynchronizeValues();
    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);  // invoked from GUI/main thread

    vtkNew<vtkLookupTable> hueLut, lutMPM;
    vtkNew<vtkActor> actor_points;
    vtkNew<vtkActor> actor_grid;
    vtkNew<vtkActor> actor_indenter;

private:
    VisOpt VisualizingVariable = VisOpt::none;

    vtkNew<vtkLookupTable> hueLut_pastel, hueLut_four;

    // indenter
    vtkNew<vtkRegularPolygonSource> indenterSource;
    vtkNew<vtkPolyDataMapper> indenterMapper;

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkCellArray> points_cells;
    vtkNew<vtkVertexGlyphFilter> points_filter;
    vtkNew<vtkFloatArray> visualized_values;

    // background grid
    vtkNew<vtkStructuredGrid> structuredGrid;
    vtkNew<vtkDataSetMapper> grid_mapper;
    vtkNew<vtkPoints> grid_points;

    static constexpr float lutArrayMPMColors[101][3] =
    {{0.25098, 0.556863, 0.756863}, {0.245961, 0.547294,
      0.749961}, {0.240941, 0.537725, 0.743059}, {0.235922, 0.528157,
      0.736157}, {0.230902, 0.518588, 0.729255}, {0.225882, 0.50902,
      0.722353}, {0.220863, 0.499451, 0.715451}, {0.215843, 0.489882,
      0.708549}, {0.210824, 0.480314, 0.701647}, {0.205804, 0.470745,
      0.694745}, {0.200784, 0.461176, 0.687843}, {0.195765, 0.451608,
      0.680941}, {0.190745, 0.442039, 0.674039}, {0.185725, 0.432471,
      0.667137}, {0.180706, 0.422902, 0.660235}, {0.175686, 0.413333,
      0.653333}, {0.170667, 0.403765, 0.646431}, {0.165647, 0.394196,
      0.639529}, {0.160627, 0.384627, 0.632627}, {0.155608, 0.375059,
      0.625725}, {0.150588, 0.36549, 0.618824}, {0.145569, 0.355922,
      0.611922}, {0.140549, 0.346353, 0.60502}, {0.135529, 0.336784,
      0.598118}, {0.13051, 0.327216, 0.591216}, {0.12549, 0.317647,
      0.584314}, {0.120471, 0.304941, 0.560941}, {0.115451, 0.292235,
      0.537569}, {0.110431, 0.279529, 0.514196}, {0.105412, 0.266824,
      0.490824}, {0.100392, 0.254118, 0.467451}, {0.0953725, 0.241412,
      0.444078}, {0.0903529, 0.228706, 0.420706}, {0.0853333, 0.216,
      0.397333}, {0.0803137, 0.203294, 0.373961}, {0.0752941, 0.190588,
      0.350588}, {0.0702745, 0.177882, 0.327216}, {0.0652549, 0.165176,
      0.303843}, {0.0602353, 0.152471, 0.280471}, {0.0552157, 0.139765,
      0.257098}, {0.0501961, 0.127059, 0.233725}, {0.0451765, 0.114353,
      0.210353}, {0.0401569, 0.101647, 0.18698}, {0.0351373, 0.0889412,
      0.163608}, {0.0301176, 0.0762353, 0.140235}, {0.025098, 0.0635294,
      0.116863}, {0.0200784, 0.0508235, 0.0934902}, {0.0150588, 0.0381176,
       0.0701176}, {0.0100392, 0.0254118, 0.0467451}, {0.00501961,
      0.0127059, 0.0233725}, {0., 0., 0.}, {0.0180392, 0.00329412,
      0.00564706}, {0.0360784, 0.00658824, 0.0112941}, {0.0541176,
      0.00988235, 0.0169412}, {0.0721569, 0.0131765,
      0.0225882}, {0.0901961, 0.0164706, 0.0282353}, {0.108235, 0.0197647,
       0.0338824}, {0.126275, 0.0230588, 0.0395294}, {0.144314, 0.0263529,
       0.0451765}, {0.162353, 0.0296471, 0.0508235}, {0.180392, 0.0329412,
       0.0564706}, {0.198431, 0.0362353, 0.0621176}, {0.216471, 0.0395294,
       0.0677647}, {0.23451, 0.0428235, 0.0734118}, {0.252549, 0.0461176,
      0.0790588}, {0.270588, 0.0494118, 0.0847059}, {0.288627, 0.0527059,
      0.0903529}, {0.306667, 0.056, 0.096}, {0.324706, 0.0592941,
      0.101647}, {0.342745, 0.0625882, 0.107294}, {0.360784, 0.0658824,
      0.112941}, {0.378824, 0.0691765, 0.118588}, {0.396863, 0.0724706,
      0.124235}, {0.414902, 0.0757647, 0.129882}, {0.432941, 0.0790588,
      0.135529}, {0.45098, 0.0823529, 0.141176}, {0.462118, 0.0840784,
      0.142745}, {0.473255, 0.0858039, 0.144314}, {0.484392, 0.0875294,
      0.145882}, {0.495529, 0.0892549, 0.147451}, {0.506667, 0.0909804,
      0.14902}, {0.517804, 0.0927059, 0.150588}, {0.528941, 0.0944314,
      0.152157}, {0.540078, 0.0961569, 0.153725}, {0.551216, 0.0978824,
      0.155294}, {0.562353, 0.0996078, 0.156863}, {0.57349, 0.101333,
      0.158431}, {0.584627, 0.103059, 0.16}, {0.595765, 0.104784,
      0.161569}, {0.606902, 0.10651, 0.163137}, {0.618039, 0.108235,
      0.164706}, {0.629176, 0.109961, 0.166275}, {0.640314, 0.111686,
      0.167843}, {0.651451, 0.113412, 0.169412}, {0.662588, 0.115137,
      0.17098}, {0.673725, 0.116863, 0.172549}, {0.684863, 0.118588,
      0.174118}, {0.696, 0.120314, 0.175686}, {0.707137, 0.122039,
      0.177255}, {0.718275, 0.123765, 0.178824}, {0.729412, 0.12549,
      0.180392}};

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
