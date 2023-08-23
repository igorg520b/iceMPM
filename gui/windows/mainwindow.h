#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QSizePolicy>
#include <QPushButton>
#include <QSplitter>
#include <QLabel>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QProgressBar>
#include <QMenu>
#include <QList>
#include <QDebug>
#include <QComboBox>
#include <QMetaEnum>
#include <QDir>
#include <QString>
#include <QCheckBox>
#include <QFile>
#include <QTextStream>
#include <QIODevice>
#include <QSettings>
#include <QDoubleSpinBox>
#include <QFileInfo>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkTextActor.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <vtkInteractorStyleRubberBand2D.h>

#include "preferences_gui.h"
#include "vtk_representation.h"
#include "model.h"
#include "backgroundworker.h"
#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <iostream>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::MainWindow *ui;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent( QCloseEvent* event ) override;

private Q_SLOTS:
    void quit_triggered();

    void background_worker_paused();
    void updateGUI();   // when simulation is started/stopped or when a step is advanced

    void simulation_start_pause(bool checked);

    void sliderValueChanged(int val);
    void comboboxIndexChanged_visualizations(int index);

    void cameraReset_triggered();
    void open_triggered();
    void createVideo_triggered();
    void screenshot_triggered();
    void limits_changed(double val);

private:
    void updateActorText();
    BackgroundWorker *worker;
    icy::Model model;
    icy::VisualRepresentation representation;

    QString settingsFileName;       // includes current dir
    QLabel *statusLabel;                    // statusbar
    QLabel *labelElapsedTime;
    QLabel *labelStepCount;
    QComboBox *comboBox_visualizations;
    QSlider *slider1;
    QDoubleSpinBox *qdsbLimitLow, *qdsbLimitHigh;   // high and low limits for value scale

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkInteractorStyleRubberBand2D> rubberBand;
    vtkNew<vtkTextActor> actorText;

    // other
    void OpenFile(QString fileName);
    void GoToStep(int step);
    QString qLastFileName, qBaseFileName, qLastDirectory;
    bool replayMode = false;
    int replayFrame;

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkPNGWriter> writerPNG;

    void save_cam_pos(QString str);
    void load_cam_pos(QString str);
};
#endif // MAINWINDOW_H
