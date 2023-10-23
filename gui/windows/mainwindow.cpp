#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::~MainWindow() {delete ui;}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    params = new ParamsWrapper(&model.prms);
    model.Reset();
    worker = new BackgroundWorker(&model);

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);

    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(rubberBand);

    // VTK - scalar bar
//    renderer->AddActor(scalarBar);
    scalarBar->SetMaximumWidthInPixels(130);
    scalarBar->SetBarRatio(0.07);
    scalarBar->SetMaximumHeightInPixels(300);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);

    // property browser
    pbrowser = new ObjectPropertyBrowser(this);

    // splitter
    splitter = new QSplitter(Qt::Orientation::Horizontal);
    splitter->addWidget(pbrowser);
    splitter->addWidget(qt_vtk_widget);
    splitter->setSizes(QList<int>({100, 500}));
    setCentralWidget(splitter);

    // toolbar - combobox
    comboBox_visualizations = new QComboBox();
    ui->toolBar->addWidget(comboBox_visualizations);

    // double spin box
    qdsbLimitLow = new QDoubleSpinBox();
    qdsbLimitHigh = new QDoubleSpinBox();
    qdsbLimitLow->setRange(-1e10, 1e10);
    qdsbLimitHigh->setRange(-1e10, 1e10);
    qdsbLimitLow->setValue(0);
    qdsbLimitHigh->setValue(1e7);
    ui->toolBar->addWidget(qdsbLimitLow);
    ui->toolBar->addWidget(qdsbLimitHigh);

    // slider
    ui->toolBar->addSeparator();
    slider1 = new QSlider(Qt::Horizontal);
    ui->toolBar->addWidget(slider1);
    slider1->setTracking(true);
    slider1->setMinimum(0);
    slider1->setMaximum(0);
    connect(slider1, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    // statusbar
    statusLabel = new QLabel();
    labelElapsedTime = new QLabel();
    labelStepCount = new QLabel();

    QSizePolicy sp;
    const int status_width = 60;
    sp.setHorizontalPolicy(QSizePolicy::Fixed);
    labelStepCount->setSizePolicy(sp);
    labelStepCount->setFixedWidth(status_width);
    labelElapsedTime->setSizePolicy(sp);
    labelElapsedTime->setFixedWidth(status_width);

    ui->statusbar->addWidget(statusLabel);
    ui->statusbar->addPermanentWidget(labelElapsedTime);
    ui->statusbar->addPermanentWidget(labelStepCount);

// anything that includes the Model


    scalarBar->SetLookupTable(representation.hueLut);
    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.actor_grid);
    renderer->AddActor(representation.actor_indenter);
    renderer->AddActor(actorText);

    // text
    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(14);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    actorText->SetDisplayPosition(500, 30);

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [=](int index){ comboboxIndexChanged_visualizations(index); });

    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/cm.ini";
    QFileInfo fi(settingsFileName);

    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;

        vtkCamera* camera = renderer->GetActiveCamera();
        renderer->ResetCamera();
        camera->ParallelProjectionOn();

        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1,1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0],vec[1],vec[2]);
            camera->SetFocalPoint(vec[3],vec[4],vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->Modified();
        }

        qLastDirectory = QDir::currentPath();
        var = settings.value("lastFile");
        if(!var.isNull())
        {
            qLastFileName = var.toString();
            QFileInfo fiLastFile(qLastFileName);
            qLastDirectory = fiLastFile.path();
            if(fiLastFile.exists()) OpenFile(qLastFileName);
        }

        comboBox_visualizations->setCurrentIndex(settings.value("vis_option").toInt());

        var = settings.value("take_screenshots");
        if(!var.isNull())
        {
            ui->actionTake_Screenshots->setChecked(var.toBool());
        }

        var = settings.value("splitter_size_0");
        if(!var.isNull())
        {
            int sz1 = var.toInt();
            int sz2 = settings.value("splitter_size_1").toInt();
            splitter->setSizes(QList<int>({sz1, sz2}));
        }
    }
    else
    {
        cameraReset_triggered();
    }
    QDir pngDir(QDir::currentPath()+ screenshot_directory.c_str());
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ screenshot_directory.c_str());

    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(1); // image quality
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOn(); // read from the back buffer
    writerPNG->SetInputConnection(windowToImageFilter->GetOutputPort());

    connect(ui->action_quit, &QAction::triggered, this, &MainWindow::quit_triggered);
    connect(ui->action_camera_reset, &QAction::triggered, this, &MainWindow::cameraReset_triggered);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::open_triggered);
    connect(ui->actionCreate_Video, &QAction::triggered, this, &MainWindow::createVideo_triggered);
    connect(qdsbLimitLow,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);
    connect(qdsbLimitHigh,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);
    connect(ui->actionScreenshot, &QAction::triggered, this, &MainWindow::screenshot_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);

    representation.model = &model;
    offscreen.model = &model;
    representation.SynchronizeTopology();
    offscreen.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();

    connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
    connect(worker, SIGNAL(stepCompleted()), SLOT(updateGUI()));
}


void MainWindow::showEvent( QShowEvent*)
{
}



void MainWindow::closeEvent( QCloseEvent* event )
{
    quit_triggered();
    event->accept();
}


void MainWindow::quit_triggered()
{
    qDebug() << "MainWindow::quit_triggered() ";
    worker->Finalize();
    // save settings and stop simulation
    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "MainWindow: closing main window; " << settings.fileName();

    double data[10];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    qDebug() << "cam pos " << data[0] << "," << data[1] << "," << data[2];
    qDebug() << "cam focal pt " << data[3] << "," << data[4] << "," << data[5];
    qDebug() << "cam par scale " << data[6];


    QByteArray arr((char*)&data[0], sizeof(double)*10);
    settings.setValue("camData", arr);
    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    if(!qLastFileName.isEmpty()) settings.setValue("lastFile", qLastFileName);
    settings.setValue("take_screenshots", ui->actionTake_Screenshots->isChecked());

    QList<int> szs = splitter->sizes();
    settings.setValue("splitter_size_0", szs[0]);
    settings.setValue("splitter_size_1", szs[1]);
    QApplication::quit();
}



void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);
    scalarBar->SetVisibility(index != 0);
    renderWindow->Render();
}

void MainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);
    camera->SetFocalPoint(0, 0., 0.);
    camera->SetPosition(0.0, 0.0, 50.0);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(2.5);

    camera->Modified();
    renderWindow->Render();
}


void MainWindow::sliderValueChanged(int val)
{
    labelStepCount->setText(QString{"step: %1"}.arg(val));
    GoToStep(val);
}

void MainWindow::open_triggered()
{
    qDebug() << "void MainWindow::open_triggered()";
/*    qLastFileName = QFileDialog::getOpenFileName(this, "Open Binary Export", qLastDirectory, "Bin Files (*.bin)");
    QFileInfo fi(qLastFileName);
    qLastDirectory = fi.path();
    OpenFile(qLastFileName);
    */
}

void MainWindow::OpenFile(QString fileName)
{
    /*
    QFileInfo fi(fileName);
    qBaseFileName = fi.baseName();
    if(qBaseFileName.isEmpty()) return;

    this->setWindowTitle(fileName);
    mesh.LoadFromBin(fileName.toStdString());
    meshRepresentation.SynchronizeTopology();
    GoToStep(1);

    slider1->blockSignals(true);
    slider1->setRange(1, mesh.nFrames);
    slider1->setValue(1);
    slider1->blockSignals(false);

*/
    renderWindow->Render();
}

void MainWindow::createVideo_triggered()
{
/*
    QDir pngDir(QDir::currentPath()+ "/png");
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ "/png");
    QString tentativeDir = QDir::currentPath()+ "/png" + "/" + qBaseFileName;
    QDir dir(tentativeDir);
    if(!dir.exists()) dir.mkdir(tentativeDir);

    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel

    for(int i=1;i<=mesh.nFrames;i++)
    {
        GoToStep(i);
        renderWindow->WaitForCompletion();

        QString outputPath = tentativeDir + "/" + QString::number(i) + ".png";

        windowToImageFilter->Update();
        windowToImageFilter->Modified();

        writerPNG->Modified();
        writerPNG->SetFileName(outputPath.toUtf8().constData());
        writerPNG->Write();
    }
    renderWindow->DoubleBufferOn();

    std::string ffmpegCommand = "ffmpeg -y -r 24 -f image2 -start_number 1 -i \"" + tentativeDir.toStdString() + "/%d.png\" -vframes " +
            std::to_string(mesh.nFrames) +" -vcodec libx264 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -crf 25  -pix_fmt yuv420p "+
        qBaseFileName.toStdString() + ".mp4\n";
    std::system(ffmpegCommand.c_str());
*/
}

void MainWindow::GoToStep(int step)
{
    /*
    mesh.GoToStep(step);
    meshRepresentation.SynchronizeValues();
    updateActorText();
    renderWindow->Render();
    statusLabel->setText(QString{"elems: %1/%2; czs: %3/%4; "}.arg(mesh.nRemovedTris).arg(mesh.elems.size()).arg(mesh.nRemovedCZs).arg(mesh.czs.size()));
    */
}

void MainWindow::limits_changed(double val)
{
    /*
    meshRepresentation.limit_low = qdsbLimitLow->value();
    meshRepresentation.limit_high = qdsbLimitHigh->value();
    meshRepresentation.SynchronizeValues();
    renderWindow->Render();
    */
}




void MainWindow::screenshot_triggered()
{
    int screenshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep;
    QString outputPath = QDir::currentPath()+ screenshot_directory.c_str() + "/" +
            QString::number(screenshot_number).rightJustified(5, '0') + ".png";

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    renderWindow->WaitForCompletion();


    windowToImageFilter->Update();
    windowToImageFilter->Modified();

    writerPNG->Modified();
    writerPNG->SetFileName(outputPath.toUtf8().constData());
    writerPNG->Write();
    renderWindow->DoubleBufferOn();


    offscreen.SynchronizeValues();
    outputPath = QDir::currentPath()+ screenshot_directory.c_str() + "/_" +
            QString::number(screenshot_number).rightJustified(5, '0') + ".png";
    offscreen.SaveScreenshot(outputPath.toStdString());
}



void MainWindow::updateGUI()
{
 //   if(worker->running) statusLabel->setText("simulation is running");
 //   else statusLabel->setText("simulation is stopped");
    labelStepCount->setText(QString::number(model.prms.SimulationStep));
    labelElapsedTime->setText(QString("%1 s").arg(model.prms.SimulationTime,0,'f',3));
    statusLabel->setText(QString("per cycle: %1 ms").arg(model.compute_time_per_cycle,0,'f',3));


    representation.SynchronizeValues();
    renderWindow->Render();

    if(worker->running && ui->actionTake_Screenshots->isChecked()) screenshot_triggered();
    worker->visual_update_requested = false;
    model.UnlockCycleMutex();
}

void MainWindow::simulation_start_pause(bool checked)
{
    if(!worker->running && checked)
    {
        qDebug() << "starting simulation via GUI";
        statusLabel->setText("starting simulation");
        worker->Resume();
    }
    else if(worker->running && !checked)
    {
        qDebug() << "pausing simulation via GUI";
        statusLabel->setText("pausing simulation");
        worker->Pause();
        ui->actionStart_Pause->setEnabled(false);
    }
}

void MainWindow::background_worker_paused()
{
    ui->actionStart_Pause->blockSignals(true);
    ui->actionStart_Pause->setEnabled(true);
    ui->actionStart_Pause->setChecked(false);
    ui->actionStart_Pause->blockSignals(false);
    statusLabel->setText("simulation stopped");
}
