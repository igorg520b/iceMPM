#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include <iostream>

#include "point.h"
#include "gridnode.h"

#include <omp.h>


int main(int argc, char *argv[])
{
    std::cout << "num threads " << omp_get_max_threads() << '\n';
#pragma omp parallel
    { std::cout << "thread " <<  omp_get_thread_num() << '\n'; }
    std::cout << std::endl;


    QApplication a(argc, argv);
    QApplication::setApplicationName("iceMPM");
    QApplication::setApplicationVersion("1.0");

    MainWindow w;
    w.resize(1400,900);
//    w.show();
    w.showMaximized();
    return a.exec();
}
