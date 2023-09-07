#include <omp.h>

#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_sinks.h"
#include <omp.h>
#include <iostream>

#include "point.h"
#include "gridnode.h"

int main(int argc, char *argv[])
{
    spdlog::info("num threads {}", omp_get_max_threads());
    int nthreads, tid;
#pragma omp parallel
    { spdlog::info("thread {}", omp_get_thread_num()); }
    std::cout << std::endl;
    spdlog::info("sizeof(Point) = {}", sizeof(icy::Point));
    spdlog::info("sizeof(GridNode) = {}", sizeof(icy::GridNode));

    QApplication a(argc, argv);
    QApplication::setApplicationName("iceMPM");
    QApplication::setApplicationVersion("1.0");

    MainWindow w;
    w.resize(1400,900);
//    w.show();
    w.showMaximized();
    return a.exec();
}
