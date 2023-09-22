#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include <iostream>

#include "point.h"
#include "gridnode.h"

//void test_cuda();

int main(int argc, char *argv[])
{
//    test_cuda();


    QApplication a(argc, argv);
    QApplication::setApplicationName("iceMPM");
    QApplication::setApplicationVersion("1.0");

    MainWindow w;
    w.resize(1400,900);
//    w.show();
    w.showMaximized();
    return a.exec();
}
