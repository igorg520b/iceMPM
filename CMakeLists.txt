cmake_minimum_required(VERSION 3.20)

project(iceMPM LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

if(UNIX)
    find_package(spdlog REQUIRED)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
endif()

find_package(VTK 9.0 HINTS "C:\\libraries\\vtk-build\\lib\\cmake\\vtk-9.1" REQUIRED COMPONENTS
  CommonColor
  CommonCore
  FiltersSources
  FiltersGeometry
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  ViewsQt
  IOGeometry
  RenderingAnnotation
  RenderingLabel
  IOImage
  QUIET
)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_compile_definitions(QT_NO_KEYWORDS)

add_executable(iceMPM
    gui/main.cpp

    gui/windows/mainwindow.cpp
    gui/windows/mainwindow.h
    gui/windows/mainwindow.ui
    icons.qrc

    gui/vtk/vtk_representation.cpp
    gui/vtk/vtk_representation.h

    gui/backgroundworker.cpp
    gui/backgroundworker.h

    gui/preferences_gui.h
    gui/modelcontrollerinterface.h

    simulation/parameters_sim.h
    simulation/model.cpp
    simulation/model.h
    simulation/point.h
    simulation/point.cpp
    simulation/gridnode.h
    simulation/gridnode.cpp
)

target_include_directories(iceMPM PRIVATE gui geometry gui/windows gui/vtk simulation)

target_link_libraries(iceMPM PRIVATE Qt5::Widgets Qt5::Core)

# OpenMP
if(WIN32)
    target_compile_options(iceMPM PRIVATE /openmp)
else()
    target_compile_options(iceMPM PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(iceMPM PRIVATE gomp)
endif()

# spdlog
if (WIN32)
    target_link_libraries(iceMPM PRIVATE spdlog)
    target_include_directories(iceMPM PRIVATE "C:\\libraries\\spdlog\\include")
    target_link_directories(iceMPM PRIVATE "C:\\libraries\\spdlog-build\\Release")
else()
    target_link_libraries(iceMPM PRIVATE spdlog::spdlog)
endif()

# VTK
target_link_libraries(iceMPM PRIVATE ${VTK_LIBRARIES})


# Eigen
find_package(Eigen3 REQUIRED NO_MODULE HINTS "C:\\libraries\\eigen-build")
target_link_libraries(iceMPM PRIVATE Eigen3::Eigen)

#PBrowser
if(WIN32)
    target_link_directories(iceMPM PRIVATE "C:\\libraries\\pbrowser-build")
    target_include_directories(iceMPM PRIVATE "C:\\libraries\\PBrowser")
    target_include_directories(iceMPM PRIVATE "C:\\libraries\\PBrowser\\propertybrowser")
    target_include_directories(iceMPM PRIVATE "C:\\libraries\\PBrowser\\objectpropertybrowser")
else()
    target_link_directories(iceMPM PRIVATE "")
    target_include_directories(iceMPM PRIVATE /home/s2/Projects-Qt/PBrowser/objectpropertybrowser)
    target_include_directories(iceMPM PRIVATE /home/s2/Projects-Qt/PBrowser/propertybrowser)
    target_include_directories(iceMPM PRIVATE /home/s2/Projects-Qt/PBrowser)
endif()
target_link_libraries(iceMPM PRIVATE PBrowser)
