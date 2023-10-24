#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>

#include "model.h"
#include "vtkoffscreen.h"


std::atomic<bool> stop = false;
std::string screenshot_directory = "cm_screenshots";

icy::Model model;
icy::VTKOffscreen offscreen;

void simulation_loop()
{
    bool result = true;
    while(!stop && result)
    {
        model.Step();
//        std::cout << "loop \n";
//        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}


int main()
{
    model.Reset();
    model.Prepare();
//    model.gpu.test();
    offscreen.model = &model;
    offscreen.SynchronizeTopology();

    model.gpu.transfer_completion_callback = [&](){
        if(model.prms.SimulationStep % (model.prms.UpdateEveryNthStep * model.prms.SaveEveryNthUpdate)) return;
        int screenshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep / model.prms.SaveEveryNthUpdate;
        model.FinalizeDataTransfer();


        std::string outputPath = screenshot_directory + "/" + std::to_string(screenshot_number) + ".png";
        std::cout << "screenshot " << outputPath << "\n";

        offscreen.SynchronizeValues();
        offscreen.SaveScreenshot(outputPath);
        model.UnlockCycleMutex();
    };

    // ensure that the folder exists
    std::filesystem::path outputFolder(screenshot_directory);
    std::filesystem::create_directory(outputFolder);
    model.Step();


    std::thread t(simulation_loop);

    char c = 0;
    do
    {
        cin.get(c);
        std::cout << "read character " << c << '\n';
    }while(c!= 25 && c!= 'x');

    stop = true;
    std::cout << "terminating...\n";
    t.join();

    model.gpu.test();
    std::cout << "cm done\n";



    return 0;
}
