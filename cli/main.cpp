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
std::thread screenshot_thread;

icy::Model model;
icy::VTKOffscreen offscreen;

void simulation_loop()
{
    bool result;
    do
    {
        result = model.Step();
    } while(!stop && result);
}


int main()
{
    model.Reset();
    model.Prepare();
    offscreen.model = &model;
    offscreen.SynchronizeTopology();

    model.gpu.transfer_completion_callback = [&](){
        if(model.prms.SimulationStep % (model.prms.UpdateEveryNthStep * model.prms.SaveEveryNthUpdate)) return;

        if(screenshot_thread.joinable()) screenshot_thread.join();
        screenshot_thread = std::thread([&](){
        int screenshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep / model.prms.SaveEveryNthUpdate;
        std::string outputPath = screenshot_directory + "/" + std::to_string(screenshot_number) + ".png";
        std::cout << "screenshot " << outputPath << "\n";
        model.UnlockCycleMutex();
        if(stop) { std::cout << "screenshot aborted\n"; return; }
        model.FinalizeDataTransfer();
        offscreen.SynchronizeValues();
        offscreen.SaveScreenshot(outputPath);
        std::cout << "screenshot done\n";
        }
        );
    };

    // ensure that the folder exists
    std::filesystem::path outputFolder(screenshot_directory);
    std::filesystem::create_directory(outputFolder);

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
    model.gpu.synchronize();
    screenshot_thread.join();

    std::cout << "cm done\n";

    return 0;
}
