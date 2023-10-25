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

}


int main(int argc, char** argv)
{
    // parse options
    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");

    options.add_options()
        ("f,file", "Configuration file", cxxopts::value<std::string>())
        ;

    auto option_parse_result = options.parse(argc, argv);

    if (option_parse_result.count("file"))
    {
        std::string params_file = option_parse_result["file"].as<std::string>();
        model.prms.ParseFile(params_file, screenshot_directory);
    }

    // initialize the model
    model.Reset();
    offscreen.model = &model;
    offscreen.SynchronizeTopology();

    // what to do once the data is available
    model.gpu.transfer_completion_callback = [&](){
        if(screenshot_thread.joinable()) screenshot_thread.join();
        screenshot_thread = std::thread([&](){
        int screenshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep;
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

    std::thread t([&](){
        bool result;
        do
        {
            result = model.Step();
        } while(!stop && result);
    });

/*    char c = 0;
    do
    {
        cin.get(c);
        std::cout << "read character " << c << '\n';
    }while(c!= 25 && c!= 'x');

    stop = true;
    std::cout << "terminating...\n";
*/
    t.join();
    model.gpu.synchronize();
    screenshot_thread.join();

    std::cout << "cm done\n";

    return 0;
}
