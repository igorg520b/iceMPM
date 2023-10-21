#include "backgroundworker.h"

BackgroundWorker::BackgroundWorker(icy::Model *controller_) : controller(controller_)
{
    this->start();
}

// resume the worker thread
void BackgroundWorker::Resume()
{
    controller->Prepare();
    condition.wakeOne();
}

// cancel current step and pause the worker thread
void BackgroundWorker::Pause()
{
    if(!running) return;
    timeToPause = true;
    controller->RequestAbort();
}

// exit the worker thread
void BackgroundWorker::Finalize()
{
    controller->RequestAbort();
    kill=true;
    condition.wakeOne();
    bool result = wait();
    qDebug() << "BackgroundWorker::Finalize() done" << result;
}

void BackgroundWorker::run()
{
    controller->Prepare();
    while(!kill)
    {
        if (timeToPause)
        {
            timeToPause = false;
            running = false;
            Q_EMIT workerPaused();
            mutex.lock();
            condition.wait(&mutex);
            mutex.unlock();
            running = true;
        }
        if(kill) break;

        bool result = controller->Step();
        if(!result) timeToPause = true;
        if(controller->isTimeToUpdate() && !visual_update_requested)
        {
            visual_update_requested = true;
            Q_EMIT stepCompleted();
        }
    }
    qDebug() << "BackgroundWorker::run() terminated";
}
