#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <string>
#include <H5Cpp.h>

namespace icy {class SnapshotManager; class Model;}


class icy::SnapshotManager
{
public:
    icy::Model *model;

    void SaveSnapshot(std::string directory);
    void ReadSnapshot(std::string fileName); // return file number
    void LoadRawPoints(std::string fileName);
    void SaveParametersAsAttributes(H5::DataSet &dataset);
};

#endif // SNAPSHOTWRITER_H
